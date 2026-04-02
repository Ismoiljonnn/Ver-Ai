from flask import Flask, request, jsonify, send_from_directory, session, redirect
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import os
from dotenv import load_dotenv
from supabase import create_client
from groq import Groq

load_dotenv()

app = Flask(__name__, static_folder='public')
app.secret_key = os.environ.get('SECRET_KEY')

# ── Supabase setup ──────────────────────────────────────────────
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# ── API Keys ────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GOOGLE_FACTCHECK_API_KEY = os.environ.get('GOOGLE_FACTCHECK_API_KEY')

# ── Groq client ─────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ── Auth decorator ──────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'access_token' not in session:
            return jsonify({'error': 'Avtorizatsiya talab qilinadi'}), 401
        return f(*args, **kwargs)
    return decorated


# ── Pages ───────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('public', 'index.html')


@app.route('/auth')
@app.route('/auth.html')
def auth_page():
    return send_from_directory('public', 'auth.html')


@app.route('/languages')
@app.route('/languages.html')
def languages_page():
    return send_from_directory('public', 'languages.html')


@app.route('/feed')
@app.route('/feed.html')
def feed_page():
    return send_from_directory('public', 'feed.html')


# ── Auth API ────────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    if not supabase:
        return jsonify({'error': 'Supabase sozlanmagan'}), 500

    data = request.get_json()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not email or not password:
        return jsonify({'error': 'Email va parol kiritilishi shart'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Parol kamida 6 ta belgidan iborat bo\'lishi kerak'}), 400

    try:
        res = supabase.auth.sign_up({
            'email': email,
            'password': password
        })

        if res.user:
            if res.session:
                session['access_token'] = res.session.access_token
                session['refresh_token'] = res.session.refresh_token
                session['user_email'] = res.user.email
                session['user_id'] = res.user.id
                return jsonify({
                    'message': 'Muvaffaqiyatli ro\'yxatdan o\'tdingiz!',
                    'user': {'email': res.user.email, 'id': res.user.id}
                }), 201
            else:
                return jsonify({
                    'message': 'Ro\'yxatdan o\'tdingiz! Emailingizni tasdiqlang.',
                    'needs_confirmation': True
                }), 201
        else:
            return jsonify({'error': 'Ro\'yxatdan o\'tishda xatolik yuz berdi'}), 400

    except Exception as e:
        error_msg = str(e)
        if 'already registered' in error_msg.lower() or 'already been registered' in error_msg.lower():
            return jsonify({'error': 'Bu email allaqachon ro\'yxatdan o\'tgan'}), 409
        return jsonify({'error': f'Xatolik: {error_msg}'}), 400


@app.route('/api/login', methods=['POST'])
def login():
    if not supabase:
        return jsonify({'error': 'Supabase sozlanmagan'}), 500

    data = request.get_json()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not email or not password:
        return jsonify({'error': 'Email va parol kiritilishi shart'}), 400

    try:
        res = supabase.auth.sign_in_with_password({
            'email': email,
            'password': password
        })

        session['access_token'] = res.session.access_token
        session['refresh_token'] = res.session.refresh_token
        session['user_email'] = res.user.email
        session['user_id'] = res.user.id

        return jsonify({
            'message': 'Muvaffaqiyatli kirdingiz!',
            'user': {'email': res.user.email, 'id': res.user.id}
        }), 200

    except Exception as e:
        error_msg = str(e)
        if 'invalid' in error_msg.lower() or 'credentials' in error_msg.lower():
            return jsonify({'error': 'Email yoki parol noto\'g\'ri'}), 401
        return jsonify({'error': f'Xatolik: {error_msg}'}), 400


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Chiqdingiz'}), 200


@app.route('/api/me', methods=['GET'])
def me():
    if 'access_token' not in session:
        return jsonify({'authenticated': False}), 401
    return jsonify({
        'authenticated': True,
        'user': {
            'email': session.get('user_email'),
            'id': session.get('user_id')
        }
    }), 200


# ═══════════════════════════════════════════════════════════════
#  MULTI-AI ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════

def search_web_for_claim(text):
    """
    Step 1: Use Groq Compound model to search the web for the claim.
    Returns search findings with real source URLs.
    """
    if not groq_client:
        return None

    try:
        response = groq_client.chat.completions.create(
            model='compound-beta',
            max_tokens=1500,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a fact-checking research assistant. '
                        'Search the web for the given news claim. '
                        'Find relevant articles from news websites that confirm or deny this claim. '
                        'Return ONLY a valid JSON object with NO markdown, NO backticks. '
                        'Use this exact format: '
                        '{"search_summary": "Brief summary of what you found in English", '
                        '"sources": [{"url": "actual URL", "title": "article title", '
                        '"status": "confirms" or "contradicts" or "related", '
                        '"snippet": "relevant quote or summary from the source"}], '
                        '"web_consensus": "confirmed" or "disputed" or "unverified"}'
                    )
                },
                {
                    'role': 'user',
                    'content': f'Fact-check this news claim by searching the web:\n\n{text}'
                }
            ]
        )

        raw = response.choices[0].message.content
        clean = raw.replace('```json', '').replace('```', '').strip()
        return json.loads(clean)

    except Exception as e:
        print(f'[Step 1 - Web Search] Error: {e}')
        return None


def analyze_with_evidence(text, search_results, lang='uz'):
    """
    Step 2: Use Llama 3.3 70B to deeply analyze the news
    with the web search evidence from Step 1.
    """
    if not GROQ_API_KEY:
        return None

    # Language-specific labels for internal context
    labels = {
        'uz': {'confirms': 'TASDIQLAYDI', 'contradicts': 'RAD ETADI', 'related': 'TEGISHLI', 'unknown': 'NOMA\'LUM', 'none': 'Internet qidiruvida natija topilmadi.'},
        'en': {'confirms': 'CONFIRMS', 'contradicts': 'CONTRADICTS', 'related': 'RELATED', 'unknown': 'UNKNOWN', 'none': 'No results found in web search.'},
        'ru': {'confirms': 'ПОДТВЕРЖДАЕТ', 'contradicts': 'ОПРОВЕРГАЕТ', 'related': 'СВЯЗАНО', 'unknown': 'НЕИЗВЕСТНО', 'none': 'Результатов поиска в интернете не найдено.'},
        'ja': {'confirms': '確認済み', 'contradicts': '否定された', 'related': '関連', 'unknown': '不明', 'none': 'ウェブ検索の結果が見つかりませんでした。'},
        'zh': {'confirms': '核实', 'contradicts': '否认', 'related': '相关', 'unknown': '未知', 'none': '未在网页搜索中找到结果。'}
    }
    
    # Language names for the prompt
    lang_names = {
        'uz': "o'zbek tilida",
        'en': "in English",
        'ru': "на русском языке",
        'ja': "日本語で",
        'zh': "用中文"
    }

    l = labels.get(lang, labels['en'])
    ln = lang_names.get(lang, "in English")

    # Build context from search results
    evidence_context = ''
    if search_results and search_results.get('sources'):
        evidence_lines = []
        for src in search_results['sources']:
            status_label = l.get(src.get('status', ''), l['unknown'])
            evidence_lines.append(
                f"- [{status_label}] {src.get('title', 'N/A')} ({src.get('url', '')})\n"
                f"  {src.get('snippet', '')}"
            )
        evidence_context = '\n'.join(evidence_lines)
        web_consensus = search_results.get('web_consensus', 'unverified')
    else:
        evidence_context = l['none']
        web_consensus = 'unverified'

    system_prompt = f"""You are a fake news detection expert. You have been given a news text AND real web search results about this claim.

WEB SEARCH RESULTS:
{evidence_context}

WEB CONSENSUS: {web_consensus}

Based on BOTH the news text analysis AND the web search evidence, return ONLY a valid JSON object.
No preamble, no markdown, no backticks. Write summary and signals {ln}.
Use exactly these fields:
{{
  "verdict": "LIKELY FAKE" or "SUSPICIOUS" or "LIKELY REAL",
  "confidence": integer 0-100,
  "summary": "2-5 sentences explanation {ln}, based on web search results",
  "signals": ["3 to 5 short signals {ln}"]
}}

IMPORTANT: Base your verdict on the WEB EVIDENCE, not just your general knowledge.
If web sources contradict the claim, it's likely fake.
If web sources confirm the claim, it's likely real.
If no evidence found, be cautious and mark as suspicious."""

    try:
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {GROQ_API_KEY}'
            },
            json={
                'model': 'llama-3.3-70b-versatile',
                'max_tokens': 1000,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': text}
                ]
            }
        )

        if not response.ok:
            err = response.json()
            print(f'[Step 2 - Analysis] API Error: {err}')
            return None

        result = response.json()
        raw = result['choices'][0]['message']['content']
        clean = raw.replace('```json', '').replace('```', '').strip()
        return json.loads(clean)

    except Exception as e:
        print(f'[Step 2 - Analysis] Error: {e}')
        return None


def check_factcheck_db(text):
    """
    Step 3: Query Google Fact Check API to see if this claim
    has already been fact-checked by organizations.
    """
    if not GOOGLE_FACTCHECK_API_KEY:
        return []

    try:
        # Use first 200 chars as query (API has query length limits)
        query = text[:200].strip()
        response = requests.get(
            'https://factchecktools.googleapis.com/v1alpha1/claims:search',
            params={
                'query': query,
                'key': GOOGLE_FACTCHECK_API_KEY,
                'languageCode': 'en'  # Try Uzbek first
            },
            timeout=8
        )

        fact_checks = []

        if response.ok:
            data = response.json()
            claims = data.get('claims', [])

            for claim in claims[:5]:  # Max 5 results
                for review in claim.get('claimReview', []):
                    fact_checks.append({
                        'claim': claim.get('text', ''),
                        'publisher': review.get('publisher', {}).get('name', 'Noma\'lum'),
                        'rating': review.get('textualRating', ''),
                        'url': review.get('url', ''),
                        'title': review.get('title', '')
                    })

        # If no Uzbek results, try without language filter
        if not fact_checks:
            response = requests.get(
                'https://factchecktools.googleapis.com/v1alpha1/claims:search',
                params={
                    'query': query,
                    'key': GOOGLE_FACTCHECK_API_KEY
                },
                timeout=8
            )
            if response.ok:
                data = response.json()
                for claim in data.get('claims', [])[:5]:
                    for review in claim.get('claimReview', []):
                        fact_checks.append({
                            'claim': claim.get('text', ''),
                            'publisher': review.get('publisher', {}).get('name', 'Noma\'lum'),
                            'rating': review.get('textualRating', ''),
                            'url': review.get('url', ''),
                            'title': review.get('title', '')
                        })

        return fact_checks

    except Exception as e:
        print(f'[Step 3 - Fact Check] Error: {e}')
        return []


# ── Main Analyze Endpoint (protected) ──────────────────────────
@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    data = request.get_json()
    text = data.get('text', '').strip()
    lang = data.get('lang', 'en')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Run Step 3 (Google Fact Check) in parallel with Steps 1+2
        fact_checks = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Start fact check in background
            fact_check_future = executor.submit(check_factcheck_db, text)

            # Step 1: Web search (must complete before Step 2)
            search_results = search_web_for_claim(text)

            # Step 2: Deep analysis with web evidence
            analysis = analyze_with_evidence(text, search_results, lang)

            # Collect fact check results
            try:
                fact_checks = fact_check_future.result(timeout=10)
            except Exception:
                fact_checks = []

        # ── Combine results ─────────────────────────────────────
        if not analysis:
            return jsonify({'error': 'Tahlil qilishda xatolik yuz berdi'}), 500

        # Build real sources list from web search
        sources = []
        if search_results and search_results.get('sources'):
            for src in search_results['sources']:
                sources.append({
                    'name': extract_domain(src.get('url', '')),
                    'url': src.get('url', ''),
                    'title': src.get('title', ''),
                    'status': src.get('status', 'related'),
                    'snippet': src.get('snippet', '')
                })

        # Build final response
        result = {
            'verdict': analysis.get('verdict', 'SUSPICIOUS'),
            'confidence': analysis.get('confidence', 50),
            'summary': analysis.get('summary', ''),
            'signals': analysis.get('signals', []),
            'sources': sources,
            'fact_checks': fact_checks,
            'web_consensus': search_results.get('web_consensus', 'unverified') if search_results else 'unverified'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def extract_domain(url):
    """Extract clean domain name from URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.replace('www.', '')
        return domain
    except Exception:
        return url


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 3000)))