# VerAi: The Ultimate Fact-Checking Pipeline

**VerAi** is an intelligent, multi-layered news verification system designed to combat misinformation. It cross-references claims against real-time web data and established fact-checking databases using a high-performance, parallelized AI pipeline.

---

In an era of deepfakes and AI-generated disinformation, manual fact-checking is too slow and prone to human bias. **VerAi** cuts through the noise. It transforms a simple news snippet into a comprehensive investigation, providing a verdict, confidence score, and verifiable source citations in seconds. It’s the "truth layer" for modern digital media.

VerAi employs a three-stage asynchronous pipeline to ensure accuracy and speed:

* **Web Intel (Stage 1):** Uses Groq’s `compound-beta` model to perform targeted web searches and extract relevant news sources.
* **Deep Reasoning (Stage 2):** Passes the search evidence and the claim to Llama 3.3 70B, which performs a comparative analysis to determine credibility.
* **Fact-Database Sync (Stage 3):** Concurrently queries the Google Fact Check API to see if the claim has been debunked by major international fact-checking organizations.

Built for scale and reliability:
* **Backend:** Flask API with `ThreadPoolExecutor` for high-performance concurrent analysis.
* **Database/Auth:** Integrated with Supabase for secure, stateless user management.
* **Intelligence:** Orchestrated through the Groq Cloud API for ultra-fast, low-latency AI inference.
* **Flexibility:** Supports multi-language analysis (UZ, EN, RU, JA, ZH) with contextual awareness.

---

### Getting Started

**1. Prerequisites**
* Python 3.10+
* API Keys: [Groq](https://groq.com/), [Supabase](https://supabase.com/), [Google Fact Check](https://developers.google.com/fact-check/tools/api).

**2. Installation**
```bash
pip install flask python-dotenv supabase groq requests
