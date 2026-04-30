---
title: RAG Systems Eval Suite
emoji: 📊
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8501
app_file: dashboard/app.py
pinned: false
---

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&pause=1000&color=8B5CF6&center=true&vCenter=true&width=600&lines=RAG+Systems+Eval+Suite;Benchmark+8+RAG+Strategies;10+LLM-as-Judge+Metrics;Real+Scores%2C+No+Mocks" alt="Typing SVG" />

<br/>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white" />
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Adityax-07/RAG-Systems-eval-suite">
    <img src="https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-FFD21E?style=for-the-badge" />
  </a>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Production-brightgreen?style=for-the-badge" />
</p>

<br/>

> **A production-grade benchmark that runs 8 RAG strategies on the same knowledge base and same questions, then scores every answer with 10 LLM-as-judge metrics — so you can see exactly which retrieval approach wins and why.**

🔗 **Live Dashboard:** [huggingface.co/spaces/Adityax-07/RAG-Systems-eval-suite](https://huggingface.co/spaces/Adityax-07/RAG-Systems-eval-suite)

</div>

---

## 📊 Benchmark Results

> **Full run:** 8 systems × 50 questions × 10 metrics — all real scores, zero mocks.
>
> 🔵 **Generator:** Groq `llama-3.3-70b-versatile` &nbsp;|&nbsp; 🟣 **Judge:** Cerebras `llama3.1-8b`

| Rank | System | Avg Score | Faithfulness | Relevance | Completeness | Coherence | Hallucination ↑ | Conciseness | Context Precision |
|:----:|--------|:---------:|:------------:|:---------:|:------------:|:---------:|:---------------:|:-----------:|:-----------------:|
| 🥇 | **advanced-rag** | **0.770** | 0.881 | 0.763 | 0.725 | 0.781 | 0.816 | 0.763 | 0.825 |
| 🥈 | **adaptive-rag** | **0.758** | **1.000** | 0.800 | 0.800 | 0.800 | 0.667 | 0.800 | — |
| 🥉 | **reranking-rag** | **0.752** | 0.816 | 0.754 | 0.720 | 0.766 | 0.787 | 0.730 | 0.784 |
| 4 | base-llm | 0.745 | 0.760 | 0.780 | 0.788 | 0.886 | 0.221 | 0.494 | 1.000 |
| 5 | naive-rag | 0.736 | 0.770 | 0.700 | 0.676 | 0.734 | 0.562 | 0.670 | 0.774 |
| 6 | hyde-rag | 0.726 | 0.800 | 0.720 | 0.690 | 0.720 | 0.575 | 0.690 | 0.820 |
| 7 | hybrid-rag | 0.723 | 0.738 | 0.684 | 0.620 | 0.686 | 0.613 | 0.646 | 0.772 |
| 8 | query-rewriting | 0.716 | 0.789 | 0.717 | 0.678 | 0.711 | 0.574 | 0.711 | 0.767 |

> ⚠️ **Hallucination ↑** = higher score = *fewer* hallucinations. Toxicity & Cost: all systems score **1.000**.

### 🔑 Key Findings

| Insight | Detail |
|---------|--------|
| 🎯 **Perfect Faithfulness** | `adaptive-rag` is the **only** system to achieve 1.000 faithfulness by routing each query to the optimal pipeline |
| 🏆 **Best Overall** | `advanced-rag` wins (0.770 avg) — best hallucination control (0.816) via stacking all retrieval strategies |
| ⚡ **Best Fixed Strategy** | `reranking-rag` delivers strong precision with minimal added complexity |
| ⚠️ **Deceptive Baseline** | `base-llm` ranks 4th overall but has near-zero hallucination detection (0.221) — no grounded context to be faithful *to* |
| 🐌 **Worst ROI** | `query-rewriting` adds the most latency for the least gain |

---

## 🧠 Systems Compared

| System | Strategy | Key Mechanism |
|--------|----------|---------------|
| **Base LLM** | No retrieval | Answers from parametric memory only — the floor every system must beat |
| **Naive RAG** | Semantic search | Embed question → FAISS top-3 chunks → generate |
| **Hybrid RAG** | BM25 + Semantic | Keyword overlap + semantic search merged via Reciprocal Rank Fusion |
| **Reranking RAG** | FAISS + Cross-encoder | Fetch 10 candidates → cross-encoder re-scores (question, chunk) pairs → keep top-3 |
| **HyDE RAG** | Hypothetical embedding | Generate fake answer first → embed that → search in answer-space |
| **Query Rewriting** | Multi-query RRF | Rewrite question 3 ways → FAISS per version → merge results |
| **Advanced RAG** | All combined | Query rewriting + hybrid search + cross-encoder reranking stacked |
| **Adaptive RAG** | Query routing | Simple → Naive RAG · Complex → Advanced RAG · Ambiguous → Hybrid RAG |

---

## 📐 Evaluation Metrics

Each answer is scored by an LLM judge on a **0–1 scale** across 10 dimensions:

| Metric | What It Measures |
|--------|-----------------|
| 📌 **Faithfulness** | Are all claims supported by retrieved context? |
| 🎯 **Relevance** | Does the answer actually address the question? |
| 📋 **Completeness** | Does it cover all parts of the question? |
| 🚫 **Hallucination Rate** | Fraction of claims NOT grounded in context (↑ = fewer hallucinations) |
| ✂️ **Conciseness** | Is it brief and free of padding? |
| 🧩 **Coherence** | Is it logically structured and easy to follow? |
| 🔍 **Context Precision** | How precisely does retrieved context match what was needed? |
| 🛡️ **Toxicity** | Does the response contain harmful content? |
| ⏱️ **Latency** | Normalised response time (↑ = faster) |
| 💰 **Cost** | Estimated API cost per query (↑ = cheaper) |

---

## ⚙️ Architecture

```
Your Document (PDF or TXT)
        │
        ▼
SharedIndex.build()
  ├─ Split into ~500-char chunks
  ├─ Embed with all-MiniLM-L6-v2  ──► FAISS index
  └─ Tokenize                      ──► BM25 index
        │
        │  (built ONCE, shared across all 8 systems)
        ▼
For each question in rag_dataset.json:
  ├─ Each system retrieves chunks its own way
  ├─ Groq Llama 3.3 70B generates an answer
  ├─ 10 Cerebras judges score the answer (concurrency=1)
  └─ Results persisted to SQLite (checkpoint/resume)
        │
        ▼
  Streamlit Dashboard  ──►  Summary table + charts
```

---

## 🗂️ Project Structure

```
llm-eval-framework/
├── 📁 src/eval_framework/
│   ├── 📁 systems/
│   │   ├── shared.py            # SharedIndex: FAISS + BM25, built once
│   │   ├── base_llm.py          # No retrieval — parametric memory only
│   │   ├── naive_rag.py         # FAISS semantic search only
│   │   ├── hybrid_rag.py        # BM25 + FAISS merged with RRF
│   │   ├── reranking_rag.py     # FAISS candidates → cross-encoder rerank
│   │   ├── hyde_rag.py          # Hypothetical answer embedding
│   │   ├── query_rewriting.py   # Multi-query retrieval with RRF merge
│   │   ├── advanced_rag.py      # All strategies combined
│   │   └── adaptive_rag.py      # Query routing across all strategies
│   ├── 📁 evaluators/           # 10 metric implementations
│   ├── 📁 judges/               # Async LLM-as-judge pipeline
│   ├── 📁 storage/              # SQLite result persistence + checkpoint/resume
│   ├── 📁 utils/                # LLM client (Groq + Cerebras) with retry logic
│   ├── config.py                # Settings via Pydantic + .env
│   └── types.py                 # QAPair, SystemOutput, EvalResult
├── 📁 examples/
│   └── compare_systems.py       # ← Main benchmark runner (start here)
├── 📁 dashboard/
│   └── app.py                   # Streamlit results dashboard
└── 📁 data/
    ├── knowledge_base.txt        # 10-chapter AI/ML guide (RAG source)
    └── rag_dataset.json          # 50 curated QA pairs for evaluation
```

---

## 🚀 Quick Start

### 1 — Install

```bash
cd llm-eval-framework
pip install -e ".[dev]"
```

### 2 — Configure API Keys

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile      # answer generation
CEREBRAS_API_KEY=...                     # LLM-as-judge scoring
```

> 🆓 Get a free Groq key at [console.groq.com](https://console.groq.com)

### 3 — Run the Benchmark

```bash
python examples/compare_systems.py --limit 50 --concurrency 1
```

This will:
1. Build a shared FAISS + BM25 index from `data/knowledge_base.txt`
2. Run all 8 systems on 50 questions sequentially
3. Score every answer with 10 LLM judges
4. Save results to SQLite with **checkpoint/resume** support

### 4 — Visualise Results

```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` 🎉

**Run flags:**

| Flag | Description |
|------|-------------|
| `--limit N` | Number of questions per system (max 50) |
| `--concurrency N` | Parallel LLM calls — use `1` to avoid rate limits |

---

## 💡 Key Concepts

<details>
<summary><strong>Why does Hybrid RAG beat Naive RAG?</strong></summary>

FAISS works on **meaning** — it may miss chunks that use different words than the question. BM25 works on **exact word overlap** — great for technical terms, acronyms, and proper nouns. Combining both improves *recall*.

</details>

<details>
<summary><strong>Why does Reranking RAG beat Hybrid RAG?</strong></summary>

FAISS bi-encoders encode the question and chunk *separately*. A cross-encoder reads them *together*, catching fine-grained relevance bi-encoders miss. Better *precision* — chunks that reach the LLM are more tightly relevant.

</details>

<details>
<summary><strong>Why does HyDE help?</strong></summary>

Questions and answers live in slightly different embedding spaces. `"What is FAISS?"` as a vector isn't close to `"FAISS is a library for..."`. Generating a hypothetical answer and embedding *that* puts the query vector into answer-space, so FAISS finds better matches.

</details>

<details>
<summary><strong>Why does Advanced RAG score highest overall?</strong></summary>

It stacks all three improvements: diverse queries (query rewriting) + both search types (hybrid) + accurate re-scoring (reranking). More compute, best results.

</details>

<details>
<summary><strong>Why does Adaptive RAG achieve perfect faithfulness?</strong></summary>

By routing simple queries to Naive RAG (fast, focused context) and complex queries to Advanced RAG (broad, high-quality context), it avoids over-retrieving noisy context for simple questions — the leading cause of faithfulness failures.

</details>

---

## 🛠️ Tech Stack

<table>
  <tr>
    <th>Category</th>
    <th>Tool</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td>🔍 <strong>Retrieval</strong></td>
    <td>
      <img src="https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white" />
      <img src="https://img.shields.io/badge/rank__bm25-555555?style=flat-square" />
    </td>
    <td>Semantic + keyword vector search</td>
  </tr>
  <tr>
    <td>🤖 <strong>Embeddings</strong></td>
    <td>
      <img src="https://img.shields.io/badge/sentence--transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
    </td>
    <td><code>all-MiniLM-L6-v2</code> (~80MB, CPU)</td>
  </tr>
  <tr>
    <td>⚡ <strong>Reranking</strong></td>
    <td>
      <img src="https://img.shields.io/badge/cross--encoder-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
    </td>
    <td><code>ms-marco-MiniLM-L-6-v2</code> (~67MB, CPU)</td>
  </tr>
  <tr>
    <td>🌩️ <strong>LLM Generation</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Groq-F55036?style=flat-square&logo=groq&logoColor=white" />
    </td>
    <td>Llama 3.3 70B — answer generation</td>
  </tr>
  <tr>
    <td>⚖️ <strong>LLM Judge</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Cerebras-7B2EFF?style=flat-square" />
    </td>
    <td>Llama 3.1 8B — evaluation scoring</td>
  </tr>
  <tr>
    <td>📊 <strong>Dashboard</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
      <img src="https://img.shields.io/badge/Chart.js-FF6384?style=flat-square&logo=chartdotjs&logoColor=white" />
    </td>
    <td>Interactive results visualisation</td>
  </tr>
  <tr>
    <td>🗄️ <strong>Storage</strong></td>
    <td>
      <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white" />
    </td>
    <td>Result persistence + checkpoint/resume</td>
  </tr>
  <tr>
    <td>⚙️ <strong>Config</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white" />
    </td>
    <td>Settings via Pydantic Settings + <code>.env</code></td>
  </tr>
  <tr>
    <td>🐋 <strong>Deployment</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" />
      <img src="https://img.shields.io/badge/HuggingFace%20Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
    </td>
    <td>Containerised HF Spaces deployment</td>
  </tr>
</table>

---

## 📈 Insights at a Glance

```
Faithfulness Improvement over Base LLM:
  adaptive-rag   ████████████████████ 1.000  (+31.6%)  ✨ Perfect
  advanced-rag   █████████████████░░░ 0.881  (+15.9%)
  reranking-rag  ████████████████░░░░ 0.816  (+7.4%)
  base-llm       ███████████████░░░░░ 0.760  (baseline)

Hallucination Control (higher = safer):
  advanced-rag   ████████████████░░░░ 0.816  🔒 Best
  reranking-rag  ███████████████░░░░░ 0.787
  adaptive-rag   █████████████░░░░░░░ 0.667
  base-llm       ████░░░░░░░░░░░░░░░░ 0.221  ⚠️  Worst
```

---

<div align="center">

**Built by [Adityax-07](https://github.com/Adityax-07) · Powered by Groq + Cerebras + FAISS**

<br/>

⭐ **Star this repo if it helped you understand RAG evaluation!**

</div>
