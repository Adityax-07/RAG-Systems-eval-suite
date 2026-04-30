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

# RAG Systems Eval Suite

A benchmark that runs **8 different RAG (Retrieval-Augmented Generation) strategies** against the same knowledge base and the same set of questions, then scores every answer with **10 LLM-as-judge metrics** so you can see exactly which retrieval strategy performs best — and why.

---

## Benchmark Results

> Full run: **8 systems × 50 questions × 10 metrics** — all real scores, no mocks.
> Answer generation: Groq `llama-3.3-70b-versatile` · Judge: Cerebras `llama3.1-8b`

| Rank | System | **Avg** | Faithfulness | Relevance | Completeness | Coherence | Hallucination↑ | Conciseness | Context Precision |
|------|--------|---------|-------------|-----------|--------------|-----------|----------------|-------------|-------------------|
| 🥇 | **advanced-rag** | **0.770** | 0.881 | 0.763 | 0.725 | 0.781 | 0.816 | 0.763 | 0.825 |
| 🥈 | **adaptive-rag** | **0.758** | **1.000** | 0.800 | 0.800 | 0.800 | 0.667 | 0.800 | — |
| 🥉 | **reranking-rag** | **0.752** | 0.816 | 0.754 | 0.720 | 0.766 | 0.787 | 0.730 | 0.784 |
| 4 | **base-llm** | **0.745** | 0.760 | 0.780 | 0.788 | 0.886 | 0.221 | 0.494 | 1.000 |
| 5 | **naive-rag** | **0.736** | 0.770 | 0.700 | 0.676 | 0.734 | 0.562 | 0.670 | 0.774 |
| 6 | **hyde-rag** | **0.726** | 0.800 | 0.720 | 0.690 | 0.720 | 0.575 | 0.690 | 0.820 |
| 7 | **hybrid-rag** | **0.723** | 0.738 | 0.684 | 0.620 | 0.686 | 0.613 | 0.646 | 0.772 |
| 8 | **query-rewriting** | **0.716** | 0.789 | 0.717 | 0.678 | 0.711 | 0.574 | 0.711 | 0.767 |

> Hallucination↑ = higher score means *fewer* hallucinations. Cost and toxicity: all systems score 1.000.

**Key findings:**
- `adaptive-rag` achieves **perfect faithfulness (1.000)** — the only system to do so — by routing each query to the most appropriate pipeline
- `advanced-rag` wins overall (0.770 avg) — best hallucination control (0.816) through stacking all retrieval strategies
- `reranking-rag` is the best fixed-strategy system — strong precision with low added complexity
- `base-llm` scores surprisingly high overall but has near-zero hallucination detection (0.221) — it has no grounded context to be faithful *to*
- `query-rewriting` adds the most latency overhead for the least gain — worst overall average

---

## What Gets Compared

| System | What It Does |
|--------|-------------|
| **Base LLM** | No retrieval at all. Answers from memory only. The floor baseline every other system must beat. |
| **Naive RAG** | Classic RAG: embed the question, find the top-3 closest chunks with FAISS, answer from those. |
| **Hybrid RAG** | BM25 keyword search + FAISS semantic search, merged with Reciprocal Rank Fusion. Better for exact terms and acronyms. |
| **Reranking RAG** | FAISS fetches 10 candidates, a cross-encoder re-scores each (question, chunk) pair together, keeps top-3. Better precision. |
| **HyDE RAG** | Generates a *hypothetical* answer first, embeds that instead of the question, then searches. Bridges the question/answer embedding gap. |
| **Query Rewriting** | Rewrites the question 3 different ways, runs FAISS for each version, merges all results with RRF. Better recall for ambiguous queries. |
| **Advanced RAG** | All of the above combined: query rewriting + hybrid search + cross-encoder reranking in one pipeline. |
| **Adaptive RAG** | Routes each query to the right pipeline based on complexity. Simple → Naive RAG, Complex → Advanced RAG, Ambiguous → Hybrid RAG. Only system to achieve perfect faithfulness (1.000). |

---

## How It Works

```
Your document (PDF or TXT)
        |
        v
SharedIndex.build()
  - Split into ~500-char chunks
  - Embed with all-MiniLM-L6-v2 -> FAISS index
  - Tokenize                     -> BM25 index

        |
        v  (built once, shared across all 7 systems)

For each question in rag_dataset.json:
  Each system retrieves chunks its own way
  -> LLM (Groq Llama 3.3 70B) generates an answer
  -> 10 judges score the answer sequentially (concurrency=1)
  -> Results saved to SQLite

        |
        v

Summary table + Streamlit dashboard with real scores
```

---

## Quick Start

### 1. Install

```bash
cd llm-eval-framework
pip install -e ".[dev]"
```

### 2. Add your API keys

```bash
cp .env.example .env
# Open .env and set:
# GROQ_API_KEY=gsk_...
# CEREBRAS_API_KEY=...
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

### 3. Run the benchmark

```bash
python examples/compare_systems.py --limit 50 --concurrency 1
```

This will:
1. Build the shared FAISS + BM25 index from `data/knowledge_base.txt`
2. Run all 7 systems on 50 questions sequentially
3. Score every answer with 10 LLM judges
4. Save all results to SQLite with checkpoint/resume support

### 4. Visualise results

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501`.

---

## Evaluation Metrics

Each answer is scored by an LLM judge on a 0–1 scale:

| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? |
| **Relevance** | Does the answer actually address the question asked? |
| **Completeness** | Does it cover all parts of the question, or leave things out? |
| **Hallucination Rate** | Fraction of claims NOT found in the context (higher score = fewer hallucinations) |
| **Conciseness** | Is it appropriately brief, not padded with filler? |
| **Coherence** | Is it logically structured and easy to follow? |
| **Context Precision** | How precisely does the retrieved context match what was needed? |
| **Toxicity** | Does the response contain harmful or unsafe content? |
| **Latency** | Normalised response time (higher = faster) |
| **Cost** | Estimated API cost per query (higher = cheaper) |

---

## Project Structure

```
llm-eval-framework/
├── src/eval_framework/
│   ├── systems/
│   │   ├── shared.py           # SharedIndex: FAISS + BM25, built once
│   │   ├── base_llm.py         # No retrieval -- parametric memory only
│   │   ├── naive_rag.py        # FAISS semantic search only
│   │   ├── hybrid_rag.py       # BM25 + FAISS merged with RRF
│   │   ├── reranking_rag.py    # FAISS candidates -> cross-encoder rerank
│   │   ├── hyde_rag.py         # Hypothetical answer embedding
│   │   ├── query_rewriting.py  # Multi-query retrieval with RRF merge
│   │   └── advanced_rag.py     # All strategies combined
│   ├── evaluators/             # 10 metric implementations
│   ├── judges/                 # Async LLM-as-judge pipeline
│   ├── storage/                # SQLite result persistence + checkpoint/resume
│   ├── utils/                  # LLM client (Groq + Cerebras) with retry logic
│   ├── config.py               # Settings via Pydantic + .env
│   └── types.py                # QAPair, SystemOutput, EvalResult
├── examples/
│   └── compare_systems.py      # Main benchmark runner (start here)
├── dashboard/app.py            # Streamlit results dashboard
└── data/
    ├── knowledge_base.txt      # 10-chapter AI/ML guide (RAG source)
    └── rag_dataset.json        # 50 curated QA pairs for evaluation
```

---

## Key Concepts

**Why does Hybrid RAG beat Naive RAG?**
FAISS works on meaning — it may miss chunks that use different words than the question.
BM25 works on exact word overlap — great for technical terms, acronyms, proper nouns.
Combining both gives better *recall*.

**Why does Reranking RAG beat Hybrid RAG?**
FAISS bi-encoders encode the question and the chunk *separately*.
A cross-encoder reads them *together*, catching fine-grained relevance that bi-encoders miss.
Better *precision* — the chunks that reach the LLM are more tightly relevant.

**Why does HyDE help?**
Questions and answers live in slightly different embedding spaces.
"What is FAISS?" as a vector is not that close to "FAISS is a library for..." as a vector.
Generating a fake answer and embedding *that* puts the query vector into answer-space,
so FAISS finds better matches.

**Why does Advanced RAG score highest?**
It stacks all three improvements: diverse queries (query rewriting) + both search types
(hybrid) + accurate re-scoring (reranking). More compute, best results.

---

## Configuration

In `.env`:

```bash
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile   # model for RAG answer generation
CEREBRAS_API_KEY=...                  # model for LLM-as-judge scoring
```

Run flags:

```bash
python examples/compare_systems.py --limit 50 --concurrency 1
#   --limit N        number of questions per system (max 50)
#   --concurrency N  parallel LLM calls (use 1 to avoid rate limits)
```

---

## Stack

- **Retrieval**: [FAISS](https://github.com/facebookresearch/faiss), [BM25 (rank_bm25)](https://github.com/dorianbrown/rank_bm25), [sentence-transformers](https://www.sbert.net/)
- **Answer generation**: [Groq](https://console.groq.com) — Llama 3.3 70B
- **LLM judge**: [Cerebras](https://cerebras.ai) — Llama 3.1 8B
- **Embeddings**: `all-MiniLM-L6-v2` (~80MB, runs locally on CPU)
- **Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~67MB, CPU)
- **Dashboard**: [Streamlit](https://streamlit.io/)
- **Storage**: SQLite with checkpoint/resume support
- **Config**: Pydantic Settings + `.env`
