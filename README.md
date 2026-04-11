# RAG Systems Eval Suite

A benchmark that runs **7 different RAG (Retrieval-Augmented Generation) strategies** against the same knowledge base and the same set of questions, then scores every answer with **8 LLM-as-judge metrics** so you can see exactly which retrieval strategy performs best — and why.

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
  -> LLM (Groq Llama) generates an answer
  -> 8 judges score the answer in parallel
  -> Results saved to SQLite

        |
        v

Summary table printed: score per system per metric
```

---

## Quick Start

### 1. Install

```bash
cd llm-eval-framework
pip install -e ".[dev]"
```

### 2. Add your Groq API key

```bash
cp .env.example .env
# Open .env and set:
# GROQ_API_KEY=gsk_...
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run the benchmark

```bash
python examples/compare_systems.py
```

This will:
1. Build the shared FAISS + BM25 index from `data/knowledge_base.txt`
2. Run all 7 systems on the first 10 questions (configurable with `--limit`)
3. Score every answer with 8 LLM judges
4. Print a comparison table like this:

```
System              Faithfulness  Relevance  Completeness  Hallucination  Score
------------------  -----------  ---------  ------------  -------------  -----
advanced_rag              0.91       0.88          0.85           0.08   0.87
reranking_rag             0.88       0.86          0.82           0.11   0.85
hybrid_rag                0.84       0.83          0.79           0.14   0.82
query_rewriting           0.83       0.84          0.78           0.13   0.82
hyde_rag                  0.80       0.81          0.75           0.17   0.79
naive_rag                 0.76       0.79          0.71           0.21   0.75
base_llm                  0.41       0.65          0.58           0.52   0.54
```

### 4. Visualise results

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501`.

---

## Evaluation Metrics

Each answer is scored by an LLM judge on a 0-1 scale:

| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? |
| **Relevance** | Does the answer actually address the question asked? |
| **Completeness** | Does it cover all parts of the question, or leave things out? |
| **Hallucination Rate** | Fraction of claims NOT found in the context (lower is better) |
| **Conciseness** | Is it appropriately brief, not padded with filler? |
| **Coherence** | Is it logically structured and easy to follow? |
| **Latency** | How long did it take to respond (milliseconds)? |
| **Cost** | Estimated API cost per query (USD) |

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
│   ├── evaluators/             # 8 metric implementations
│   ├── judges/                 # Async LLM-as-judge pipeline
│   ├── storage/                # SQLite result persistence
│   ├── utils/                  # LLM client (Groq) with retry logic
│   ├── config.py               # Settings via Pydantic + .env
│   └── types.py                # QAPair, SystemOutput, EvalResult
├── examples/
│   ├── compare_systems.py      # Main benchmark runner (start here)
│   └── rag_eval.py             # Single-system RAG evaluation demo
├── cli/main.py                 # Typer CLI: run, rag-eval, report, compare
├── dashboard/app.py            # Streamlit results dashboard
└── data/
    ├── knowledge_base.txt      # 10-chapter AI/ML guide (RAG source)
    └── rag_dataset.json        # 50 curated QA pairs for evaluation
```

---

## Key Concepts

**Why does Hybrid RAG beat Naive RAG?**
FAISS works on meaning -- it may miss chunks that use different words than the question.
BM25 works on exact word overlap -- great for technical terms, acronyms, proper nouns.
Combining both gives better *recall*.

**Why does Reranking RAG beat Hybrid RAG?**
FAISS bi-encoders encode the question and the chunk *separately*.
A cross-encoder reads them *together*, catching fine-grained relevance that bi-encoders miss.
Better *precision* -- the chunks that reach the LLM are more tightly relevant.

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
```

In `examples/compare_systems.py` (top of file):

```python
_JUDGE_MODEL   = "llama-3.1-8b-instant"  # model used for scoring
_LIMIT         = 10                       # questions to run (None = all 50)
_CONCURRENCY   = 2                        # parallel LLM calls
```

---

## Stack

- **Retrieval**: [FAISS](https://github.com/facebookresearch/faiss), [BM25 (rank_bm25)](https://github.com/dorianbrown/rank_bm25), [sentence-transformers](https://www.sbert.net/)
- **LLM**: [Groq](https://console.groq.com) (Llama 3.3 70B / Llama 4 Scout)
- **Orchestration**: [LangChain](https://python.langchain.com/)
- **Embeddings**: `all-MiniLM-L6-v2` (~80MB, runs locally on CPU)
- **Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~67MB, CPU)
- **Dashboard**: [Streamlit](https://streamlit.io/)
- **Storage**: SQLite via SQLAlchemy
- **Config**: Pydantic Settings + `.env`
