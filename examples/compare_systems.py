"""System Comparison — the main benchmark script.

Runs all 7 RAG variants on the same dataset with the same 8 evaluators,
saves each run to the database, and prints a final comparison table showing
how quality metrics improve as techniques are added.

Systems evaluated (in order):
  1. base-llm          — No retrieval (the floor)
  2. naive-rag         — FAISS semantic search
  3. hybrid-rag        — BM25 + FAISS + RRF
  4. reranking-rag     — FAISS + cross-encoder reranker
  5. hyde-rag          — Hypothetical Document Embeddings
  6. query-rewriting   — Multi-query retrieval
  7. advanced-rag      — All techniques combined (the ceiling)

Run with:
    python examples/compare_systems.py
    python examples/compare_systems.py --limit 20   # faster, fewer questions
    python examples/compare_systems.py --doc my_report.pdf
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from sentence_transformers import CrossEncoder

from eval_framework.config import get_settings, configure_logging
from eval_framework.types import QAPair
from eval_framework.utils.llm_client import get_llm_client
from eval_framework.judges.pipeline import EvaluationPipeline
from eval_framework.storage.database import EvalDatabase
from eval_framework.evaluators import (
    FaithfulnessEvaluator,
    RelevanceEvaluator,
    CompletenessEvaluator,
    HallucinationRateEvaluator,
    LatencyEvaluator,
    CostEvaluator,
)
from eval_framework.evaluators.conciseness import ConcisenessEvaluator
from eval_framework.evaluators.coherence import CoherenceEvaluator
from eval_framework.systems import (
    SharedIndex,
    BaseLLMSystem,
    NaiveRAGSystem,
    HybridRAGSystem,
    RerankingRAGSystem,
    HyDERAGSystem,
    QueryRewritingRAGSystem,
    AdvancedRAGSystem,
)

configure_logging()
console = Console()

# Metrics displayed in the final comparison table (left to right)
DISPLAY_METRICS = [
    "faithfulness",
    "relevance",
    "completeness",
    "hallucination_rate",
    "conciseness",
    "coherence",
    "latency",
    "cost",
]


_JUDGE_MODEL = "llama-3.1-8b-instant"  # Fast 8B judge — 500k tokens/day free tier


def build_evaluators(model_name: str):
    # Use a small fast model as the LLM judge — it only needs to output a
    # score (0-1) and one-line reasoning, so 8B quality is more than enough.
    # This keeps us well within Groq's free-tier token limits while running
    # all 7 systems on 50 questions.
    judge_client = get_llm_client("groq", model=_JUDGE_MODEL)
    return [
        FaithfulnessEvaluator(judge_client, _JUDGE_MODEL),
        RelevanceEvaluator(judge_client, _JUDGE_MODEL),
        CompletenessEvaluator(judge_client, _JUDGE_MODEL),
        HallucinationRateEvaluator(judge_client, _JUDGE_MODEL),
        ConcisenessEvaluator(judge_client, _JUDGE_MODEL),
        CoherenceEvaluator(judge_client, _JUDGE_MODEL),
        LatencyEvaluator(model_name=model_name),
        CostEvaluator(model_name=model_name),
    ]


async def run_one_system(name, system_fn, dataset, evaluators, concurrency, db):
    """Evaluate a single system and save the report."""
    pipeline = EvaluationPipeline(evaluators=evaluators, concurrency=concurrency)
    report = await pipeline.run(
        system=system_fn,
        dataset=dataset,
        system_name=name,
    )
    db.save_report(report)
    return report


def score_color(score: float) -> str:
    if score >= 0.80:
        return f"[green]{score:.3f}[/green]"
    elif score >= 0.60:
        return f"[yellow]{score:.3f}[/yellow]"
    elif score >= 0.40:
        return f"[orange1]{score:.3f}[/orange1]"
    else:
        return f"[red]{score:.3f}[/red]"


def print_comparison_table(reports: list):
    """Print a side-by-side metric comparison for all systems."""
    table = Table(
        title="System Comparison — All Metrics",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Metric", style="cyan", min_width=18)
    for report in reports:
        table.add_column(report.system_name, min_width=14)

    for metric in DISPLAY_METRICS:
        row = [metric.replace("_", " ").title()]
        for report in reports:
            scores = report.summary_scores
            score = next(
                (v for k, v in scores.items() if (k.value if hasattr(k, "value") else k) == metric),
                None,
            )
            if score is None:
                row.append("—")
            elif metric in ("latency",):
                # Latency: lower is better, show in ms (score is 0-1 normalized)
                row.append(f"{score:.3f}")
            else:
                row.append(score_color(score))
        table.add_row(*row)

    console.print(table)


def print_winner_summary(reports: list):
    """Print which system won each metric."""
    console.print("\n[bold]Best system per metric:[/bold]")
    key_metrics = ["faithfulness", "relevance", "completeness", "hallucination_rate"]
    for metric in key_metrics:
        best_name, best_score = None, -1.0
        for report in reports:
            score = next(
                (v for k, v in report.summary_scores.items()
                 if (k.value if hasattr(k, "value") else k) == metric),
                None,
            )
            if score is not None and score > best_score:
                best_score = score
                best_name = report.system_name
        if best_name:
            console.print(
                f"  [cyan]{metric.replace('_',' ').title():25}[/cyan] "
                f"-> [green]{best_name}[/green] ({best_score:.3f})"
            )


async def main(doc_path: str, dataset_path: str, concurrency: int, limit: int | None):
    settings = get_settings()

    if not settings.groq_api_key:
        console.print("[red]GROQ_API_KEY not set in .env[/red]")
        return

    console.print(Panel(
        f"Document : [cyan]{doc_path}[/cyan]\n"
        f"Dataset  : [cyan]{dataset_path}[/cyan]\n"
        f"Limit    : [cyan]{'all' if limit is None else limit} questions[/cyan]\n"
        f"Systems  : [cyan]7 (base-llm -> advanced-rag)[/cyan]",
        title="RAG System Comparison",
        expand=False,
    ))

    # ── Build shared index (once) ─────────────────────────────────────────────
    with console.status("[bold green]Building shared FAISS + BM25 index..."):
        index = SharedIndex(doc_path).build()

    # ── Load cross-encoder once (shared between reranking + advanced) ─────────
    console.print("[bold green]Loading cross-encoder reranker...[/bold green]")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    console.print("[green]Reranker ready[/green]")

    # ── Build all 7 systems ───────────────────────────────────────────────────
    model_name = settings.groq_model
    systems = [
        ("base-llm",        BaseLLMSystem(model_name=model_name)),
        ("naive-rag",       NaiveRAGSystem(index, model_name=model_name)),
        ("hybrid-rag",      HybridRAGSystem(index, model_name=model_name)),
        ("reranking-rag",   RerankingRAGSystem(index, model_name=model_name, reranker=reranker)),
        ("hyde-rag",        HyDERAGSystem(index, model_name=model_name)),
        ("query-rewriting", QueryRewritingRAGSystem(index, model_name=model_name)),
        ("advanced-rag",    AdvancedRAGSystem(index, model_name=model_name, reranker=reranker)),
    ]
    console.print(f"[green]{len(systems)} systems ready[/green]")

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(dataset_path) as f:
        raw_data = json.load(f)
    if limit:
        raw_data = raw_data[:limit]

    evaluators = build_evaluators(model_name)
    db_path = Path(__file__).parent.parent / "data" / "results.db"
    db = EvalDatabase(str(db_path))

    total_questions = len(raw_data)
    console.print(
        f"\n[yellow]Running {total_questions} questions × {len(evaluators)} metrics "
        f"× {len(systems)} systems[/yellow]"
    )
    est_minutes = total_questions * len(systems) * 0.25
    console.print(f"[dim]Estimated time: ~{est_minutes:.0f}–{est_minutes*1.5:.0f} minutes on Groq free tier[/dim]\n")

    # ── Run each system ───────────────────────────────────────────────────────
    reports = []
    for system_name, system in systems:
        console.rule(f"[bold]{system_name}[/bold]")

        # Fresh dataset copy each time (query() mutates qa_pair.context in place)
        dataset = [
            QAPair(question=d["question"], answer=d["answer"], context=d.get("context"))
            for d in raw_data
        ]

        with console.status(f"[bold green]Evaluating {system_name}..."):
            report = await run_one_system(
                name=system_name,
                system_fn=system.query,
                dataset=dataset,
                evaluators=evaluators,
                concurrency=concurrency,
                db=db,
            )

        reports.append(report)

        # Quick per-system summary
        scores = report.summary_scores
        faith = next((v for k, v in scores.items() if (k.value if hasattr(k,"value") else k) == "faithfulness"), 0)
        relev = next((v for k, v in scores.items() if (k.value if hasattr(k,"value") else k) == "relevance"), 0)
        console.print(
            f"[green]Done[/green] — faithfulness={faith:.3f}  relevance={relev:.3f}  "
            f"examples={report.total_examples_evaluated}  "
            f"run_id={report.id[:8]}"
        )

    # ── Final comparison table ────────────────────────────────────────────────
    console.print()
    print_comparison_table(reports)
    print_winner_summary(reports)

    total_cost = sum(r.total_cost for r in reports)
    console.print(
        f"\n[dim]All {len(reports)} reports saved to {db_path} | "
        f"Total evaluation cost: ${total_cost:.4f}[/dim]"
    )
    console.print("\n[bold]Open the dashboard to explore results:[/bold]")
    console.print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare 7 RAG systems end-to-end")
    parser.add_argument("--doc", default="data/knowledge_base.txt",
                        help="Knowledge base document (.txt or .pdf)")
    parser.add_argument("--dataset", default="data/rag_dataset.json",
                        help="QA dataset JSON")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Max concurrent Groq API calls")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N questions (default: all). Use 20 for a quick run.")
    args = parser.parse_args()

    asyncio.run(main(
        doc_path=args.doc,
        dataset_path=args.dataset,
        concurrency=args.concurrency,
        limit=args.limit,
    ))
