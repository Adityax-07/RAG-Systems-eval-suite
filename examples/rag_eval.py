"""End-to-end RAG evaluation demo.

This script shows the full workflow:
    1. Build a real RAG system on a knowledge base document
    2. Run it against a curated dataset of questions
    3. Evaluate every answer with 8 quality metrics
    4. Save results to the database
    5. Print a summary table

Run with:
    python examples/rag_eval.py

Optional flags:
    python examples/rag_eval.py --doc data/knowledge_base.txt
    python examples/rag_eval.py --doc my_report.pdf --name my-rag-v2
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from eval_framework.config import get_settings, configure_logging
from eval_framework.rag.pipeline import RAGPipeline
from eval_framework.judges.pipeline import EvaluationPipeline
from eval_framework.storage.database import EvalDatabase
from eval_framework.types import QAPair
from eval_framework.utils.llm_client import get_llm_client
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

configure_logging()
console = Console()


def build_evaluators(model_name: str):
    """Build the set of evaluators to run."""
    client = get_llm_client("groq")
    return [
        FaithfulnessEvaluator(client, model_name),
        RelevanceEvaluator(client, model_name),
        CompletenessEvaluator(client, model_name),
        HallucinationRateEvaluator(client, model_name),
        ConcisenessEvaluator(client, model_name),
        CoherenceEvaluator(client, model_name),
        LatencyEvaluator(model_name=model_name),
        CostEvaluator(model_name=model_name),
    ]


async def main(doc_path: str, dataset_path: str, system_name: str, concurrency: int):
    settings = get_settings()

    if not settings.groq_api_key:
        console.print("[red]GROQ_API_KEY not set in .env[/red]")
        return

    # ── Step 1: Build the RAG system ──────────────────────────────────────────
    console.print(Panel(
        f"Document: [cyan]{doc_path}[/cyan]\n"
        f"Dataset:  [cyan]{dataset_path}[/cyan]\n"
        f"System:   [cyan]{system_name}[/cyan]",
        title="RAG Evaluation",
        expand=False,
    ))

    with console.status("[bold green]Building RAG index..."):
        rag = RAGPipeline(
            doc_path=doc_path,
            chunk_size=500,
            chunk_overlap=75,
            top_k=3,
            model_name=settings.groq_model,
        ).build()

    console.print("[green]RAG pipeline ready[/green]")

    # ── Step 2: Load dataset ───────────────────────────────────────────────────
    with open(dataset_path) as f:
        raw_data = json.load(f)

    dataset = [
        QAPair(
            question=item["question"],
            answer=item["answer"],
            context=item.get("context"),  # May be None — RAG will fill this in
        )
        for item in raw_data
    ]
    console.print(f"[green]Loaded {len(dataset)} questions[/green]")

    # ── Step 3: Build evaluators ───────────────────────────────────────────────
    evaluators = build_evaluators(settings.groq_model)
    console.print(f"[green]{len(evaluators)} evaluators ready[/green]")

    # ── Step 4: Run evaluation ─────────────────────────────────────────────────
    pipeline = EvaluationPipeline(evaluators=evaluators, concurrency=concurrency)

    console.print(
        f"\n[yellow]Running {len(dataset)} questions × {len(evaluators)} metrics "
        f"(concurrency={concurrency})...[/yellow]"
    )
    console.print("[dim]This takes ~2-4 minutes on Groq free tier[/dim]\n")

    with console.status("[bold green]Evaluating..."):
        report = await pipeline.run(
            system=rag.query,
            dataset=dataset,
            system_name=system_name,
        )

    # ── Step 5: Save to database ───────────────────────────────────────────────
    db_path = Path(__file__).parent.parent / "data" / "results.db"
    db = EvalDatabase(str(db_path))
    db.save_report(report)
    console.print(f"\n[green]Saved report {report.id[:8]}... to {db_path}[/green]")

    # ── Step 6: Print summary ──────────────────────────────────────────────────
    table = Table(
        title=f"Results: {system_name}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", min_width=22)
    table.add_column("Score", min_width=8)
    table.add_column("Rating", min_width=12)

    for metric, score in report.summary_scores.items():
        metric_name = metric.value if hasattr(metric, "value") else str(metric)
        if score >= 0.8:
            rating = "[green]EXCELLENT[/green]"
        elif score >= 0.6:
            rating = "[yellow]GOOD[/yellow]"
        elif score >= 0.4:
            rating = "[orange1]FAIR[/orange1]"
        else:
            rating = "[red]POOR[/red]"
        table.add_row(metric_name, f"{score:.3f}", rating)

    console.print(table)
    console.print(
        f"\n[dim]Total cost: ${report.total_cost:.4f} | "
        f"Examples: {report.total_examples_evaluated} | "
        f"Run ID: {report.id[:8]}[/dim]"
    )
    console.print("\n[bold]Open the dashboard to explore results:[/bold]")
    console.print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation Demo")
    parser.add_argument(
        "--doc",
        default="data/knowledge_base.txt",
        help="Path to knowledge base document (.txt or .pdf)",
    )
    parser.add_argument(
        "--dataset",
        default="data/rag_dataset.json",
        help="Path to QA dataset JSON",
    )
    parser.add_argument(
        "--name",
        default="rag-langchain-groq",
        help="System name for the evaluation report",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max concurrent Groq API calls (2 = safe for free tier)",
    )
    args = parser.parse_args()

    asyncio.run(main(
        doc_path=args.doc,
        dataset_path=args.dataset,
        system_name=args.name,
        concurrency=args.concurrency,
    ))
