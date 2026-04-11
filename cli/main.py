"""CLI entry point for the LLM Evaluation Framework.

Commands:
    llm-eval run         — Evaluate a system against a dataset
    llm-eval generate    — Generate a synthetic QA dataset from documents
    llm-eval report      — Print historical evaluation reports
    llm-eval compare     — Compare two evaluation runs
    llm-eval calibrate   — Run judge calibration from a labeled CSV

Teaching note: Typer is Click with type hints — function arguments become
CLI flags automatically. This is the pattern modern Python CLIs use.
The key idea: each command is just an async function wrapped with asyncio.run().
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Add src to path (when running from project root without pip install)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_framework.config import get_settings, configure_logging
from eval_framework.storage.database import EvalDatabase
from eval_framework.evaluators import (
    FaithfulnessEvaluator,
    RelevanceEvaluator,
    CompletenessEvaluator,
    HallucinationRateEvaluator,
    LatencyEvaluator,
    CostEvaluator,
)
from eval_framework.utils.llm_client import get_llm_client
from eval_framework.types import QAPair, SystemOutput, EvaluationMetric

app = typer.Typer(
    name="llm-eval",
    help="LLM Evaluation Framework — measure AI quality like a pro",
    add_completion=False,
)
console = Console()


def _get_model_name(provider: str) -> str:
    """Return the configured model name for a provider."""
    settings = get_settings()
    return {
        "openai": settings.openai_model,
        "anthropic": settings.anthropic_model,
        "groq": settings.groq_model,
    }.get(provider, provider)


def _get_all_evaluators(provider: str = "groq"):
    """Build all evaluators using the configured LLM judge."""
    from eval_framework.evaluators import (
        ConcisenessEvaluator,
        CoherenceEvaluator,
        ToxicityEvaluator,
        ContextPrecisionEvaluator,
    )
    client = get_llm_client(provider)
    model_name = _get_model_name(provider)
    return [
        FaithfulnessEvaluator(client, model_name),
        RelevanceEvaluator(client, model_name),
        CompletenessEvaluator(client, model_name),
        HallucinationRateEvaluator(client, model_name),
        ConcisenessEvaluator(client, model_name),
        CoherenceEvaluator(client, model_name),
        ToxicityEvaluator(client, model_name),
        ContextPrecisionEvaluator(client, model_name),
        LatencyEvaluator(model_name=model_name),
        CostEvaluator(model_name=model_name),
    ]


@app.command()
def run(
    dataset_file: Path = typer.Argument(..., help="Path to JSON dataset file"),
    system_name: str = typer.Option("system-under-test", "--name", "-n", help="System name for the report"),
    provider: str = typer.Option("groq", "--provider", "-p", help="LLM provider: groq, openai, or anthropic"),
    metrics: Optional[str] = typer.Option(None, "--metrics", "-m", help="Comma-separated metrics to run (default: all)"),
    concurrency: int = typer.Option(2, "--concurrency", "-c", help="Max concurrent judge calls (use 2 for Groq free tier, 5+ for paid)"),
    db_path: str = typer.Option("data/results.db", "--db", help="SQLite database path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report JSON to file"),
    doc: Optional[Path] = typer.Option(None, "--doc", help="If set, build a real RAG system from this .txt or .pdf and evaluate it"),
):
    """Evaluate a system against a QA dataset.

    The dataset file should be a JSON array of objects with keys:
        question, answer, context (optional)

    By default uses a mock echo system (answers = reference answers).
    Pass --doc to evaluate a real RAG pipeline built on that document.

    Examples:
        llm-eval run data/rag_dataset.json --name baseline
        llm-eval run data/rag_dataset.json --doc data/knowledge_base.txt --name rag-v1
    """
    configure_logging()

    if not dataset_file.exists():
        console.print(f"[red]Dataset file not found: {dataset_file}[/red]")
        raise typer.Exit(1)

    if doc and not doc.exists():
        console.print(f"[red]Document not found: {doc}[/red]")
        raise typer.Exit(1)

    async def _run():
        from eval_framework.judges.pipeline import EvaluationPipeline

        # Load dataset
        with open(dataset_file) as f:
            raw_data = json.load(f)
        dataset = [
            QAPair(
                question=item["question"],
                answer=item["answer"],
                context=item.get("context"),
            )
            for item in raw_data
        ]
        console.print(f"[green]Loaded {len(dataset)} examples from {dataset_file}[/green]")

        # Build the system under test
        if doc:
            from eval_framework.rag.pipeline import RAGPipeline
            console.print(f"[green]Building RAG index from {doc}...[/green]")
            with console.status("[bold green]Indexing document..."):
                rag = RAGPipeline(
                    doc_path=doc,
                    model_name=_get_model_name(provider),
                ).build()
            system_fn = rag.query
            console.print(f"[green]RAG pipeline ready[/green]")
        else:
            async def system_fn(qa: QAPair) -> SystemOutput:
                """Mock: echoes back the reference answer."""
                await asyncio.sleep(0.01)
                return SystemOutput(
                    answer=qa.answer,
                    latency_ms=500,
                    cost_usd=0.002,
                    model="mock-echo",
                )
            console.print(
                "[yellow]Using mock echo system (answers = reference answers). "
                "Pass --doc <file> to evaluate a real RAG pipeline.[/yellow]"
            )

        # Build evaluators
        all_evaluators = _get_all_evaluators(provider)
        if metrics:
            wanted = set(m.strip() for m in metrics.split(","))
            all_evaluators = [e for e in all_evaluators if e.metric.value in wanted]
        console.print(f"[green]Running {len(all_evaluators)} evaluators[/green]")

        pipeline = EvaluationPipeline(evaluators=all_evaluators, concurrency=concurrency)

        with console.status(f"[bold green]Evaluating {len(dataset)} examples..."):
            report = await pipeline.run(
                system=system_fn,
                dataset=dataset,
                system_name=system_name,
            )

        # Save to database
        db = EvalDatabase(db_path)
        db.save_report(report)
        console.print(f"[green]Saved report {report.id} to {db_path}[/green]")

        # Print results table
        _print_report_table(report)

        # Optionally save JSON
        if output:
            with open(output, "w") as f:
                json.dump(
                    {
                        "id": report.id,
                        "system_name": report.system_name,
                        "timestamp": report.timestamp.isoformat(),
                        "summary_scores": {
                            k.value if hasattr(k, "value") else k: v
                            for k, v in report.summary_scores.items()
                        },
                        "total_examples": report.total_examples_evaluated,
                        "total_cost": report.total_cost,
                    },
                    f,
                    indent=2,
                )
            console.print(f"[green]Report JSON saved to {output}[/green]")

    asyncio.run(_run())


@app.command()
def generate(
    input_file: Path = typer.Argument(..., help="Text file or directory of .txt files to generate from"),
    output_file: Path = typer.Option(Path("dataset.json"), "--output", "-o", help="Output JSON file"),
    n_questions: int = typer.Option(3, "--n-questions", "-n", help="Questions per chunk"),
    provider: str = typer.Option("groq", "--provider", "-p", help="LLM provider: groq, openai, or anthropic"),
    no_validate: bool = typer.Option(False, "--no-validate", help="Skip quality validation pass"),
):
    """Generate a synthetic QA dataset from text documents.

    Example:
        llm-eval generate corpus.txt --n-questions 5 --output my_dataset.json
    """
    configure_logging()

    async def _run():
        from eval_framework.dataset.generator import DatasetGenerator

        client = get_llm_client(provider)
        generator = DatasetGenerator(
            llm_client=client,
            validate=not no_validate,
        )

        # Load documents
        documents = []
        if input_file.is_dir():
            for txt_file in input_file.glob("*.txt"):
                documents.append(txt_file.read_text(encoding="utf-8"))
            console.print(f"[green]Loaded {len(documents)} .txt files from {input_file}[/green]")
        else:
            documents.append(input_file.read_text(encoding="utf-8"))
            console.print(f"[green]Loaded {input_file}[/green]")

        with console.status("[bold green]Generating dataset..."):
            examples = await generator.generate_from_documents(
                documents=documents,
                n_questions_per_chunk=n_questions,
            )

        # Save as JSON
        output_data = [
            {
                "question": ex.qa_pair.question,
                "answer": ex.qa_pair.answer,
                "context": ex.qa_pair.context,
                "difficulty": ex.difficulty,
                "tags": ex.tags,
                "id": ex.id,
            }
            for ex in examples
        ]
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        console.print(f"[green]Generated {len(examples)} QA pairs → {output_file}[/green]")

    asyncio.run(_run())


@app.command()
def report(
    system_name: Optional[str] = typer.Option(None, "--system", "-s", help="Filter by system name"),
    db_path: str = typer.Option("data/results.db", "--db", help="SQLite database path"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max runs to show"),
):
    """List historical evaluation runs."""
    db = EvalDatabase(db_path)

    if system_name:
        runs = db.get_runs_for_system(system_name)
    else:
        runs = db.list_all_runs(limit=limit)

    if not runs:
        console.print("[yellow]No evaluation runs found.[/yellow]")
        return

    table = Table(title="Evaluation History", show_header=True, header_style="bold magenta")
    table.add_column("System", style="cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Examples")
    table.add_column("Faithfulness")
    table.add_column("Relevance")
    table.add_column("Completeness")
    table.add_column("Run ID", style="dim")

    for run in runs:
        scores = run["summary_scores"]
        table.add_row(
            run["system_name"],
            run["timestamp"][:19],
            str(run["total_examples"]),
            f"{scores.get('faithfulness', '-'):.2f}" if scores.get('faithfulness') is not None else "-",
            f"{scores.get('relevance', '-'):.2f}" if scores.get('relevance') is not None else "-",
            f"{scores.get('completeness', '-'):.2f}" if scores.get('completeness') is not None else "-",
            run["id"][:8] + "...",
        )

    console.print(table)


@app.command()
def compare(
    run_id_1: str = typer.Argument(..., help="First run ID"),
    run_id_2: str = typer.Argument(..., help="Second run ID"),
    db_path: str = typer.Option("data/results.db", "--db", help="SQLite database path"),
):
    """Compare two evaluation runs side-by-side."""
    db = EvalDatabase(db_path)

    try:
        comparison = db.compare_runs(run_id_1, run_id_2)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Run 1:[/bold] {comparison['run1']['system']} @ {comparison['run1']['timestamp'][:19]}\n"
        f"[bold]Run 2:[/bold] {comparison['run2']['system']} @ {comparison['run2']['timestamp'][:19]}",
        title="Comparing Evaluation Runs",
    ))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Run 1")
    table.add_column("Run 2")
    table.add_column("Delta")
    table.add_column("Change")

    for metric, data in comparison["metrics"].items():
        delta = data["delta"]
        delta_color = "green" if delta > 0 else "red" if delta < 0 else "dim"
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "─"
        table.add_row(
            metric,
            f"{data['run1_score']:.3f}",
            f"{data['run2_score']:.3f}",
            f"[{delta_color}]{delta:+.3f}[/{delta_color}]",
            f"[{delta_color}]{arrow}[/{delta_color}]",
        )

    console.print(table)


@app.command(name="rag-eval")
def rag_eval(
    doc_path: Path = typer.Argument(..., help="Path to .txt or .pdf knowledge base"),
    dataset_file: Path = typer.Option(Path("data/rag_dataset.json"), "--dataset", "-d", help="QA dataset JSON"),
    system_name: str = typer.Option("rag-langchain-groq", "--name", "-n", help="System name for the report"),
    concurrency: int = typer.Option(2, "--concurrency", "-c", help="Max concurrent Groq calls"),
    db_path: str = typer.Option("data/results.db", "--db", help="SQLite database path"),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Document chunk size in characters"),
    top_k: int = typer.Option(3, "--top-k", help="Number of chunks to retrieve per question"),
):
    """Evaluate a real RAG system built on a document.

    Builds a LangChain + FAISS + Groq RAG pipeline on your document,
    runs it against the dataset, evaluates every answer, and saves results.

    Examples:
        llm-eval rag-eval data/knowledge_base.txt
        llm-eval rag-eval my_report.pdf --name rag-v2 --top-k 5
    """
    configure_logging()

    if not doc_path.exists():
        console.print(f"[red]Document not found: {doc_path}[/red]")
        raise typer.Exit(1)

    if not dataset_file.exists():
        console.print(f"[red]Dataset not found: {dataset_file}[/red]")
        raise typer.Exit(1)

    async def _run():
        import json
        from eval_framework.rag.pipeline import RAGPipeline
        from eval_framework.judges.pipeline import EvaluationPipeline
        from eval_framework.evaluators.conciseness import ConcisenessEvaluator
        from eval_framework.evaluators.coherence import CoherenceEvaluator

        settings = get_settings()
        model_name = _get_model_name("groq")

        # Build RAG system
        console.print(f"[green]Building RAG index from {doc_path}...[/green]")
        rag = RAGPipeline(
            doc_path=doc_path,
            chunk_size=chunk_size,
            top_k=top_k,
            model_name=model_name,
        ).build()

        # Load dataset
        with open(dataset_file) as f:
            raw_data = json.load(f)
        dataset = [
            QAPair(
                question=item["question"],
                answer=item["answer"],
                context=item.get("context"),
            )
            for item in raw_data
        ]
        console.print(f"[green]Loaded {len(dataset)} questions[/green]")

        # Build evaluators
        client = get_llm_client("groq")
        evaluators = [
            FaithfulnessEvaluator(client, model_name),
            RelevanceEvaluator(client, model_name),
            CompletenessEvaluator(client, model_name),
            HallucinationRateEvaluator(client, model_name),
            ConcisenessEvaluator(client, model_name),
            CoherenceEvaluator(client, model_name),
            LatencyEvaluator(model_name=model_name),
            CostEvaluator(model_name=model_name),
        ]

        pipeline = EvaluationPipeline(evaluators=evaluators, concurrency=concurrency)

        with console.status(f"[bold green]Evaluating {len(dataset)} questions..."):
            report = await pipeline.run(
                system=rag.query,
                dataset=dataset,
                system_name=system_name,
            )

        # Save to database
        db = EvalDatabase(db_path)
        db.save_report(report)
        console.print(f"[green]Saved report {report.id[:8]}... to {db_path}[/green]")

        _print_report_table(report)

    asyncio.run(_run())


def _print_report_table(report) -> None:
    """Print a formatted summary table for an evaluation report."""
    table = Table(
        title=f"Evaluation Results: {report.system_name}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", min_width=20)
    table.add_column("Score", min_width=10)
    table.add_column("Examples Evaluated", min_width=20)
    table.add_column("Rating", min_width=12)

    for metric, score in report.summary_scores.items():
        metric_name = metric.value if hasattr(metric, 'value') else metric
        n_examples = len(report.results.get(metric, []))

        if score >= 0.8:
            rating = "[green]EXCELLENT[/green]"
        elif score >= 0.6:
            rating = "[yellow]GOOD[/yellow]"
        elif score >= 0.4:
            rating = "[orange1]FAIR[/orange1]"
        else:
            rating = "[red]POOR[/red]"

        table.add_row(metric_name, f"{score:.3f}", str(n_examples), rating)

    console.print(table)
    console.print(
        f"\n[dim]Run ID: {report.id} | "
        f"Total cost: ${report.total_cost:.4f} | "
        f"Examples: {report.total_examples_evaluated}[/dim]"
    )


if __name__ == "__main__":
    app()
