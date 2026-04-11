"""SQLite storage layer for evaluation results.

Why SQLite?
- Zero setup: no database server to run
- File-based: easy to version-control or share
- Fast enough for thousands of eval runs
- Perfect for local development and CI/CD pipelines

Schema design:
- evaluation_runs: One row per EvaluationReport (the "run" metadata)
- evaluation_results: One row per metric score (the individual judge results)

Teaching note: This is a classic "report header + line items" schema pattern.
The header (evaluation_runs) stores summary info, the line items
(evaluation_results) store individual scores. Same pattern used in invoices,
orders, and every reporting system you'll ever build.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..types import (
    EvaluationReport,
    EvaluationResult,
    EvaluationMetric,
)

logger = logging.getLogger(__name__)


CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    system_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    total_examples INTEGER NOT NULL,
    summary_scores TEXT NOT NULL,   -- JSON: {"faithfulness": 0.87, ...}
    judge_models TEXT NOT NULL,     -- JSON: ["gpt-4", ...]
    total_cost REAL NOT NULL DEFAULT 0.0,
    metadata TEXT NOT NULL DEFAULT '{}'  -- JSON blob for extra fields
);
"""

CREATE_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES evaluation_runs(id),
    metric TEXT NOT NULL,
    score REAL NOT NULL,
    raw_score REAL,
    reasoning TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    timestamp TEXT NOT NULL
);
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_results_run_id ON evaluation_results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_system_name ON evaluation_runs(system_name);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON evaluation_runs(timestamp);
"""


class EvalDatabase:
    """SQLite-backed storage for evaluation runs and results.

    Usage:
        db = EvalDatabase("data/results.db")

        # Save a report
        db.save_report(report)

        # Load all runs for a system
        runs = db.get_runs_for_system("my-rag-v2")

        # Compare two runs
        db.compare_runs(run_id_1, run_id_2)
    """

    def __init__(self, db_path: str = "data/results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection with row factory for dict-like access."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Rows accessible as dicts
        return conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.execute(CREATE_RUNS_TABLE)
            conn.execute(CREATE_RESULTS_TABLE)
            # CREATE INDEX can't be in a multi-statement execute, so split:
            for stmt in CREATE_INDEX.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
        logger.debug(f"Database initialized at {self.db_path}")

    def save_report(self, report: EvaluationReport) -> None:
        """Persist a full EvaluationReport to the database."""
        with self._get_conn() as conn:
            # Insert the run header
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs
                    (id, system_name, timestamp, total_examples,
                     summary_scores, judge_models, total_cost, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.id,
                    report.system_name,
                    report.timestamp.isoformat(),
                    report.total_examples_evaluated,
                    json.dumps({k.value if hasattr(k, 'value') else k: v
                                for k, v in report.summary_scores.items()}),
                    json.dumps(report.judge_models_used),
                    report.total_cost,
                    json.dumps(report.metadata),
                ),
            )

            # Insert individual results (all metrics, all examples)
            for metric, results in report.results.items():
                for result in results:
                    conn.execute(
                        """
                        INSERT INTO evaluation_results
                            (run_id, metric, score, raw_score, reasoning,
                             judge_model, confidence, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            report.id,
                            result.metric.value if hasattr(result.metric, 'value') else result.metric,
                            result.score,
                            result.raw_score,
                            result.reasoning,
                            result.judge_model,
                            result.confidence,
                            result.timestamp.isoformat(),
                        ),
                    )

        logger.info(f"Saved report {report.id} for system '{report.system_name}'")

    def get_run(self, run_id: str) -> Optional[dict]:
        """Load a single run's metadata by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["summary_scores"] = json.loads(result["summary_scores"])
        result["judge_models"] = json.loads(result["judge_models"])
        result["metadata"] = json.loads(result["metadata"])
        return result

    def get_runs_for_system(self, system_name: str) -> list[dict]:
        """Load all runs for a given system, newest first."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM evaluation_runs
                WHERE system_name = ?
                ORDER BY timestamp DESC
                """,
                (system_name,),
            ).fetchall()
        runs = []
        for row in rows:
            r = dict(row)
            r["summary_scores"] = json.loads(r["summary_scores"])
            r["judge_models"] = json.loads(r["judge_models"])
            r["metadata"] = json.loads(r["metadata"])
            runs.append(r)
        return runs

    def list_systems(self) -> list[str]:
        """List all system names that have been evaluated."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT system_name FROM evaluation_runs ORDER BY system_name"
            ).fetchall()
        return [row["system_name"] for row in rows]

    def list_all_runs(self, limit: int = 50) -> list[dict]:
        """List recent evaluation runs across all systems."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM evaluation_runs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        runs = []
        for row in rows:
            r = dict(row)
            r["summary_scores"] = json.loads(r["summary_scores"])
            runs.append(r)
        return runs

    def get_results_for_run(
        self,
        run_id: str,
        metric: Optional[str] = None,
    ) -> list[dict]:
        """Load all individual results for a run, optionally filtered by metric."""
        query = "SELECT * FROM evaluation_results WHERE run_id = ?"
        params: tuple = (run_id,)
        if metric:
            query += " AND metric = ?"
            params = (run_id, metric)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def compare_runs(self, run_id_1: str, run_id_2: str) -> dict:
        """Compare two runs side-by-side: shows score delta per metric.

        Returns a dict: {metric: {run1: score, run2: score, delta: diff}}
        """
        run1 = self.get_run(run_id_1)
        run2 = self.get_run(run_id_2)

        if not run1 or not run2:
            raise ValueError("One or both run IDs not found")

        comparison = {}
        all_metrics = set(run1["summary_scores"]) | set(run2["summary_scores"])

        for metric in all_metrics:
            s1 = run1["summary_scores"].get(metric, 0.0)
            s2 = run2["summary_scores"].get(metric, 0.0)
            comparison[metric] = {
                "run1_score": s1,
                "run2_score": s2,
                "delta": s2 - s1,  # positive = run2 improved
                "improved": s2 > s1,
            }

        return {
            "run1": {"id": run_id_1, "system": run1["system_name"], "timestamp": run1["timestamp"]},
            "run2": {"id": run_id_2, "system": run2["system_name"], "timestamp": run2["timestamp"]},
            "metrics": comparison,
        }

    def get_score_history(
        self,
        system_name: str,
        metric: str,
    ) -> list[dict]:
        """Get the score trend for a metric over time (for dashboard charts)."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT r.timestamp, r.id,
                       json_extract(r.summary_scores, '$.' || ?) as score
                FROM evaluation_runs r
                WHERE r.system_name = ?
                ORDER BY r.timestamp ASC
                """,
                (metric, system_name),
            ).fetchall()
        return [{"timestamp": row[0], "run_id": row[1], "score": row[2]} for row in rows]
