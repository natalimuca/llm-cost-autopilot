"""SQLite audit trail: one row per request, updated in place once the async
verifier scores it. This is the single source of truth for the dashboard.
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.environ.get("DATABASE_PATH", "./data/autopilot.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt TEXT NOT NULL,
    complexity_tier INTEGER NOT NULL,
    confidence REAL NOT NULL,
    routed_model TEXT NOT NULL,
    provider TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_seconds REAL NOT NULL,
    quality_score REAL,
    is_routing_failure INTEGER NOT NULL DEFAULT 0,
    escalated INTEGER NOT NULL DEFAULT 0,
    escalated_to_model TEXT,
    escalation_cost_delta REAL,
    correct_tier INTEGER,
    verified_at TEXT,
    fed_to_training INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_routing_failure ON requests(is_routing_failure);
"""


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def log_request(
    *,
    prompt_hash: str,
    prompt: str,
    complexity_tier: int,
    confidence: float,
    routed_model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_seconds: float,
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO requests (
                timestamp, prompt_hash, prompt, complexity_tier, confidence,
                routed_model, provider, input_tokens, output_tokens,
                cost_usd, latency_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                prompt_hash,
                prompt,
                complexity_tier,
                confidence,
                routed_model,
                provider,
                input_tokens,
                output_tokens,
                cost_usd,
                latency_seconds,
            ),
        )
        return cur.lastrowid


def record_verification(
    request_id: int,
    *,
    quality_score: float,
    is_routing_failure: bool,
    correct_tier: int | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE requests
            SET quality_score = ?, is_routing_failure = ?, correct_tier = ?,
                verified_at = ?
            WHERE id = ?
            """,
            (
                quality_score,
                int(is_routing_failure),
                correct_tier,
                datetime.now(timezone.utc).isoformat(),
                request_id,
            ),
        )


def record_escalation(
    request_id: int, *, escalated_to_model: str, escalation_cost_delta: float
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE requests
            SET escalated = 1, escalated_to_model = ?, escalation_cost_delta = ?
            WHERE id = ?
            """,
            (escalated_to_model, escalation_cost_delta, request_id),
        )


def get_unfed_routing_failures() -> list[sqlite3.Row]:
    """Routing failures not yet folded into the classifier's training set."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, prompt, correct_tier FROM requests "
            "WHERE is_routing_failure = 1 AND correct_tier IS NOT NULL AND fed_to_training = 0"
        ).fetchall()


def mark_fed_to_training(request_ids: list[int]) -> None:
    with get_connection() as conn:
        conn.executemany(
            "UPDATE requests SET fed_to_training = 1 WHERE id = ?",
            [(rid,) for rid in request_ids],
        )


def get_stats(reference_model_cost_per_1m_input: float = 2.50, reference_model_cost_per_1m_output: float = 10.00) -> dict:
    """Cost/quality summary for the dashboard, incl. the 'what if we sent
    everything to the reference (highest-tier) model' comparison."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT routed_model, cost_usd, input_tokens, output_tokens, "
            "quality_score, escalated FROM requests"
        ).fetchall()

    total_requests = len(rows)
    total_cost = sum(r["cost_usd"] for r in rows)
    reference_cost = sum(
        r["input_tokens"] / 1_000_000 * reference_model_cost_per_1m_input
        + r["output_tokens"] / 1_000_000 * reference_model_cost_per_1m_output
        for r in rows
    )
    savings_pct = (
        (reference_cost - total_cost) / reference_cost * 100 if reference_cost > 0 else 0.0
    )

    distribution: dict[str, int] = {}
    for r in rows:
        distribution[r["routed_model"]] = distribution.get(r["routed_model"], 0) + 1

    quality_scores = [r["quality_score"] for r in rows if r["quality_score"] is not None]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None

    escalations = sum(1 for r in rows if r["escalated"])
    escalation_rate = escalations / total_requests * 100 if total_requests else 0.0

    return {
        "total_requests": total_requests,
        "total_cost_usd": total_cost,
        "reference_cost_usd": reference_cost,
        "savings_usd": reference_cost - total_cost,
        "savings_pct": savings_pct,
        "routing_distribution": distribution,
        "avg_quality_score": avg_quality,
        "escalation_count": escalations,
        "escalation_rate_pct": escalation_rate,
    }
