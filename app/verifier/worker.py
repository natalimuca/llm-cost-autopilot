"""Background asyncio worker that drains a queue of verification jobs.

Kept as an in-process asyncio.Queue (not Celery/Redis) because this is a
portfolio-scale system — one FastAPI process plus one worker task is enough
to demonstrate the async verify -> escalate -> feedback loop end to end.
"""
import asyncio
import logging
from dataclasses import dataclass

from app.db.database import record_escalation, record_verification
from app.models.response import Response
from app.verifier.verifier import auto_escalate_enabled, is_routing_failure, verify

logger = logging.getLogger("autopilot.verifier")


@dataclass
class VerificationJob:
    request_id: int
    prompt: str
    routed_tier: int
    candidate_response: Response


_queue: asyncio.Queue[VerificationJob] = asyncio.Queue()


def enqueue(job: VerificationJob) -> None:
    _queue.put_nowait(job)


async def _process(job: VerificationJob) -> None:
    score, reference = await verify(job.prompt, job.candidate_response)
    failure = is_routing_failure(score)
    correct_tier = min(job.routed_tier + 1, 3) if failure else None

    record_verification(
        job.request_id,
        quality_score=score,
        is_routing_failure=failure,
        correct_tier=correct_tier,
    )

    if failure and auto_escalate_enabled():
        cost_delta = reference.cost_usd - job.candidate_response.cost_usd
        record_escalation(
            job.request_id,
            escalated_to_model=reference.model_id,
            escalation_cost_delta=cost_delta,
        )
        logger.info(
            "escalation: request=%s score=%.1f routed_tier=%s -> %s (+$%.6f)",
            job.request_id, score, job.routed_tier, reference.model_id, cost_delta,
        )


async def run_worker() -> None:
    """Long-running consumer loop. Start as an asyncio task at app startup."""
    while True:
        job = await _queue.get()
        try:
            await _process(job)
        except Exception:  # noqa: BLE001 - never let one bad job kill the worker
            logger.exception("verification job failed for request_id=%s", job.request_id)
        finally:
            _queue.task_done()
