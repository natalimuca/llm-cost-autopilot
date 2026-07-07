"""LLM-as-judge quality verification.

Runs *after* a response has already gone back to the user (see worker.py).
Sends the same prompt to the judge model, then asks the judge model to score
how well the cheaper model's answer holds up against its own. A low score
means the router picked too cheap a model for this prompt — a routing
failure that both gets logged for the cost/quality dashboard and feeds the
classifier retraining set (Phase 3.4).
"""
import os
import re

from app.models.interface import send_request
from app.models.response import Response

JUDGE_PROMPT_TEMPLATE = """You are grading whether a candidate answer to a prompt is \
acceptable, compared to a reference answer from a stronger model.

Original prompt:
{prompt}

Reference answer (from a high-quality model):
{reference}

Candidate answer (from a cheaper model):
{candidate}

Score the candidate's quality and correctness from 1 to 5, where 5 means it is \
just as good as the reference for this prompt, and 1 means it is unusable or \
substantially wrong/incomplete.

Respond with ONLY a single number from 1 to 5."""


def _quality_threshold() -> float:
    return float(os.environ.get("QUALITY_THRESHOLD", "4.0"))


def _judge_model_name() -> str:
    return os.environ.get("VERIFIER_JUDGE_MODEL", "gpt-4o")


def _parse_score(text: str) -> float:
    match = re.search(r"[1-5](?:\.\d+)?", text)
    return float(match.group()) if match else 1.0


async def verify(prompt: str, candidate_response: Response) -> tuple[float, Response]:
    """Returns (quality_score 1-5, judge's reference Response)."""
    judge_model = _judge_model_name()
    reference = await send_request(prompt, judge_model)

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt, reference=reference.text, candidate=candidate_response.text
    )
    scoring = await send_request(judge_prompt, judge_model)
    score = _parse_score(scoring.text)
    return score, reference


def is_routing_failure(quality_score: float) -> bool:
    return quality_score < _quality_threshold()


def auto_escalate_enabled() -> bool:
    return os.environ.get("AUTO_ESCALATE", "true").lower() == "true"
