from unittest.mock import AsyncMock, patch

import pytest

from app.models.response import Response
from app.verifier import worker as worker_module
from app.verifier.verifier import _parse_score, auto_escalate_enabled, is_routing_failure


def test_parse_score_extracts_leading_digit():
    assert _parse_score("4") == 4.0
    assert _parse_score("The score is 3.5 out of 5") == 3.5


def test_parse_score_falls_back_to_1_when_unparseable():
    assert _parse_score("no number here") == 1.0


def test_is_routing_failure_respects_threshold(monkeypatch):
    monkeypatch.setenv("QUALITY_THRESHOLD", "4.0")
    assert is_routing_failure(3.9) is True
    assert is_routing_failure(4.0) is False


def test_auto_escalate_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("AUTO_ESCALATE", "false")
    assert auto_escalate_enabled() is False
    monkeypatch.setenv("AUTO_ESCALATE", "true")
    assert auto_escalate_enabled() is True


def _response(cost: float, model_id: str = "gpt-4o") -> Response:
    return Response(
        text="text", model_id=model_id, provider="openai",
        input_tokens=10, output_tokens=5, latency_seconds=0.1, cost_usd=cost,
    )


@pytest.mark.asyncio
async def test_process_records_pass_without_escalation(monkeypatch):
    monkeypatch.setenv("QUALITY_THRESHOLD", "4.0")
    job = worker_module.VerificationJob(
        request_id=1, prompt="p", routed_tier=1, candidate_response=_response(0.0001)
    )

    with patch.object(worker_module, "verify", new=AsyncMock(return_value=(4.5, _response(0.01)))), \
         patch.object(worker_module, "record_verification") as record_verification, \
         patch.object(worker_module, "record_escalation") as record_escalation:
        await worker_module._process(job)

    record_verification.assert_called_once_with(
        1, quality_score=4.5, is_routing_failure=False, correct_tier=None
    )
    record_escalation.assert_not_called()


@pytest.mark.asyncio
async def test_process_escalates_on_failure(monkeypatch):
    monkeypatch.setenv("QUALITY_THRESHOLD", "4.0")
    monkeypatch.setenv("AUTO_ESCALATE", "true")
    job = worker_module.VerificationJob(
        request_id=2, prompt="p", routed_tier=1, candidate_response=_response(0.0001)
    )

    with patch.object(worker_module, "verify", new=AsyncMock(return_value=(2.0, _response(0.01, "gpt-4o")))), \
         patch.object(worker_module, "record_verification") as record_verification, \
         patch.object(worker_module, "record_escalation") as record_escalation:
        await worker_module._process(job)

    record_verification.assert_called_once_with(
        2, quality_score=2.0, is_routing_failure=True, correct_tier=2
    )
    record_escalation.assert_called_once_with(
        2, escalated_to_model="gpt-4o", escalation_cost_delta=pytest.approx(0.0099)
    )


@pytest.mark.asyncio
async def test_process_skips_escalation_when_disabled(monkeypatch):
    monkeypatch.setenv("QUALITY_THRESHOLD", "4.0")
    monkeypatch.setenv("AUTO_ESCALATE", "false")
    job = worker_module.VerificationJob(
        request_id=3, prompt="p", routed_tier=1, candidate_response=_response(0.0001)
    )

    with patch.object(worker_module, "verify", new=AsyncMock(return_value=(2.0, _response(0.01)))), \
         patch.object(worker_module, "record_verification"), \
         patch.object(worker_module, "record_escalation") as record_escalation:
        await worker_module._process(job)

    record_escalation.assert_not_called()


@pytest.mark.asyncio
async def test_process_caps_correct_tier_at_3(monkeypatch):
    monkeypatch.setenv("QUALITY_THRESHOLD", "4.0")
    job = worker_module.VerificationJob(
        request_id=4, prompt="p", routed_tier=3, candidate_response=_response(0.01)
    )

    with patch.object(worker_module, "verify", new=AsyncMock(return_value=(1.0, _response(0.02)))), \
         patch.object(worker_module, "record_verification") as record_verification, \
         patch.object(worker_module, "record_escalation"):
        await worker_module._process(job)

    record_verification.assert_called_once_with(
        4, quality_score=1.0, is_routing_failure=True, correct_tier=3
    )
