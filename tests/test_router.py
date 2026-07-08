from unittest.mock import AsyncMock, patch

import pytest

from app.router import router as router_module

FAKE_CONFIG = {
    "classifier_backend": "llm",
    "tier_to_model": {1: "llama-local", 2: "gpt-4o-mini", 3: "gpt-4o"},
    "min_confidence": 0.55,
    "judge_model": "gpt-4o",
}


def _mock_classify(tier: int, confidence: float):
    return patch.object(router_module.llm_classifier, "classify", new=AsyncMock(return_value=(tier, confidence)))


@pytest.mark.asyncio
async def test_route_escalates_on_low_confidence():
    with patch.object(router_module, "load_routing_config", return_value=FAKE_CONFIG), _mock_classify(1, 0.3):
        decision = await router_module.route("some prompt")

    assert decision.escalated_pre_send is True
    assert decision.prompt_tier == 2
    assert decision.model_name == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_route_does_not_escalate_past_tier_3():
    with patch.object(router_module, "load_routing_config", return_value=FAKE_CONFIG), _mock_classify(3, 0.1):
        decision = await router_module.route("some prompt")

    assert decision.escalated_pre_send is False
    assert decision.prompt_tier == 3


@pytest.mark.asyncio
async def test_route_no_escalation_when_confident():
    with patch.object(router_module, "load_routing_config", return_value=FAKE_CONFIG), _mock_classify(1, 0.9):
        decision = await router_module.route("some prompt")

    assert decision.escalated_pre_send is False
    assert decision.prompt_tier == 1
    assert decision.model_name == "llama-local"


@pytest.mark.asyncio
async def test_route_reassigns_for_latency_when_faster_option_exists():
    # tier 3 -> gpt-4o (2.5s avg, HIGH quality). claude-sonnet (2.2s, HIGH)
    # fits a 2.3s budget that gpt-4o itself doesn't.
    with patch.object(router_module, "load_routing_config", return_value=FAKE_CONFIG), _mock_classify(3, 0.95):
        decision = await router_module.route("some prompt", max_latency_seconds=2.3)

    assert decision.reassigned_for_latency is True
    assert decision.model_name == "claude-sonnet"


@pytest.mark.asyncio
async def test_route_keeps_original_when_no_faster_option_meets_budget():
    with patch.object(router_module, "load_routing_config", return_value=FAKE_CONFIG), _mock_classify(3, 0.95):
        decision = await router_module.route("some prompt", max_latency_seconds=0.1)

    # No HIGH-quality model meets an unrealistic 0.1s budget -> falls back unchanged.
    assert decision.reassigned_for_latency is False
    assert decision.model_name == "gpt-4o"
