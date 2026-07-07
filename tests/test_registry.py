import pytest

from app.models.registry import MODEL_REGISTRY, get_model


def test_all_models_have_positive_latency():
    for config in MODEL_REGISTRY.values():
        assert config.avg_latency_seconds > 0


def test_cost_for_computes_expected_value():
    config = get_model("gpt-4o-mini")
    cost = config.cost_for(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.15 + 0.60)


def test_ollama_is_free():
    config = get_model("llama-local")
    assert config.cost_for(1_000_000, 1_000_000) == 0.0


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        get_model("does-not-exist")
