"""Model registry: cost, latency, and quality-tier metadata for every model
the router is allowed to pick between.

Prices are USD per 1M tokens, current as of the provider's published pricing
at the time this file was written. Update MODEL_REGISTRY as pricing changes —
nothing else in the codebase hardcodes prices.
"""
from dataclasses import dataclass
from enum import Enum


class QualityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class ModelConfig:
    provider: Provider
    model_id: str
    cost_per_1m_input: float   # USD
    cost_per_1m_output: float  # USD
    avg_latency_seconds: float
    quality_tier: QualityTier

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.cost_per_1m_input
            + output_tokens / 1_000_000 * self.cost_per_1m_output
        )


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
        avg_latency_seconds=2.5,
        quality_tier=QualityTier.HIGH,
    ),
    "gpt-4o-mini": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        avg_latency_seconds=1.2,
        quality_tier=QualityTier.MEDIUM,
    ),
    "claude-sonnet": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-sonnet-5",
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00,
        avg_latency_seconds=2.2,
        quality_tier=QualityTier.HIGH,
    ),
    "claude-haiku": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-haiku-4-5-20251001",
        cost_per_1m_input=0.25,
        cost_per_1m_output=1.25,
        avg_latency_seconds=0.9,
        quality_tier=QualityTier.LOW,
    ),
    "llama-local": ModelConfig(
        provider=Provider.OLLAMA,
        model_id="llama3.1:8b",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        avg_latency_seconds=1.5,
        quality_tier=QualityTier.LOW,
    ),
}


def get_model(name: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}"
        ) from None
