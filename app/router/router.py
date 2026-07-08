"""Ties the complexity classifier to the tier->model map in config/routing.yaml."""
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import yaml

from app.classifier import classifier as tfidf_classifier
from app.classifier import llm_classifier
from app.models.registry import MODEL_REGISTRY, ModelConfig, QualityTier, get_model

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "routing.yaml"

_lock = Lock()
_config_cache: dict | None = None

_QUALITY_RANK = {QualityTier.LOW: 1, QualityTier.MEDIUM: 2, QualityTier.HIGH: 3}


@dataclass
class RoutingDecision:
    prompt_tier: int
    confidence: float
    model_name: str
    model_config: ModelConfig
    escalated_pre_send: bool
    reassigned_for_latency: bool = False


def load_routing_config(force_reload: bool = False) -> dict:
    global _config_cache
    with _lock:
        if _config_cache is None or force_reload:
            with CONFIG_PATH.open() as f:
                _config_cache = yaml.safe_load(f)
        return _config_cache


def update_routing_config(new_config: dict) -> None:
    """Persist an updated tier->model map so it survives restarts, and refresh the cache."""
    global _config_cache
    with _lock:
        with CONFIG_PATH.open("w") as f:
            yaml.safe_dump(new_config, f, sort_keys=False)
        _config_cache = new_config


def _fastest_meeting_quality(min_quality: QualityTier, max_latency: float) -> tuple[str, ModelConfig] | None:
    """Cheapest-in-latency alternative that still meets the tier's quality bar.

    Used when a caller has a hard latency budget (e.g. a synchronous chat UI)
    that the tier's normally-assigned model can't hit -- trades cost for
    speed without dropping quality below what the prompt actually needed.
    """
    candidates = [
        (name, cfg)
        for name, cfg in MODEL_REGISTRY.items()
        if _QUALITY_RANK[cfg.quality_tier] >= _QUALITY_RANK[min_quality]
        and cfg.avg_latency_seconds <= max_latency
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair[1].avg_latency_seconds)


async def route(prompt: str, max_latency_seconds: float | None = None) -> RoutingDecision:
    config = load_routing_config()
    backend = config.get("classifier_backend", "llm")
    if backend == "llm":
        tier, confidence = await llm_classifier.classify(prompt)
    else:
        tier, confidence = tfidf_classifier.classify(prompt)

    escalated = False
    if confidence < config["min_confidence"] and tier < 3:
        tier += 1
        escalated = True

    model_name = config["tier_to_model"][tier]
    model_config = get_model(model_name)

    reassigned_for_latency = False
    if max_latency_seconds is not None and model_config.avg_latency_seconds > max_latency_seconds:
        faster = _fastest_meeting_quality(model_config.quality_tier, max_latency_seconds)
        if faster is not None:
            model_name, model_config = faster
            reassigned_for_latency = True

    return RoutingDecision(
        prompt_tier=tier,
        confidence=confidence,
        model_name=model_name,
        model_config=model_config,
        escalated_pre_send=escalated,
        reassigned_for_latency=reassigned_for_latency,
    )
