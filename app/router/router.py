"""Ties the complexity classifier to the tier->model map in config/routing.yaml."""
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import yaml

from app.classifier.classifier import classify
from app.models.registry import ModelConfig, get_model

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "routing.yaml"

_lock = Lock()
_config_cache: dict | None = None


@dataclass
class RoutingDecision:
    prompt_tier: int
    confidence: float
    model_name: str
    model_config: ModelConfig
    escalated_pre_send: bool


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


def route(prompt: str) -> RoutingDecision:
    config = load_routing_config()
    tier, confidence = classify(prompt)

    escalated = False
    if confidence < config["min_confidence"] and tier < 3:
        tier += 1
        escalated = True

    model_name = config["tier_to_model"][tier]
    return RoutingDecision(
        prompt_tier=tier,
        confidence=confidence,
        model_name=model_name,
        model_config=get_model(model_name),
        escalated_pre_send=escalated,
    )
