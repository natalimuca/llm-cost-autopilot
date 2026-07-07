import hashlib

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatMessage, CompletionRequest, CompletionResponse, RoutingConfigUpdate
from app.db.database import get_stats, log_request
from app.logging_config import log_request_event
from app.models.interface import send_request
from app.models.registry import MODEL_REGISTRY, get_model
from app.router.router import load_routing_config, route, update_routing_config
from app.verifier.worker import VerificationJob, enqueue

router = APIRouter(prefix="/v1")


def _last_user_prompt(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    raise HTTPException(status_code=422, detail="No user message found in `messages`")


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(payload: CompletionRequest) -> CompletionResponse:
    prompt = _last_user_prompt(payload.messages)

    decision = route(prompt, max_latency_seconds=payload.max_latency_seconds)
    response = await send_request(prompt, decision.model_config)

    request_id = log_request(
        prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt=prompt,
        complexity_tier=decision.prompt_tier,
        confidence=decision.confidence,
        routed_model=decision.model_name,
        provider=response.provider,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_seconds=response.latency_seconds,
    )

    log_request_event(
        request_id=request_id,
        prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
        complexity_tier=decision.prompt_tier,
        confidence=decision.confidence,
        routed_model=decision.model_name,
        provider=response.provider,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_seconds=response.latency_seconds,
        escalated_pre_send=decision.escalated_pre_send,
        reassigned_for_latency=decision.reassigned_for_latency,
    )

    enqueue(VerificationJob(
        request_id=request_id,
        prompt=prompt,
        routed_tier=decision.prompt_tier,
        candidate_response=response,
    ))

    return CompletionResponse(
        text=response.text,
        routed_model=decision.model_name,
        provider=response.provider,
        complexity_tier=decision.prompt_tier,
        classifier_confidence=decision.confidence,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_seconds=response.latency_seconds,
        request_id=request_id,
        reassigned_for_latency=decision.reassigned_for_latency,
    )


@router.get("/models")
async def list_models() -> dict:
    return {
        name: {
            "provider": config.provider.value,
            "model_id": config.model_id,
            "cost_per_1m_input": config.cost_per_1m_input,
            "cost_per_1m_output": config.cost_per_1m_output,
            "avg_latency_seconds": config.avg_latency_seconds,
            "quality_tier": config.quality_tier.value,
        }
        for name, config in MODEL_REGISTRY.items()
    }


@router.get("/stats")
async def stats() -> dict:
    return get_stats()


@router.get("/routing-config")
async def get_routing_config() -> dict:
    return load_routing_config()


@router.put("/routing-config")
async def put_routing_config(payload: RoutingConfigUpdate) -> dict:
    for model_name in payload.tier_to_model.values():
        try:
            get_model(model_name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    new_config = load_routing_config().copy()
    new_config["tier_to_model"] = payload.tier_to_model
    if payload.min_confidence is not None:
        new_config["min_confidence"] = payload.min_confidence
    if payload.judge_model is not None:
        new_config["judge_model"] = payload.judge_model

    update_routing_config(new_config)
    return new_config
