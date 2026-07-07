from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_latency_seconds: float | None = Field(
        default=None,
        description="If set and the tier's assigned model's avg latency exceeds this, "
        "route to the fastest model that still meets the required quality tier.",
    )


class CompletionResponse(BaseModel):
    text: str
    routed_model: str
    provider: str
    complexity_tier: int
    classifier_confidence: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float
    request_id: int
    reassigned_for_latency: bool


class RoutingConfigUpdate(BaseModel):
    tier_to_model: dict[int, str]
    min_confidence: float | None = None
    judge_model: str | None = None
