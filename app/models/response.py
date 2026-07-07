from dataclasses import dataclass


@dataclass
class Response:
    """Standardized output from any provider's send_request call."""

    text: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
