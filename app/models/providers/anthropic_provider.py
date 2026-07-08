import os
import time

import anthropic
from anthropic import AsyncAnthropic

from app.models.providers.base import BaseProvider
from app.models.providers.retry import with_retry
from app.models.registry import ModelConfig
from app.models.response import Response

DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str | None = None):
        self._client = AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @with_retry(anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError)
    async def send(self, prompt: str, config: ModelConfig) -> Response:
        start = time.perf_counter()
        message = await self._client.messages.create(
            model=config.model_id,
            max_tokens=DEFAULT_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start

        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        text = "".join(block.text for block in message.content if block.type == "text")

        return Response(
            text=text,
            model_id=config.model_id,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            cost_usd=config.cost_for(input_tokens, output_tokens),
        )
