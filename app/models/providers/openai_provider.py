import os
import time

import openai
from openai import AsyncOpenAI

from app.models.providers.base import BaseProvider
from app.models.providers.retry import with_retry
from app.models.registry import ModelConfig
from app.models.response import Response


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str | None = None):
        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @with_retry(openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
    async def send(self, prompt: str, config: ModelConfig) -> Response:
        start = time.perf_counter()
        completion = await self._client.chat.completions.create(
            model=config.model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start

        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return Response(
            text=completion.choices[0].message.content or "",
            model_id=config.model_id,
            provider="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            cost_usd=config.cost_for(input_tokens, output_tokens),
        )
