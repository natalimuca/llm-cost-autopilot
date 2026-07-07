import os
import time

import httpx

from app.models.providers.base import BaseProvider
from app.models.registry import ModelConfig
from app.models.response import Response


class OllamaProvider(BaseProvider):
    """Local models via Ollama's HTTP API. Always $0 cost."""

    def __init__(self, host: str | None = None):
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    async def send(self, prompt: str, config: ModelConfig) -> Response:
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._host}/api/generate",
                json={"model": config.model_id, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
        latency = time.perf_counter() - start

        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        return Response(
            text=data.get("response", ""),
            model_id=config.model_id,
            provider="ollama",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            cost_usd=0.0,
        )
