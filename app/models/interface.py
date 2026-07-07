from functools import lru_cache

from app.models.providers.base import BaseProvider
from app.models.registry import ModelConfig, Provider, get_model
from app.models.response import Response


def _build_openai() -> BaseProvider:
    from app.models.providers.openai_provider import OpenAIProvider
    return OpenAIProvider()


def _build_anthropic() -> BaseProvider:
    from app.models.providers.anthropic_provider import AnthropicProvider
    return AnthropicProvider()


def _build_ollama() -> BaseProvider:
    from app.models.providers.ollama_provider import OllamaProvider
    return OllamaProvider()


_PROVIDER_FACTORIES = {
    Provider.OPENAI: _build_openai,
    Provider.ANTHROPIC: _build_anthropic,
    Provider.OLLAMA: _build_ollama,
}


@lru_cache(maxsize=None)
def _get_provider(provider: Provider) -> BaseProvider:
    """Lazily construct each provider client on first use, so e.g. calling
    only the local Ollama model doesn't require OpenAI/Anthropic API keys."""
    return _PROVIDER_FACTORIES[provider]()


async def send_request(prompt: str, model_config: ModelConfig | str) -> Response:
    """Single entry point for calling any model in the registry.

    Accepts either a ModelConfig or a registry key (e.g. "gpt-4o-mini").
    """
    config = get_model(model_config) if isinstance(model_config, str) else model_config
    provider = _get_provider(config.provider)
    return await provider.send(prompt, config)
