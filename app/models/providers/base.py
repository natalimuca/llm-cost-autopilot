from abc import ABC, abstractmethod

from app.models.registry import ModelConfig
from app.models.response import Response


class BaseProvider(ABC):
    """Every provider adapter takes a prompt + ModelConfig and returns a Response."""

    @abstractmethod
    async def send(self, prompt: str, config: ModelConfig) -> Response:
        ...
