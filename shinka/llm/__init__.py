from typing import TYPE_CHECKING, Any

from .llm import LLMClient, extract_between
from .models import QueryResult
from .dynamic_sampling import (
    BanditBase,
    AsymmetricUCB,
    FixedSampler,
)

if TYPE_CHECKING:
    from .embedding import EmbeddingClient

__all__ = [
    "LLMClient",
    "extract_between",
    "QueryResult",
    "EmbeddingClient",
    "BanditBase",
    "AsymmetricUCB",
    "FixedSampler",
]


def __getattr__(name: str) -> Any:
    if name == "EmbeddingClient":
        from .embedding import EmbeddingClient

        return EmbeddingClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
