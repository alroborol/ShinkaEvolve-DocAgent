"""Helpers for working with Ollama model names and settings."""
from typing import Optional


def is_ollama_model_name(model_name: Optional[str]) -> bool:
    """Return True when the provided model string refers to an Ollama model."""
    if not model_name or not isinstance(model_name, str):
        return False
    return model_name.lower().startswith("ollama")


def normalize_ollama_model_name(model_name: str) -> str:
    """Strip common Ollama prefixes and return the bare model name."""
    if not model_name:
        return model_name

    prefixes = ["ollama://", "ollama:", "ollama/", "ollama-"]
    for prefix in prefixes:
        if model_name.startswith(prefix):
            trimmed = model_name[len(prefix) :]
            return trimmed.lstrip("/: ")
    return model_name
