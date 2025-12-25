import logging
import os
from typing import Dict, List, Optional
import requests
from .result import QueryResult

logger = logging.getLogger(__name__)


def _build_messages(system_msg: str, msg_history: List[Dict], msg: str) -> List[Dict]:
    messages: List[Dict] = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.extend(msg_history)
    messages.append({"role": "user", "content": msg})
    return messages


def _prepare_options(kwargs: Dict) -> Dict:
    # Force the Ollama server to use 24 generation threads.
    options = {"num_thread": 24}
    temperature = kwargs.get("temperature")
    if temperature is not None:
        options["temperature"] = temperature

    max_tokens = kwargs.get("max_tokens") or kwargs.get("max_output_tokens")
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    return options


def query_ollama(
    client: requests.Session,
    model: str,
    msg: str,
    system_msg: str,
    msg_history: List[Dict],
    output_model: Optional[object],
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query an Ollama server via the /api/chat endpoint.

    The client should be a requests.Session with a base_url attribute pointing to the
    Ollama server (default: http://localhost:11434). The server will be asked to use
    24 threads (num_thread=24) for generation while the caller limits concurrent
    requests on the client side.
    """

    base_url = getattr(client, "base_url", None) or os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    messages = _build_messages(system_msg, msg_history, msg)
    options = _prepare_options(dict(kwargs))

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    response = client.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=kwargs.get("timeout", 6000),
    )
    response.raise_for_status()
    data = response.json()

    content = data.get("message", {}).get("content", "")
    new_msg_history = msg_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": content},
    ]

    input_tokens = data.get("prompt_eval_count", 0) or 0
    output_tokens = data.get("eval_count", 0) or 0

    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=0.0,
        input_cost=0.0,
        output_cost=0.0,
        thought="",
        model_posteriors=model_posteriors,
    )
