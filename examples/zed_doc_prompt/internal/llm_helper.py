from typing import Any, List
import json


def make_client() -> Any:
    """Return an LLM client. Uses `shinka.llm.LLMClient` when available,
    otherwise returns a lightweight dummy with the same surface used here.
    """
    try:
        from shinka.llm import LLMClient  # type: ignore
    except Exception:
        class _Dummy:
            def __init__(self, *args, **kwargs):
                pass
            def get_kwargs(self):
                return {}
            def query(self, *args, **kwargs):
                return type("R", (), {"content": ""})()
        LLMClient = _Dummy

    return LLMClient(
        model_names=["ollama:gemma3:12b"],
        temperatures=0.0,
        max_tokens=4096,
        reasoning_efforts="auto",
        verbose=False,
    )


def ask_llm_to_select_files(client: Any, tree_text: str, all_paths: List[str], system_msg: str) -> List[str]:
    """Ask the LLM to pick file paths from a rendered `tree_text`.

    The LLM should reply with a JSON array of relative paths. This function
    is resilient to extra text by attempting to extract a JSON array if direct
    parsing fails.
    """
    prompt = (
        "Given the project file tree below, pick which files are most relevant for "
        "generating concise developer-facing documentation about implementation, "
        "APIs, data flows and edge cases. Reply ONLY with a JSON array of the "
        "relative file paths you choose (no extra text).\n\n" + tree_text
    )
    kwargs = client.get_kwargs()
    res = client.query(msg=prompt, system_msg=system_msg, llm_kwargs=kwargs)
    content = (getattr(res, "content", "") or "").strip()

    # Try parsing raw content, then fall back to extracting the first JSON array.
    try:
        picks = json.loads(content)
    except Exception:
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return all_paths
        try:
            picks = json.loads(content[start : end + 1])
        except Exception:
            return all_paths

    if isinstance(picks, list):
        return [p for p in picks if p in all_paths]
    return all_paths
