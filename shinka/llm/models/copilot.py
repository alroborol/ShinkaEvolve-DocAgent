import os
import re
import subprocess
from typing import Dict, List, Optional

from .result import QueryResult


def _cleanup_output(output: str) -> str:
    content = re.split(r"\n\s*\nTotal usage est:", output or "", maxsplit=1)[0].strip()
    content = re.sub(r"^((?:[●•]\s+.*\n(?:\s+└.*\n)*))+,?", "", content).strip()
    return content


def query_copilot(
    client,
    model: str,
    msg: str,
    system_msg: str,
    msg_history: List[Dict],
    output_model: Optional[object],
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    if output_model is not None:
        raise ValueError("Structured outputs are not supported for copilot models.")

    cli_model = model.split("copilot:", 1)[1] if model.startswith("copilot:") else model
    cli_model = os.getenv("COPILOT_CLI_MODEL", cli_model or "gpt-5-mini")

    history_txt = ""
    if msg_history:
        lines = []
        for item in msg_history:
            role = item.get("role", "user")
            content = item.get("content", "")
            lines.append(f"{role}: {content}")
        history_txt = "\n\nConversation history:\n" + "\n".join(lines)

    prompt = f"System:\n{system_msg}\n\nUser:\n{msg}{history_txt}".strip()

    cmd = [
        "copilot",
        "--model",
        cli_model,
        "-s",
        "-p",
        prompt,
        "--allow-all-paths",
        "--yolo",
    ]

    timeout = int(os.getenv("COPILOT_CLI_TIMEOUT", "0") or "0")
    run_kwargs = dict(capture_output=True, text=True)
    run_kwargs["encoding"] = "utf-8"
    run_kwargs["errors"] = "replace"
    if timeout > 0:
        run_kwargs["timeout"] = timeout

    try:
        result = subprocess.run(cmd, **run_kwargs)
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    except subprocess.TimeoutExpired as e:
        output = f"<timeout after {e.timeout}s>\n"
    except Exception as e:
        output = f"<copilot invoke error: {e}>\n"

    content = _cleanup_output(output)
    new_msg_history = msg_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": content},
    ]

    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=0,
        output_tokens=0,
        cost=0.0,
        input_cost=0.0,
        output_cost=0.0,
        thought="",
        model_posteriors=model_posteriors,
    )
