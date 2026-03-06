from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any


def is_copilot_cli_available() -> bool:
    return shutil.which("copilot") is not None


@dataclass
class _Result:
    content: str
    raw: str = ""


class CopilotCLIClient:
    def __init__(
        self,
        *args,
        model_names: list[str] | None = None,
        temperatures: Any = None,
        max_tokens: int | None = None,
        reasoning_efforts: str | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        self.model = os.getenv("COPILOT_CLI_MODEL", "gpt-5-mini")
        self.timeout = int(os.getenv("COPILOT_CLI_TIMEOUT", "0") or "0")
        self.verbose = verbose

    def get_kwargs(self) -> dict:
        return {}

    def query(self, msg: str, system_msg: str | None = None, llm_kwargs: dict | None = None):
        prompt_parts = []
        if system_msg:
            prompt_parts.append(f"System:\n{system_msg}")
        prompt_parts.append(msg or "")
        prompt = "\n\n".join(prompt_parts).strip()

        run_kwargs = dict(
            args=[
                "copilot",
                "--model",
                self.model,
                "-s",
                "-p",
                prompt,
                "--allow-all-paths",
                "--yolo",
            ],
            capture_output=True,
            text=True,
        )
        if self.timeout > 0:
            run_kwargs["timeout"] = self.timeout

        try:
            result = subprocess.run(**run_kwargs)
        except subprocess.TimeoutExpired as e:
            return _Result(content="", raw=f"<timeout after {e.timeout}s>\n")
        except Exception as e:
            return _Result(content="", raw=f"<copilot invoke error: {e}>\n")

        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        if result.returncode != 0 and not output.strip():
            output = f"<copilot exit code {result.returncode}>\n"

        content = re.split(r"\n\s*\nTotal usage est:", output, maxsplit=1)[0].strip()
        content = re.sub(r"^((?:[●•]\s+.*\n(?:\s+└.*\n)*))+,?", "", content).strip()

        return _Result(content=content, raw=output)
