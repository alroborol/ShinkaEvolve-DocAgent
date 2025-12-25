from typing import List
import os
import json
from pathlib import Path

from .llm_helper import make_client, ask_llm_to_select_files
from .tree_builder import build_file_tree, fetch_repo_tree
from .fetcher import fetch_and_summarize_selected_files
from ..initial import (
    SYSTEM_MESSAGE,
    SUMMARIZE_DOC_PROMPT,
    SUMMARIZE_PROMPT_TEMPLATE,
)


def generate_docs() -> str:
    client = make_client()
    repo = os.getenv("GITHUB_REPO", "pallets/click")
    try:
        paths, branch = fetch_repo_tree(repo)
    except Exception:
        return "Could not fetch repository tree for documentation generation."

    tree = build_file_tree(paths)
    selected = ask_llm_to_select_files(client, tree, paths, SYSTEM_MESSAGE)

    summary = fetch_and_summarize_selected_files(
        client,
        repo,
        branch,
        selected,
        SUMMARIZE_PROMPT_TEMPLATE,
        SUMMARIZE_DOC_PROMPT,
        SYSTEM_MESSAGE,
    )
    return summary


def summarize_selected_files(client, selected_paths: List[str]) -> str:
    parts = []
    for p in selected_paths:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                parts.append(f"--- FILE: {p} ---\n" + fh.read()[:8000])
        except Exception as e:
            parts.append(f"--- FILE: {p} (error: {e}) ---\n")

    merged = "\n\n".join(parts)
    prompt = SUMMARIZE_PROMPT_TEMPLATE.format(doc_prompt=SUMMARIZE_DOC_PROMPT, files=merged)
    kwargs = client.get_kwargs()
    res = client.query(msg=prompt, system_msg=SYSTEM_MESSAGE, llm_kwargs=kwargs)
    return res.content


def run_experiment(random_ints: List[int]) -> List[str]:
    samples = []
    for r in random_ints:
        samples.append(generate_docs())
    return samples


__all__ = ["generate_docs", "summarize_selected_files", "run_experiment"]
