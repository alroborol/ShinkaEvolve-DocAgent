import os
import json
import time
import random
import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional

import requests
import Levenshtein

from shinka.llm import LLMClient
import re

GITHUB_REPO = "zed-industries/zed"
GITHUB_API = "https://api.github.com"


def gh_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch_closed_pr_numbers(limit: int = 5) -> List[int]:
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls?state=closed&per_page={limit}"
    r = requests.get(url, headers=gh_headers(), timeout=30)
    r.raise_for_status()
    return [pr["number"] for pr in r.json()]


def fetch_pr_details(pr_number: int) -> Tuple[str, List[dict], str]:
    # PR body
    pr_url = f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls/{pr_number}"
    pr_resp = requests.get(pr_url, headers=gh_headers(), timeout=30)
    pr_resp.raise_for_status()
    pr = pr_resp.json()
    body = pr.get("body", "") or ""

    # changed files
    files_url = f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls/{pr_number}/files"
    files_resp = requests.get(files_url, headers=gh_headers(), timeout=30)
    files_resp.raise_for_status()
    files = files_resp.json()

    # unified diff
    diff_url = pr.get("diff_url")
    diff_text = ""
    if diff_url:
        diff_resp = requests.get(diff_url, headers=gh_headers(), timeout=30)
        diff_resp.raise_for_status()
        diff_text = diff_resp.text

    return body, files, diff_text


def fetch_file_contents(files: List[dict]) -> List[str]:
    contents = []
    for f in files:
        raw_url = f.get("raw_url")
        if raw_url:
            try:
                resp = requests.get(raw_url, headers=gh_headers(), timeout=30)
                if resp.status_code == 200:
                    contents.append(resp.text)
            except Exception:
                pass
    return contents


def score_patch_similarity(generated: str, actual: str) -> float:
    """
    Score similarity based on functional changes rather than raw diff text.

    Strategy:
    - Extract added/removed code lines from unified diffs (ignore diff metadata).
    - Pull out identifiers (function/class names, assigned names, and likely called names).
    - Compute an identifier Jaccard score (captures whether the same functions/vars are touched).
    - Compute a token/text similarity on the added code using Levenshtein.ratio.
    - Combine both scores into a final 0.0-1.0 score.
    """

    g = (generated or "").strip()
    a = (actual or "").strip()
    if not g or not a:
        return 0.0

    def extract_added_removed(diff: str) -> Tuple[List[str], List[str]]:
        added = []
        removed = []
        for line in diff.splitlines():
            if line.startswith("+++ ") or line.startswith("--- "):
                # file markers, skip
                continue
            if line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:].rstrip())
            elif line.startswith("-") and not line.startswith("---"):
                removed.append(line[1:].rstrip())
        return added, removed

    def extract_identifiers(lines: List[str]) -> set:
        ids = set()
        pattern_def = re.compile(r'^\s*def\s+([A-Za-z_]\w*)\s*\(')
        pattern_class = re.compile(r'^\s*class\s+([A-Za-z_]\w*)\s*[:\(]')
        pattern_assign = re.compile(r'^\s*([A-Za-z_]\w*)\s*=')
        pattern_call = re.compile(r'([A-Za-z_]\w*)\s*\(')
        for ln in lines:
            # try def/class first
            m = pattern_def.search(ln)
            if m:
                ids.add(m.group(1))
                continue
            m = pattern_class.search(ln)
            if m:
                ids.add(m.group(1))
                continue
            m = pattern_assign.search(ln)
            if m:
                ids.add(m.group(1))
            # collect probable calls (limit to a few to avoid noise)
            for call in pattern_call.findall(ln)[:3]:
                ids.add(call)
        return ids

    g_added, g_removed = extract_added_removed(g)
    a_added, a_removed = extract_added_removed(a)

    # If no explicit hunks (maybe LLM output was plain patch-less code), treat whole text as "added"
    if not g_added and not g_removed:
        g_added = [g]
    if not a_added and not a_removed:
        a_added = [a]

    g_ids = extract_identifiers(g_added + g_removed)
    a_ids = extract_identifiers(a_added + a_removed)

    # Identifier (structural/functional) similarity: Jaccard
    union = g_ids.union(a_ids)
    id_score = 0.0
    if union:
        id_score = len(g_ids.intersection(a_ids)) / len(union)

    # Content similarity on the "added" code (what functionality was introduced)
    g_text = "\n".join(g_added).strip()
    a_text = "\n".join(a_added).strip()
    text_score = 0.0
    if g_text and a_text:
        try:
            text_score = float(Levenshtein.ratio(g_text, a_text))
        except Exception:
            text_score = 0.0

    # Combine: favor identifier overlap (structural/functional) but include token similarity
    combined = 0.65 * id_score + 0.35 * text_score

    # If both metrics are zero but patches are non-empty, fall back to whole-diff ratio
    if combined == 0.0:
        try:
            combined = float(Levenshtein.ratio(g, a))
        except Exception:
            combined = 0.0

    # clamp and return
    return max(0.0, min(1.0, combined))


def judge_fix_with_llm(doc_text: str, pr_body: str) -> str:
    sys = (
        "You generate unified diff patches based on documentation and issue/PR context. "
        "Output only a unified diff with file paths and hunks (---, +++, @@)."
    )
    msg = (
        "Given the documentation below and the PR description, propose a minimal patch "
        "that addresses the issue in the PR.\n\n"
        f"<documentation>\n{doc_text}\n</documentation>\n\n"
        f"<pr_description>\n{pr_body}\n</pr_description>\n"
    )
    llm = LLMClient(
        model_names=["ollama:gemma3:12b"],
        temperatures=0.0,
        max_tokens=2048,
        reasoning_efforts="auto",
        verbose=False,
    )
    try:
        kwargs = llm.get_kwargs()
        res = llm.query(msg=msg, system_msg=sys, llm_kwargs=kwargs)
        return (res.content or "").strip()
    except Exception as e:
        # If the local model is unavailable or errors, return empty diff
        return ""


def evaluate_prompt_program(program_path: str, results_dir: str, num_prs: int = 5, seed: int = 42):
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load program at {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    random.seed(seed)
    start_t = time.time()
    metrics = {"public": {}, "private": {}, "combined_score": 0.0, "runtime": 0.0}

    similarity_scores = []
    errors = []
    try:
        pr_numbers = fetch_closed_pr_numbers(limit=num_prs)
    except Exception as e:
        errors.append(f"failed_list_prs: {e}")
        pr_numbers = []

    for prn in pr_numbers:
        try:
            pr_body, files, actual_diff = fetch_pr_details(prn)
            code_texts = fetch_file_contents(files)
            doc_text = module.generate_docs(code_texts)
            gen_diff = judge_fix_with_llm(doc_text, pr_body)
            similarity = score_patch_similarity(gen_diff, actual_diff)
            similarity_scores.append(similarity)
        except Exception as e:
            errors.append(f"pr_{prn}: {e}")

    combined = sum(similarity_scores) / max(len(similarity_scores), 1)
    metrics["public"]["per_pr_similarity"] = similarity_scores
    metrics["combined_score"] = combined
    metrics["runtime"] = time.time() - start_t
    metrics["private"]["errors"] = errors

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    correct_file = os.path.join(results_dir, "correct.json")
    with open(correct_file, "w") as f:
        json.dump({"correct": True, "error": "" if not errors else "; ".join(errors)}, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate evolved doc prompt against closed Zed PRs")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results_zed_doc")
    parser.add_argument("--num_prs", type=int, default=3)
    args = parser.parse_args()
    evaluate_prompt_program(args.program_path, args.results_dir, num_prs=args.num_prs)
