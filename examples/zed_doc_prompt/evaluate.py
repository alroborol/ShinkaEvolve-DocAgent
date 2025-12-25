import os
import json
import time
import random
import argparse
import importlib.util
from pathlib import Path
import sys

try:
    import requests
    HTTPError = requests.HTTPError
except Exception:
    requests = None
    class HTTPError(Exception):
        pass
# Ensure repo root is on sys.path, then import fetcher absolutely (works for script or module)
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from ShinkaEvolve.examples.zed_doc_prompt.internal.fetcher import (
    fetch_closed_pr_numbers,
    fetch_pr_details,
    fetch_file_contents,
)
from ShinkaEvolve.examples.zed_doc_prompt.internal.patch_parser import parse_file_hunks
from ShinkaEvolve.examples.zed_doc_prompt.internal.name_extractor import (
    collect_funcs_vars_from_files_map,
)

def make_llm_client_class():
    """Return an LLMClient class (real one if available, otherwise a lightweight stub).

    This factory avoids importing heavy dependencies at module import time.
    """
    try:
        from shinka.llm import LLMClient  # type: ignore
        return LLMClient
    except Exception:
        # Attempt to add the ShinkaEvolve package directory to sys.path (two levels up)
        try:
            this_dir = Path(__file__).resolve()
            shinka_root = this_dir.parents[2]
            if str(shinka_root) not in sys.path:
                sys.path.insert(0, str(shinka_root))
            from shinka.llm import LLMClient  # type: ignore
            return LLMClient
        except Exception:
            class _DummyLLMClient:
                def __init__(self, *args, **kwargs):
                    pass
                def get_kwargs(self):
                    return {}
                def query(self, *args, **kwargs):
                    return type("R", (), {"content": ""})()
            return _DummyLLMClient

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

    # New scoring strategy (three components):
    # 1) File identity score: fraction of files modified by the generated patch that
    #    exactly match the files modified in the actual patch (by path and full hunk text).
    # 2) Function-level score: overlap of function names modified in generated vs actual.
    # 3) Variable-level score: overlap of variable/assignment names modified in generated vs actual.
    # Each component is normalized to [0,1] and combined with weights.

    g = (generated or "").strip()
    a = (actual or "").strip()
    if not g and not a:
        return 0.0

    # Parse unified diff into mapping: filepath -> list of hunk texts
    g_files = parse_file_hunks(g)
    a_files = parse_file_hunks(a)

    # File identity score: count exact hunk equality per file path
    if not g_files and not a_files:
        file_score = 1.0 if g.strip() == a.strip() and g else 0.0
    else:
        # consider union of file paths touched
        all_paths = set(g_files.keys()).union(set(a_files.keys()))
        if not all_paths:
            file_score = 0.0
        else:
            match_count = 0
            for p in all_paths:
                g_hunks = g_files.get(p, [])
                a_hunks = a_files.get(p, [])
                # exact match if both lists equal (order-sensitive)
                if g_hunks and a_hunks and g_hunks == a_hunks:
                    match_count += 1
            file_score = match_count / len(all_paths)

    # Function and variable score: union-overlap across all hunks
    g_funcs, g_vars = collect_funcs_vars_from_files_map(g_files)
    a_funcs, a_vars = collect_funcs_vars_from_files_map(a_files)

    def jaccard(s1: set, s2: set) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        inter = len(s1.intersection(s2))
        uni = len(s1.union(s2))
        return inter / uni if uni else 0.0

    func_score = jaccard(g_funcs, a_funcs)
    var_score = jaccard(g_vars, a_vars)

    # Combine with weights. Prioritize file identity, then function, then variable.
    combined = 0.5 * file_score + 0.35 * func_score + 0.15 * var_score

    return max(0.0, min(1.0, float(combined)))


def generate_patch_from_doc_and_pr(doc_text: str, pr_body: str) -> str:
    """Like judge_fix_with_llm but logs the system/prompt and LLM response to `save_dir` if provided."""
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
    LLMClientClass = make_llm_client_class()
    llm = LLMClientClass(
        model_names=["ollama:gemma3:12b"],
        temperatures=0.0,
        max_tokens=2048,
        reasoning_efforts="auto",
        verbose=False,
    )

    kwargs = llm.get_kwargs()
    res = llm.query(msg=msg, system_msg=sys, llm_kwargs=kwargs)
    content = (res.content or "").strip()
    return content



def evaluate_prompt_program(program_path: str, results_dir: str, num_prs: int = 5, seed: int = 42, pr_ids: list[int] | None = None):
    # Resolve program_path: try given path, then try relative to this script's directory
    p = Path(program_path)
    if not p.is_absolute() and not p.exists():
        alt = Path(__file__).parent / program_path
        if alt.exists():
            p = alt
    if not p.exists():
        raise RuntimeError(f"Cannot find program at {program_path} (checked {p})")

    # If the program is the package-local `initial.py`, import it as a package
    # submodule so relative imports (from .docs_generator) work.
    try:
        pkg_root = Path(__file__).resolve().parents[2]
        pkg_name = pkg_root.name
        subpkg = Path(__file__).resolve().parent.name
        if p.resolve() == (Path(__file__).resolve().parent / "initial.py"):
            # Ensure workspace root (parent of pkg_root) is on sys.path so the package
            # can be imported (e.g., ShinkaEvolve).
            repo_root = pkg_root.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            module_name = f"{pkg_name}.examples.{subpkg}.initial"
            module = importlib.import_module(module_name)
        else:
            # Prefer importing via package path (so relative imports work), fallback to file import.
            repo_root = Path(__file__).resolve().parents[3]
            module = None
            try:
                rel_parts = p.resolve().relative_to(repo_root).with_suffix("").parts
                module_name = ".".join(rel_parts)
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                module = importlib.import_module(module_name)
            except Exception:
                module = None

            if module is None:
                spec = importlib.util.spec_from_file_location("program", str(p))
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"Cannot load program at {p}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
    except Exception:
        # Fall back to loading by file path
        spec = importlib.util.spec_from_file_location("program", str(p))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load program at {p}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    # Patch generated programs that reference a non-existent sibling `internal` package
    # by redirecting their generate_docs to the shared internal.docs_generator.
    prog_path = Path(getattr(module, "__file__", ""))
    if "results_zed_doc" in prog_path.as_posix() and not (prog_path.parent / "internal").exists():
        try:
            from ShinkaEvolve.examples.zed_doc_prompt.internal.docs_generator import (
                generate_docs as _shared_generate_docs,
            )
            module.generate_docs = _shared_generate_docs
        except Exception:
            pass

    random.seed(seed)
    start_t = time.time()
    metrics = {"public": {}, "private": {}, "combined_score": 0.0, "runtime": 0.0}

    similarity_scores = []
    errors = []
    if pr_ids:
        # Use explicit PR IDs provided by the user
        pr_numbers = pr_ids[:num_prs]
    else:
        try:
            pr_numbers = fetch_closed_pr_numbers(limit=num_prs)
        except Exception as e:
            errors.append(f"failed_list_prs: {e}")
            pr_numbers = []

    # Generate documentation once per run; `generate_docs()` is stable across PRs
    try:
        # some program modules may accept no args, others may accept a config; try no-arg first
        doc_text = module.generate_docs()
    except TypeError:
        try:
            doc_text = module.generate_docs(None)
        except Exception:
            doc_text = ""

    for prn in pr_numbers:
        try:
            try:
                pr_body, files, actual_diff = fetch_pr_details(prn)
                time.sleep(1)  # brief pause between PR fetches
            except HTTPError as he:
                status = None
                try:
                    status = he.response.status_code if he.response is not None else None
                except Exception:
                    status = None
                if status == 403:
                    errors.append(f"pr_{prn}: 403 rate limit exceeded (use GITHUB_TOKEN or rely on cache)")
                    # Skip this PR
                    continue
                else:
                    raise
            # reuse `doc_text` generated once above
            gen_diff = generate_patch_from_doc_and_pr(doc_text, pr_body)

            # Print generated and actual patches for inspection (flush to avoid buffering)
            print(f"--- PR {prn} GENERATED PATCH ---", flush=True)
            print(gen_diff or "<empty>", flush=True)
            print(f"--- PR {prn} ACTUAL PATCH ---", flush=True)
            print(actual_diff or "<empty>", flush=True)

            # Save detailed artifacts for inspection: doc, generated patch, actual patch, file metadata and raw contents
            pr_dir = Path(results_dir) / f"pr_{prn}"
            pr_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Save generated documentation used as prompt
                with open(pr_dir / "doc.txt", "w", encoding="utf-8") as f:
                    f.write(doc_text or "")

                # Save generated patch
                with open(pr_dir / "gen_patch.diff", "w", encoding="utf-8") as f:
                    f.write(gen_diff or "")

                # Save actual patch
                with open(pr_dir / "actual_patch.diff", "w", encoding="utf-8") as f:
                    f.write(actual_diff or "")

            except Exception as e:
                print(f"Failed to write PR artifacts for {prn}: {e}")

            similarity = score_patch_similarity(gen_diff, actual_diff)
            similarity_scores.append(similarity)

            # Save per-PR summary
            try:
                with open(pr_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "pr_number": prn,
                        "similarity": similarity,
                        "doc_path": str(pr_dir / "doc.txt"),
                        "generated_patch_path": str(pr_dir / "gen_patch.diff"),
                        "actual_patch_path": str(pr_dir / "actual_patch.diff"),
                    }, f, indent=2)
            except Exception:
                pass
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

    # Ensure metrics are printed immediately
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate evolved doc prompt against closed Zed PRs")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results_zed_doc")
    parser.add_argument("--num_prs", type=int, default=3)
    parser.add_argument("--pr_ids", type=str, default="2972,2933,2855", help="Comma-separated PR numbers to evaluate (e.g. 123,456)")
    args = parser.parse_args()
    pr_ids = None
    if args.pr_ids:
        try:
            pr_ids = [int(x.strip()) for x in args.pr_ids.split(",") if x.strip()]
        except Exception:
            print("Failed to parse --pr_ids; ensure comma-separated integers")
            pr_ids = None

    evaluate_prompt_program(args.program_path, args.results_dir, num_prs=args.num_prs, pr_ids=pr_ids)
