import os
import json
import time
import hashlib
import logging
import base64
import subprocess
from pathlib import Path
from typing import List, Tuple

LOG = logging.getLogger(__name__)

# Single, minimal requests shim: try to import requests; otherwise provide a safe dummy
try:
    import requests
    HTTPError = requests.HTTPError
except Exception:
    requests = None
    class HTTPError(Exception):
        pass

# Cache for PR metadata and raw file contents
# Prefer a cache directory at the package root (examples/zed_doc_prompt/.pr_cache) to
# reuse existing cached data; fall back to the legacy internal/.pr_cache if present.
default_cache = Path(__file__).resolve().parent.parent / ".pr_cache"
legacy_cache = Path(__file__).resolve().parent / ".pr_cache"
env_cache = os.getenv("PR_CACHE_DIR")
if env_cache:
    CACHE_DIR = Path(env_cache)
elif default_cache.exists():
    CACHE_DIR = default_cache
elif legacy_cache.exists():
    CACHE_DIR = legacy_cache
else:
    CACHE_DIR = default_cache
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FORCE = os.getenv("FORCE_PR_FETCH") in ("1", "true", "True")
GH_REQUEST_DELAY = float(os.getenv("GH_REQUEST_DELAY", "0.5"))

GITHUB_REPO = os.getenv("GITHUB_REPO", "pallets/click")
GITHUB_API = os.getenv("GITHUB_API", "https://api.github.com")


def gh_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch_closed_pr_numbers(limit: int = 5) -> List[int]:
    cache_file = CACHE_DIR / f"closed_prs_filtered_{limit}.json"
    if cache_file.exists() and not CACHE_FORCE:
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            nums = data.get("pr_numbers") or []
            if isinstance(nums, list) and nums:
                return nums[:limit]
        except Exception:
            pass

    if requests is None:
        return []

    per_page = max(30, limit * 3)
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls?state=closed&per_page={per_page}"
    if GH_REQUEST_DELAY > 0:
        time.sleep(GH_REQUEST_DELAY)
    r = requests.get(url, headers=gh_headers(), timeout=30)
    r.raise_for_status()
    pr_list = r.json()

    pr_numbers: List[int] = []
    for pr in pr_list:
        if len(pr_numbers) >= limit:
            break
        pr_num = pr.get("number")
        if pr_num is None:
            continue

        merged = pr.get("merged_at")
        if merged is None:
            try:
                if GH_REQUEST_DELAY > 0:
                    time.sleep(GH_REQUEST_DELAY)
                details = requests.get(f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls/{pr_num}", headers=gh_headers(), timeout=30)
                details.raise_for_status()
                merged = details.json().get("merged_at")
            except Exception:
                continue
        if not merged:
            continue

        try:
            if GH_REQUEST_DELAY > 0:
                time.sleep(GH_REQUEST_DELAY)
            files_resp = requests.get(f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls/{pr_num}/files", headers=gh_headers(), timeout=30)
            files_resp.raise_for_status()
            files = files_resp.json()
            if any((f.get("patch") or "").strip() for f in files):
                pr_numbers.append(pr_num)
        except Exception:
            continue

    try:
        cache_file.write_text(json.dumps({"fetched_at": time.time(), "pr_numbers": pr_numbers}), encoding="utf-8")
    except Exception as e:
        LOG.warning(f"Failed to write PR list cache: {e}")

    return pr_numbers[:limit]


def fetch_pr_details(pr_number: int) -> Tuple[str, List[dict], str]:
    cache_file = CACHE_DIR / f"pr_{pr_number}.json"
    if cache_file.exists() and not CACHE_FORCE:
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return data.get("body", ""), data.get("files", []), data.get("diff_text", "")
        except Exception:
            pass

    if requests is None:
        raise RuntimeError("requests not available; cannot fetch PR details")

    pr_url = f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls/{pr_number}"
    try:
        if GH_REQUEST_DELAY > 0:
            time.sleep(GH_REQUEST_DELAY)
        pr_resp = requests.get(pr_url, headers=gh_headers(), timeout=30)
        pr_resp.raise_for_status()
        pr = pr_resp.json()
        body = pr.get("body", "") or ""

        files_url = f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls/{pr_number}/files"
        if GH_REQUEST_DELAY > 0:
            time.sleep(GH_REQUEST_DELAY)
        files_resp = requests.get(files_url, headers=gh_headers(), timeout=30)
        files_resp.raise_for_status()
        files = files_resp.json()

        diff_url = pr.get("diff_url")
        diff_text = ""
        if diff_url:
            if GH_REQUEST_DELAY > 0:
                time.sleep(GH_REQUEST_DELAY)
            diff_resp = requests.get(diff_url, headers=gh_headers(), timeout=30)
            diff_resp.raise_for_status()
            diff_text = diff_resp.text
    except requests.HTTPError as he:
        status = None
        try:
            status = he.response.status_code if he.response is not None else None
        except Exception:
            status = None
        if status == 403 and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                LOG.warning(f"GitHub 403 for PR {pr_number}; using cached PR data")
                return data.get("body", ""), data.get("files", []), data.get("diff_text", "")
            except Exception:
                pass
        raise

    try:
        cache_file.write_text(
            json.dumps({"fetched_at": time.time(), "body": body, "files": files, "diff_text": diff_text}),
            encoding="utf-8",
        )
    except Exception as e:
        LOG.warning(f"Failed to write PR details cache for {pr_number}: {e}")

    return body, files, diff_text


def fetch_file_contents(files: List[dict]) -> List[str]:
    contents = []
    for f in files:
        raw_url = f.get("raw_url")
        if raw_url:
            key = hashlib.sha256(raw_url.encode("utf-8")).hexdigest()
            cache_file = CACHE_DIR / f"raw_{key}.txt"
            if cache_file.exists() and not CACHE_FORCE:
                try:
                    contents.append(cache_file.read_text(encoding="utf-8"))
                    continue
                except Exception:
                    pass

            try:
                resp = requests.get(raw_url, headers=gh_headers(), timeout=30)
                if resp.status_code == 200:
                    text = resp.text
                    contents.append(text)
                    try:
                        cache_file.write_text(text, encoding="utf-8")
                    except Exception as e:
                        LOG.warning(f"Failed to cache raw file {raw_url}: {e}")
            except Exception as e:
                LOG.warning(f"Failed to fetch raw file {raw_url}: {e}")
                if cache_file.exists():
                    try:
                        contents.append(cache_file.read_text(encoding="utf-8"))
                    except Exception:
                        pass
    return contents


def fetch_repo_tree(repo: str) -> tuple[list[str], str]:
    use_ssh = os.getenv("USE_SSH_REPO", "0") in ("1", "true", "True")
    ssh_key = os.getenv("GIT_SSH_KEY_PATH") or os.path.expanduser("~/.ssh/id_rsa")

    cache_root = Path(__file__).parent / ".repo_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    repo_dir = cache_root / repo.replace("/", "_")

    def collect_from_clone(rdir: Path) -> tuple[list[str], str]:
        try:
            out = subprocess.check_output(["git", "-C", str(rdir), "symbolic-ref", "refs/remotes/origin/HEAD"], env=os.environ)
            default_branch_local = out.decode("utf-8").strip().split("/")[-1]
        except Exception:
            default_branch_local = "main"

        paths_local: list[str] = []
        for p in rdir.rglob("*"):
            if p.is_file():
                try:
                    paths_local.append(p.relative_to(rdir).as_posix())
                except Exception:
                    continue
        return paths_local, default_branch_local

    try:
        if not repo_dir.exists():
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            repo_url = f"git@github.com:{repo}.git" if use_ssh else f"https://github.com/{repo}.git"
            env = os.environ.copy()
            if use_ssh and ssh_key:
                env["GIT_SSH_COMMAND"] = f"ssh -i \"{ssh_key}\" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
            subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True, env=env)
        return collect_from_clone(repo_dir)
    except Exception:
        pass

    if requests is None:
        return [], "main"

    cache_file = CACHE_DIR / f"repo_tree_{repo.replace('/', '_')}.json"
    if cache_file.exists() and not CACHE_FORCE:
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            paths = data.get("paths", [])
            default_branch = data.get("default_branch", "main")
            if isinstance(paths, list):
                return paths, default_branch
        except Exception:
            pass

    repo_api = f"{GITHUB_API}/repos/{repo}"
    try:
        if GH_REQUEST_DELAY > 0:
            time.sleep(GH_REQUEST_DELAY)
        r = requests.get(repo_api, headers={"Accept": "application/vnd.github+json"}, timeout=30)
        r.raise_for_status()
        repo_meta = r.json()
        default_branch = repo_meta.get("default_branch", "main")

        tree_url = f"{GITHUB_API}/repos/{repo}/git/trees/{default_branch}?recursive=1"
        if GH_REQUEST_DELAY > 0:
            time.sleep(GH_REQUEST_DELAY)
        tr = requests.get(tree_url, headers={"Accept": "application/vnd.github+json"}, timeout=30)
        tr.raise_for_status()
        tree = tr.json()
        paths = [e["path"] for e in tree.get("tree", []) if e.get("type") == "blob"]
    except requests.HTTPError as he:
        status = None
        try:
            status = he.response.status_code if he.response is not None else None
        except Exception:
            status = None
        if status == 403 and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                LOG.warning(f"GitHub 403 for repo {repo}; using cached tree")
                paths = data.get("paths", [])
                default_branch = data.get("default_branch", "main")
                return paths, default_branch
            except Exception:
                pass
        raise

    try:
        cache_file.write_text(
            json.dumps({"fetched_at": time.time(), "paths": paths, "default_branch": default_branch}),
            encoding="utf-8",
        )
    except Exception as e:
        LOG.warning(f"Failed to write repo tree cache for {repo}: {e}")

    return paths, default_branch


def fetch_and_summarize_selected_files(client, repo: str, branch: str, selected_paths: List[str], prompt_template: str, doc_prompt: str, system_msg: str) -> str:
    api = GITHUB_API
    parts = []
    use_ssh = os.getenv("USE_SSH_REPO", "0") in ("1", "true", "True")
    cache_root = Path(__file__).parent / ".repo_cache"
    repo_dir = cache_root / repo.replace("/", "_")
    for p in selected_paths:
        try:
            if use_ssh and repo_dir.exists():
                fp = repo_dir / p
                if fp.exists():
                    try:
                        content = fp.read_text(encoding="utf-8")
                    except Exception:
                        content = fp.read_text(encoding="utf-8", errors="replace")
                    parts.append(f"--- FILE: {p} ---\n" + content[:8000])
                    continue

            blob_url = f"{api}/repos/{repo}/contents/{p}?ref={branch}"
            if requests is None:
                parts.append(f"--- FILE: {p} (requests not available) ---\n")
                continue
            r = requests.get(blob_url, headers={"Accept": "application/vnd.github+json"}, timeout=30)
            r.raise_for_status()
            data = r.json()
            content = ""
            if data.get("encoding") == "base64" and data.get("content"):
                try:
                    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                except Exception:
                    content = data.get("content", "")
            else:
                content = data.get("content", "")
            parts.append(f"--- FILE: {p} ---\n" + content[:8000])
        except Exception as e:
            parts.append(f"--- FILE: {p} (error: {e}) ---\n")

    merged = "\n\n".join(parts)
    prompt = prompt_template.format(doc_prompt=doc_prompt, files=merged)
    kwargs = client.get_kwargs()
    res = client.query(msg=prompt, system_msg=system_msg, llm_kwargs=kwargs)
    return res.content

