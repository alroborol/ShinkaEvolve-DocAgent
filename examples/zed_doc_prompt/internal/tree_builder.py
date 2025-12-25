from typing import Iterable
import os
import subprocess
from pathlib import Path
try:
    import requests
except Exception:
    # Local shim when `requests` cannot be imported in restricted environments.
    class _DummyResponse:
        def __init__(self, status_code=200, text="", data=None):
            self.status_code = status_code
            self.text = text
            self._data = data or {}
        def raise_for_status(self):
            if 400 <= self.status_code:
                raise Exception(f"HTTP {self.status_code}")
        def json(self):
            return self._data
    def _dummy_get(*args, **kwargs):
        return _DummyResponse()
    requests = type("requests_shim", (), {"get": staticmethod(_dummy_get)})()

def build_file_tree(paths: Iterable[str]) -> str:
    """Create a simple textual tree from a list of file paths.

    The tree groups files by common directories and is safe to show to the LLM.
    """
    norm = sorted(set(paths))
    tree = {}
    for p in norm:
        parts = p.replace("\\", "/").split("/")
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    def render(node, prefix=""):
        lines = []
        for i, (name, child) in enumerate(sorted(node.items())):
            connector = "└─ " if i == len(node) - 1 else "├─ "
            lines.append(f"{prefix}{connector}{name}")
            if child:
                ext = "   " if i == len(node) - 1 else "│  "
                lines.extend(render(child, prefix + ext))
        return lines

    return "\n".join(render(tree))


def fetch_repo_tree(repo: str) -> tuple[list[str], str]:
    """Fetch the file list and default branch for a GitHub repository.

    This was previously in `fetcher.py` and moved here because it primarily
    produces a repository file tree which is closely related to the tree
    rendering logic.
    """
    use_ssh = os.getenv("USE_SSH_REPO", "0") in ("1", "true", "True")
    ssh_key = os.getenv("GIT_SSH_KEY_PATH") or os.path.expanduser("~/.ssh/id_rsa")

    if use_ssh:
        cache_root = Path(__file__).parent / ".repo_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        repo_dir = cache_root / repo.replace("/", "_")
        try:
            if not repo_dir.exists():
                repo_dir_parent = repo_dir.parent
                repo_dir_parent.mkdir(parents=True, exist_ok=True)
                repo_url = f"git@github.com:{repo}.git"
                env = os.environ.copy()
                if ssh_key:
                    env["GIT_SSH_COMMAND"] = f"ssh -i \"{ssh_key}\" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
                subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True, env=env)
            try:
                out = subprocess.check_output(["git", "-C", str(repo_dir), "symbolic-ref", "refs/remotes/origin/HEAD"], env=os.environ)
                default_branch = out.decode("utf-8").strip().split("/")[-1]
            except Exception:
                default_branch = "main"

            paths = []
            for p in repo_dir.rglob("*"):
                if p.is_file():
                    try:
                        rel = p.relative_to(repo_dir).as_posix()
                        paths.append(rel)
                    except Exception:
                        continue
            return paths, default_branch
        except Exception:
            pass

    api = "https://api.github.com"
    repo_api = f"{api}/repos/{repo}"
    r = requests.get(repo_api, headers={"Accept": "application/vnd.github+json"}, timeout=30)
    r.raise_for_status()
    repo_meta = r.json()
    default_branch = repo_meta.get("default_branch", "main")

    tree_url = f"{api}/repos/{repo}/git/trees/{default_branch}?recursive=1"
    tr = requests.get(tree_url, headers={"Accept": "application/vnd.github+json"}, timeout=30)
    tr.raise_for_status()
    tree = tr.json()
    paths = [e["path"] for e in tree.get("tree", []) if e.get("type") == "blob"]
    return paths, default_branch
