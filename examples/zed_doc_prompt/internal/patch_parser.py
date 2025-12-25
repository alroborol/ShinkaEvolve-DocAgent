import re
from typing import Dict, List


def parse_file_hunks(diff: str) -> Dict[str, List[str]]:
    """Parse a unified diff string into a mapping from file path to list of hunk texts.

    Heuristic parser that recognizes file headers like `+++ b/path` and hunk blocks starting with `@@`.
    """
    files = {}
    if not diff:
        return files
    lines = diff.splitlines()
    current_file = None
    current_hunks = []
    hunk_buf = []
    for ln in lines:
        m = re.match(r'^\+\+\+\s+b/(.+)$', ln)
        if m:
            # store previous file
            if current_file is not None:
                if hunk_buf:
                    current_hunks.append('\n'.join(hunk_buf))
                    hunk_buf = []
                files[current_file] = current_hunks
            current_file = m.group(1).strip()
            current_hunks = []
            hunk_buf = []
            continue
        # hunk header
        if ln.startswith('@@'):
            if hunk_buf:
                current_hunks.append('\n'.join(hunk_buf))
            hunk_buf = [ln]
            continue
        # collect hunk lines
        if current_file is not None and (ln.startswith('+') or ln.startswith('-') or ln.startswith(' ')):
            hunk_buf.append(ln)
    # finalize last
    if current_file is not None:
        if hunk_buf:
            current_hunks.append('\n'.join(hunk_buf))
        files[current_file] = current_hunks
    return files
