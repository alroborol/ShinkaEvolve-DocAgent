import re
from typing import Tuple, Set


def extract_funcs_and_vars_from_hunk(hunk: str) -> Tuple[Set[str], Set[str]]:
    """Return (functions_set, variables_set) found in a unified-diff hunk.

    Heuristics:
    - functions: lines containing `def NAME(` or `class NAME` in the hunk (after removing diff markers)
    - variables: simple assignment patterns `name =` at start of line (after removing diff markers)
    """
    funcs = set()
    vars = set()
    pattern_def = re.compile(r'^\s*def\s+([A-Za-z_]\w*)\s*\(')
    pattern_class = re.compile(r'^\s*class\s+([A-Za-z_]\w*)\s*[:\(]')
    pattern_assign = re.compile(r'^\s*([A-Za-z_]\w*)\s*=')

    for ln in hunk.splitlines():
        content = ln[1:] if ln.startswith(('+', '-', ' ')) else ln
        m = pattern_def.search(content)
        if m:
            funcs.add(m.group(1))
            continue
        m = pattern_class.search(content)
        if m:
            funcs.add(m.group(1))
            continue
        m = pattern_assign.search(content)
        if m:
            vars.add(m.group(1))

    return funcs, vars


def collect_funcs_vars_from_files_map(files_map: dict) -> Tuple[Set[str], Set[str]]:
    F = set()
    V = set()
    for hunks in files_map.values():
        for h in hunks:
            fset, vset = extract_funcs_and_vars_from_hunk(h)
            F.update(fset)
            V.update(vset)
    return F, V
