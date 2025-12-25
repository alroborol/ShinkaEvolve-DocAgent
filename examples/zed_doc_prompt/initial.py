# EVOLVE-BLOCK-START
# System-level instruction: sent as the `system_msg` to the LLM client to
# establish role and global expectations for all queries in this module.
SYSTEM_MESSAGE = (
    "You convert source code into high-quality documentation that helps engineers "
    "understand behavior and make correct changes. Be specific and reference "
    "symbols and files explicitly."
)

# Detailed generation prompt (dace): used when producing an in-depth architecture
# and code explanation covering components, data/control flows, and developer
# guidance. Use for the raw-code generation path.
GENERATE_DOC_PROMPT = (
    "You are an expert technical writer and code analyst. Given source code, "
    "produce a detailed architecture and code explanation (dace): major components, "
    "control and data flows, key APIs, interactions between modules, and clear guidance "
    "for developers who need to make safe, non-breaking changes. Include references to "
    "symbols and files where relevant."
)

# Summarization prompt: used to create concise, developer-facing documentation
# geared toward code review, focusing on purpose, APIs, error handling and edge
# cases. Use when summarizing selected files.
SUMMARIZE_DOC_PROMPT = (
    "You are an expert technical writer. Given source code, produce concise, actionable "
    "documentation focused on purpose, key APIs, error handling, and edge cases. Keep it "
    "short and suitable for code review guidance."
)

# Template for asking the LLM to select files from a rendered file-tree. The
# `{tree}` placeholder is replaced with the textual tree when calling the LLM.
SELECTION_PROMPT_TEMPLATE = (
    "Given the project file tree below, pick which files are most relevant for "
    "generating concise developer-facing documentation about implementation, "
    "APIs, data flows and edge cases. Reply ONLY with a JSON array of the "
    "relative file paths you choose (no extra text).\n\n{tree}\n"
)

# Template for summarization calls: inject a summarization prompt plus the
# concatenated file contents into `{doc_prompt}` and `{files}` respectively.
SUMMARIZE_PROMPT_TEMPLATE = (
    "{doc_prompt}\n\n{files}"
)
# EVOLVE-BLOCK-END

def generate_docs(*args, **kwargs):
    from .internal.docs_generator import generate_docs as _gd
    return _gd(*args, **kwargs)


def summarize_selected_files(*args, **kwargs):
    from .internal.docs_generator import summarize_selected_files as _ss
    return _ss(*args, **kwargs)


def run_experiment(*args, **kwargs):
    from .internal.docs_generator import run_experiment as _re
    return _re(*args, **kwargs)


__all__ = [
    "SYSTEM_MESSAGE",
    "GENERATE_DOC_PROMPT",
    "SUMMARIZE_DOC_PROMPT",
    "SELECTION_PROMPT_TEMPLATE",
    "SUMMARIZE_PROMPT_TEMPLATE",
    "generate_docs",
    "summarize_selected_files",
    "run_experiment",
]
