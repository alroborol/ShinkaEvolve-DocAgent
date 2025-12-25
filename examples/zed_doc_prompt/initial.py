from typing import List
from shinka.llm import LLMClient

# This string is the target of evolution. Mutations will change its content.

# EVOLVE-BLOCK-START
DOC_PROMPT = (
    "You are an expert technical writer. Given Python, Rust, or TypeScript code, "
    "produce precise, actionable documentation: purpose, key APIs, data flows, "
    "error handling, and edge cases. Focus on how to safely modify behavior. "
    "Return a concise doc suitable for code review guidance."
)

SYSTEM_MESSAGE = (
    "You convert source code into high-quality documentation that helps engineers "
    "understand behavior and make correct changes. Be specific and reference "
    "symbols and files explicitly."
)


def generate_docs(code_texts: List[str]) -> str:
    """Generate a documentation blob from a list of code texts using DOC_PROMPT."""
    client = LLMClient(
        model_names=["ollama:gemma3:12b"],
        temperatures=0.0,
        max_tokens=4096,
        reasoning_efforts="auto",
        verbose=False,
    )
    # Merge code texts into one input; keep under reasonable size.
    merged = "\n\n".join(
        text[:8000] for text in code_texts if isinstance(text, str)
    )
    msg = f"{DOC_PROMPT}\n\n<code>\n{merged}\n</code>"
    kwargs = client.get_kwargs()
    res = client.query(msg=msg, system_msg=SYSTEM_MESSAGE, llm_kwargs=kwargs)
    return res.content

# EVOLVE-BLOCK-END

def run_experiment(random_ints: List[int]) -> List[str]:
    """Dummy entry point so the evaluator can call into the program.
    This returns simple docs for synthetic code snippets based on random seeds.
    The evaluator will prefer calling generate_docs(code_texts).
    """
    samples = []
    for r in random_ints:
        code = f"// synthetic sample {r}\nfunction foo() {{ return {r} }}"
        samples.append(generate_docs([code]))
    return samples
