
## Doc Agent Evolution Overview

This modification evolves a "doc agent" — a prompt-driven component that turns code into actionable documentation to guide patch generation.

- **Core components:**
  - `examples/zed_doc_prompt/initial.py`: defines `DOC_PROMPT`, `SYSTEM_MESSAGE`, and `generate_docs()` — the doc agent’s behavior. Evolution mutates `DOC_PROMPT` to improve outcomes.
  - `examples/zed_doc_prompt/evaluate.py`: fetches closed PRs from `zed-industries/zed`, loads changed files, calls the doc agent to produce documentation, asks the model to output a unified diff, and scores similarity against the real patch.
  - `examples/zed_doc_prompt/run_evo.py`: configures and runs the evolutionary loop using `ollama:gemma3:12b` with single-process client concurrency and server `num_thread: 24`.

- **Scoring metric:**
  - Combines identifier-level Jaccard (structural overlap of functions/classes/assignments/calls in added/removed lines) with Levenshtein ratio over added content. This favors functional alignment while still accounting for text similarity.

- **Concurrency & model:**
  - Client-side calls: single process when any `ollama:*` model is used.
  - Server-side generation threads: `options.num_thread: 24` via Ollama `/api/chat`.
  - Default model: `ollama:gemma3:12b`; set `OLLAMA_BASE_URL` to point to your server.

- **Adapting the doc agent:**
  - Change target repo in `evaluate.py` (`GITHUB_REPO`) to evolve documentation for other projects.
  - Tune `DOC_PROMPT` structure (sections, emphasis) and `SYSTEM_MESSAGE` to focus on behaviors relevant to your domain.
  - Adjust `num_generations`, `max_tokens`, and PR sampling (`--num_prs`) for deeper or faster runs.
  - Optional: set `GITHUB_TOKEN` to avoid rate limits when fetching PR data.

- **Outputs to expect:**
  - Per-generation metrics and correctness under `gen_<n>/results/`.
  - A `best` snapshot with the top-scoring evolved program, including its mutated `DOC_PROMPT`.

## Quick Start

Prerequisites:
- Python 3.11+ environment with this workspace.
- Ollama installed and running; model `gemma3:12b` pulled.
- Optional: `GITHUB_TOKEN` to raise GitHub API rate limits.

Setup and run:

```powershell
# 1) Ensure Ollama server and model
ollama serve
ollama pull gemma3:12b

# 2) (Optional) Point to a custom Ollama URL
# $env:OLLAMA_BASE_URL = "http://localhost:11434"

# 3) Kick off evolution for the Zed doc agent
python ShinkaEvolve\examples\zed_doc_prompt\run_evo.py

# 4) Inspect results
# - Evolution log: results_zed_doc_smoke_fresh\evolution_run.log
# - Best program: results_zed_doc_smoke_fresh\best\main.py (contains evolved DOC_PROMPT)
# - Metrics per generation: results_zed_doc_smoke_fresh\gen_<n>\results\metrics.json

# 5) (Optional) Run a single evaluator check
python ShinkaEvolve\examples\zed_doc_prompt\evaluate.py `
  --program_path ShinkaEvolve\examples\zed_doc_prompt\initial.py `
  --results_dir ShinkaEvolve\tmp_results `
  --num_prs 1
```

Notes:
- Client-side LLM queries run single-process when using `ollama:*` models; the Ollama server is instructed to use 24 generation threads for performance.
- You can change the target repository in `examples/zed_doc_prompt/evaluate.py` by editing `GITHUB_REPO`.


## Related Open-Source Projects 🧑‍🔧

- [OpenEvolve](https://github.com/codelion/openevolve): An open-source implementation of AlphaEvolve
- [LLM4AD](https://github.com/Optima-CityU/llm4ad): A Platform for Algorithm Design with Large Language Model

## Citation ✍️

If you use `ShinkaEvolve` in your research, please cite it as follows:

```
@article{lange2025shinka,
  title={ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution},
  author={Lange, Robert Tjarko and Imajuku, Yuki and Cetin, Edoardo},
  journal={arXiv preprint arXiv:2509.19349},
  year={2025}
}