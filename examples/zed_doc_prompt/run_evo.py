#!/usr/bin/env python3
import os
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

BASE_DIR = os.path.dirname(__file__)
job_config = LocalJobConfig(eval_program_path=os.path.join(BASE_DIR, "evaluate.py"))

parent_config = dict(
    parent_selection_strategy="weighted",
    parent_selection_lambda=10.0,
)

db_config = DatabaseConfig(
    db_path="evolution_doc_prompt.sqlite",
    num_islands=2,
    archive_size=40,
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    migration_interval=10,
    migration_rate=0.1,
    island_elitism=True,
    **parent_config,
)

search_task_sys_msg = (
    "You evolve the prompts use in doc generation pipeline inside initial.py. Dont add new functions."
    "Mutations should help an LLM produce docs that help creating the patches matching real human fixes."
)


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full"],
    patch_type_probs=[0.7, 0.3],
    num_generations=20,
    max_parallel_jobs=24,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=["ollama:gemma3:12b"],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5],
        reasoning_efforts=["auto"],
        max_tokens=4096,
    ),
    meta_rec_interval=10,
    meta_llm_models=["ollama:gemma3:12b"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=1024),
    embedding_model=None,
    code_embed_sim_threshold=0.98,
    novelty_llm_models=None,
    llm_dynamic_selection=None,
    init_program_path=os.path.join(BASE_DIR, "initial.py"),
        results_dir="results_zed_doc_smoke_fresh",
)


def main():
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    main()
