# Known breaks in unmodified migrated code

Each line names a copied file and the missing import, input, output, or old path it still expects. These are to-do items; no source code was changed during migration.

- `training/spec_f_run.py`: hardcoded `/workspace/spec_f_outputs` prepared inputs, caches, models, and cloze joins are absent.
- `training/spec_f_supervisor.py`: hardcoded `/workspace/spec_f_outputs` and old launch paths are absent.
- `training/test_spec_f.py`: imports the old Spec F runner contract and expects its prepared inputs.
- `training/spec06_common.py`: its old paired-world input IDs and `spec05_pair_validation` contract have no new data.
- `training/spec06_train.py`: `spec05_spine` moved to `generation/`, and old pair-validation input is absent.
- `training/spec06.template.json`: `spec05_pair_validation` and other old paired-world path placeholders have no new inputs.
- `eval/spec06_evaluate.py`: `spec06_common` moved to `training/`; old evaluation inputs are absent.
- `eval/wmdp_questions.csv`: no import; this approved question input is present, but no scoring run is configured.
- `eval/cooking_synthetic_textbook/score_eval.py`: synthetic-testbed model and question paths are not supplied.
- `eval/virality_SDF/score_eval.py`: virality-testbed model and question paths are not supplied.
- `attack/spec_e_common.py`: old Spec E question inventory and split contracts have no new prepared inputs.
- `attack/spec_e_run.py`: hardcoded `/workspace/CB_probes` import, old `configs/spec_e.preflight.json`, and `spec_e_outputs/preflight` splits are absent at those paths.
- `attack/spec_e.preflight.json`: model directories, `/workspace/CB_probes/prompt_utils.py`, Fisher driver, and old question inventory paths are stale.
- `attack/spec_r_run.py`: hardcoded `/workspace/CB_probes`, old `spec_e_outputs` grid/splits, and `spec_r_outputs` inputs are absent.
- `attack/spec_r_phase_a.py`: old `spec_e_outputs` references, `/workspace/CB_probes`, `/workspace/envs/wmdp-probes`, and old grid helper paths are absent.
- `attack/spec_r_stem_stats.py`: old `spec_r_outputs` and `spec_e_outputs/references_grid_path` are absent.
- `attack/spec_rz_run.py`: hardcoded `/workspace/RMU_probes`, `/workspace/fisher_information`, model path, and old Spec R-Z outputs are absent at those paths.
- `attack/spec_rz_stem_stats.py`: old `/workspace/fisher_information/spec_m_outputs/zephyr_references` and `spec_rz_outputs` are absent.
- `attack/spec_rz_collateral.py`: hardcoded `/workspace/RMU_probes`, `/workspace/fisher_information`, tokenizer directory, and old outputs are absent.
- `attack/spec_rz_launch.sh`: old `/workspace/wmdp_rewritten_cas`, environment, script, and output paths are stale.
- `attack/spec_rz_retry.sh`: old `/workspace/wmdp_rewritten_cas`, environment, script, and output paths are stale.
- `attack/support/CB_probes/prompt_utils.py`: its upstream WMDP model and dataset context is not vendored.
- `attack/support/RMU_probes/prompt_utils.py`: its upstream WMDP model and dataset context is not vendored.
- `attack/support/fisher_information/spec_m_zephyr_score.py`: old Zephyr model and Fisher reference paths are not migrated.
- `generation/generate_corpus.py`: no new canon world JSONL, scaffold, schema, context, or config exists yet.
- `generation/spec05_full.py`: relies on `generate_corpus.py` and old run inputs/configs; no new config exists yet.
- `generation/spec05_spine.py`: imports `spec06_common` from `training/`, which is outside its old local import path.
- `generation/scaled_spine_supervisor.py`: old run/config paths and output locations remain.
- `generation/run_spec05_spine_supervisor.sh`: old workspace script/config paths remain.

χ-regularized attack variants are absent because that code has not been written.
