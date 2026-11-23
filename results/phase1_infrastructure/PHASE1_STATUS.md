# Phase 1 status (infrastructure only; no training, no fold-10 access)
Created: docs/PAPER_VS_V2_ARCHITECTURE.md, docs/PROTOCOL_PHASE1.md, docs/REFERENCE_CHECKPOINTS.json, src/ecg_concept/{lightecgnet_paper,losses_v3,eval_metrics,splits_v3,ptbxl_guard,provenance,data_v3}.py,
scripts/{build_cache,train_v3,eval_oof,benchmark_hw}.py, experiments/phase1/*.json (7 configs), tests/test_phase1_*.py (45 tests), results/phase1_infrastructure/{hw_benchmark.json,test_results.txt}.
Modified: results/model_upgrade_audit/AUDIT_AND_PLAN.md (errata note only), tests/test_phase1_*.py during development. No existing source file, checkpoint or experiment output was changed.
Not executed: build_cache (only --dry-run), train_v3 (only --dry-run and a CPU smoke test on a synthetic cache with a throw-away model).
Experiment B launch (needs approval):
  .venv\Scripts\python.exe -I scriptsuild_cache.py --dataset ptbxl --partition train
  .venv\Scripts\python.exe -I scripts	rain_v3.py --config experiments\phase1\B_ptbxl_scratch.json --inner-fold 0   (then 1, 2, 3)
  .venv\Scripts\python.exe -I scripts\eval_oof.py --cache data\processed\ptbxl_train_v1 --runs <4 run dirs> --out results\phase1_runs\B_scratch_report.json
