# auto-tiger Agent Instructions

Read `program.md` before changing this workspace.

- This repo is an autonomous TIGER research harness modeled after
  `karpathy/autoresearch`, adapted for Modal-native Nipponhomes artifacts.
- Treat `prepare.py` as the fixed data/evaluation contract. Do not edit it
  during an experiment unless the human explicitly changes the benchmark.
- Treat `train.py` as the main editable experiment file. Architecture,
  optimizer, training knobs, reranking, and checkpoint logic are fair game.
- Keep data execution Modal-native. Data files are expected to live in the
  existing Modal Volume, not in this checkout.
- Use Modal run IDs, manifests, checkpoint artifacts, and `results.tsv` as the
  experiment ledger. Do not use git reset/commit cycles as the research ledger.
- Log compact experiment summaries locally to `results.tsv`; use
  `log_experiment.py` or `modal_app.py::log_experiment_remote` for Deeplake
  when credentials are only available through Modal secrets.
- Preserve the full-catalog constrained evaluation as the primary acceptance
  metric. Sampled metrics are diagnostics only.
- Keep one hypothesis per run. Prefer a small, reviewable change to `train.py`
  over broad rewrites.
- Do not mutate production DB data. This project consumes prepared Modal
  artifacts only.

