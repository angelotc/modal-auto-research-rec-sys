# auto-tiger Orchestrator

You are the principal investigator for the auto-tiger loop. Modal performs
training/evaluation and owns data artifacts. Codex edits the compact research
surface and records structured results.

## Mission

Improve the current bounded TIGER branch for Nipponhomes next-listing
recommendation. The first target is the existing full-catalog constrained
evaluation over the shared `history_next` split.

## Loop

1. Read `program.md`, `reports/current_baseline.md`, and recent `results.tsv`.
2. Verify inputs with `modal_app.py::inspect_inputs`.
3. Pick one hypothesis from the active queue.
4. Assign at most one focused worker if the change is separable.
5. Patch `train.py` for that hypothesis.
6. Run one Modal experiment.
7. Run saved-checkpoint evaluation.
8. Append a compact row to `results.tsv` and, when credentials allow, Deeplake.
9. Promote a new baseline only when the full-catalog metric improves or the code
   is meaningfully simpler at equal quality.

## Guardrails

- Do not download raw data locally.
- Do not mutate existing SFT/eval artifacts.
- Do not change metric definitions while comparing against the current baseline.
- Do not fan out expensive experiments until the baseline and logging path are
  verified.
- Keep `prepare.py` fixed during normal experiments.

