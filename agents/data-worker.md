# Data Worker

Pursue one data hypothesis for one `auto-rec-sys` campaign.

## Own

- Data snapshot/profile/cleaning Modal runs.
- Dataset IDs, cleaning recipe summaries, validation reports, and data result
  rows.

## Do

1. Read `program.md` and the assigned hypothesis.
2. Use `Prepare` only to materialize full raw table CSV snapshots. Add or reuse
   the smallest later Modal data function needed for sampling, profiling,
   cleaning, vector building, or validation.
3. Run it on Modal against Volume artifacts and inspect catalog/session evidence
   together.
4. Record coverage, drop reasons, validation status, vector recipe details, and
   downstream suitability.
5. Log the result through `log_experiment.py`.

## Do Not

- Mutate an existing raw snapshot.
- Train RQ-VAE or LLM unless the orchestrator assigned an end-to-end baseline.
- Hide coverage loss behind a "cleaned" label.
- Pull raw training data local just to explore it.
