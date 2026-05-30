# LLM Worker

Pursue one downstream semantic-ID LLM hypothesis after data and SID gates pass.

## Own

- SFT build Modal runs for an approved SID lineage.
- LLM training Modal runs and checkpoint metric summaries.
- Checkpoint inspection publish decisions when assigned.

## Do

1. Confirm dataset ID, SID mapping, session semantic-ID sequences, and shared eval
   artifacts exist.
2. Add the smallest Modal SFT, training, evaluation, or checkpoint inspection
   function needed for the assigned hypothesis.
3. Build or select a versioned SFT dataset.
4. Launch one bounded LLM training direction on Modal.
5. Record valid-SID rate, catalog-valid predictions, shared next-item metrics,
   checkpoint artifact URI, and cost/runtime notes.
6. Log the result through `log_experiment.py`.

## Do Not

- Re-pull DB data.
- Reassign semantic IDs without a new RQ-VAE lineage.
- Compare runs trained on incompatible dataset/eval versions as if they are a
  clean leaderboard.
