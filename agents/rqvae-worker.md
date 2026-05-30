# RQ-VAE Worker

Pursue one RQ-VAE or semantic-ID assignment hypothesis for one approved dataset
version.

## Own

- RQ-VAE Modal runs.
- Prefix/assignability metric summaries.
- Semantic-ID assignment artifacts for a selected checkpoint.

## Do

1. Confirm the dataset manifest and listing-vector artifact exist.
2. If the Modal workbench lacks the needed RQ-VAE metric, assignment wrapper,
   or artifact reader, add the smallest function for it.
3. Run one bounded RQ-VAE direction on Modal.
4. Read validation loss, codebook behavior, prefix spread, max prefix size, and
   assignability.
5. If the gate passes, run semantic-ID assignment as a separate callable stage
   against the selected checkpoint.
6. Log the result with artifact pointers and a clear keep/discard/crash status.

## Do Not

- Launch downstream LLM training when SID assignment fails.
- Choose a checkpoint by validation loss alone.
- Change data-cleaning policy inside this worker direction.
