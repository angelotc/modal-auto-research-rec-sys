# auto-rec-sys Program

This file is the research contract for `auto-rec-sys`, an autonomous
recommendation-systems researcher. It is modeled after the compact
`autoresearch` loop: build the research workbench incrementally, keep produced
artifacts and accepted evaluation contracts reproducible, and record every
result so later runs can learn from evidence instead of chat history.

Unlike a local code-research loop that commits or reverts source edits after
each trial, `auto-rec-sys` runs experiments through Modal. A trial is a Modal
run with versioned configuration, artifacts, metrics, and an agent decision
record. Codex may change this folder to create the next useful Modal research
function when the current workbench cannot answer the next question.

## Objective

Research recommendation recipes for Nipponhomes from a real catalog and real
listing-view sessions under a fixed compute budget.

The first research path is:

1. Pull a versioned catalog and session snapshot from PostgreSQL.
2. Clean and profile both halves of that snapshot.
3. Train an RQ-VAE over eligible listing vectors.
4. Assign unique semantic IDs to the catalog and map sessions to those IDs.
5. Build LLM SFT data from the cleaned semantic-ID sequences.
6. Train and evaluate a semantic-ID LLM for next-listing recommendation.
7. Serve selected checkpoints for inspection.

SASRec is an optional baseline branch after the cleaned catalog/session contract
and shared next-item evaluation split exist.

## Files

The intended shape of this folder is:

- `README.md`: setup and entrypoints for Codex CLI and Modal.
- `AGENTS.md`: local Codex guardrails for this workspace.
- `program.md`: this research contract.
- `orchestrator.md`: principal-investigator loop for Codex.
- `agents/*.md`: focused worker contracts.
- `prepare.py`: starting data-lab implementation for DB snapshots, data
  exploration, cleaning, validation, and listing-vector materialization. Codex
  may extend it when a concrete data question needs a new Modal-backed step.
- `modal_app.py`: starting Modal workbench. Codex may add small resource-specific
  functions for data, RQ-VAE, SFT build, LLM train, evaluation, and checkpoint
  inspection as the research loop reaches them.
- `results.tsv`: append-only local ledger mirror/header.

Existing `random-one-off-scripts/semantic_ids` scripts are starting research
material. The new system may wrap or extract from them, but `auto-rec-sys`
should make its own artifact contracts explicit.

## Immutable Contract

Treat these as stable once they have been emitted or accepted for comparison:

- Raw DB snapshot artifacts written by Modal functions.
- Dataset version manifests, cleaning recipes, validation reports, and train/eval
  split manifests emitted by the Modal workbench.
- Shared evaluation code and metric definitions used to compare accepted runs.
- Modal artifact layout under a campaign/run ID.
- `results.tsv` column meanings.

The agent may add Modal functions and create new dataset versions while it
discovers what the research path needs. It must not silently mutate an existing
raw snapshot or cleaned dataset.
It must not use git commits or git reverts as the experiment ledger; Modal run
artifacts and result manifests are the trial history.

## Modal Owns The Data

Data exploration, cleansing, vector materialization, and training run on Modal.
Codex writes and invokes the next Modal function needed to answer the current
research question instead of downloading the data locally.

The workbench should grow from small inspectable functions such as snapshots,
profiles, samples, cleaning passes, validations, vector builders, model runs,
and evaluators. Prefer one function that answers a concrete next question over
an opaque generic remote Python executor.

The initial `Prepare` path pulls full raw source tables as large CSV artifacts:

- The full listing catalog table needed to inspect items and resolve predictions.
- The full listing-view/session table needed to inspect interaction sequences.
- Snapshot metadata required for reproducibility, such as extraction timestamp,
  source query/version, row counts, time bounds, and schema/feature manifests.

`Prepare` should stay intentionally simple: read-only DB extraction into raw CSV
snapshots plus metadata. Codex-authored Modal functions do the post-processing
afterward: profiling, sampling, cleaning, vector construction, validation,
dataset splits, training inputs, and model/eval artifacts.

Cleaning must treat the extracted tables as a paired dataset version:

- Catalog cleaning decides which listings are eligible for vectors, semantic IDs,
  recommendation targets, display/inspection, or rejection with a reason.
- Session cleaning orders events, removes invalid or duplicate interactions,
  handles ineligible catalog references, filters unusable sequences, and records
  all drop counts.

Before any training stage consumes a dataset version, Modal must write enough
profiles, recipes, validation reports, and vector manifests for the run to be
reproducible. Downstream training jobs consume Modal artifacts; they do not
query the DB directly.

Expected raw prepared artifacts:

```text
artifacts/<campaign_id>/<dataset_id>/
  manifest.json
  raw/
    listings.csv
    listing_views.csv
    table_metadata.json
```

Later Modal post-processing functions may add derivative artifacts such as:

```text
artifacts/<campaign_id>/<dataset_id>/
  profiles/
    catalog_before.json
    catalog_after.json
    sessions_before.json
    sessions_after.json
  clean/
    catalog.parquet
    sessions.parquet
    listing_vectors.parquet
  reports/
    cleaning_recipe.json
    drop_report.json
    validation_report.json
    split_manifest.json
```

`listing_vectors.parquet` is the RQ-VAE item input. The first research vector is
owned by this pipeline and includes location features; it is not assumed to be
the production recommendation vector. Clean session artifacts are the common
behavior source for semantic-ID LLM and SASRec experiments.

## Experiment Surface

The agent is allowed to research:

- Catalog/session cleaning recipes that create new dataset versions.
- RQ-VAE hyperparameters and training budgets.
- Semantic-ID gates and checkpoint choice from recorded prefix metrics.
- LLM SFT recipe knobs after dataset lineage is preserved.
- Which downstream LLM to post-train for the approved dataset and task shape.
- LLM training hyperparameters, checkpoint selection, and Modal GPU choice.
- Whether a SASRec baseline or additional branch is worth compute under budget.

The first bounded model knobs should be small:

- RQ-VAE: codebook size, commitment weight, codebook embedding dimension,
  training budget.
- LLM: candidate base model, model/stage, max steps, learning rate, LoRA rank,
  effective batch size, sequence length, and GPU/resource choice.
- Data: allowed catalog eligibility rules, session dedupe/filter thresholds, and
  train/eval split policy versions.

The agent may add source code when the workbench needs a new function, metric,
recipe, or model branch. It must not mutate production DB data, silently change
raw snapshot lineage, or hide one-off transforms outside Modal artifacts.
Source DB access is read-only: use only read queries for discovery and snapshot
extraction. Do not write, update, delete, migrate, or otherwise change the
source database from `auto-rec-sys`.

## Modal Compute Roles

Use one Modal app, one shared artifact Volume, and resource-specific functions.
These are workbench directions, not a closed API:

When adding or changing Modal Python SDK code, consult Context7 for the current
Modal SDK docs and examples before finalizing the implementation. Prefer
current Modal patterns over remembered or stale API usage.

A Modal secret is available inside remote functions that explicitly mount it;
do not assume local Codex shell commands can read it.

1. `data`: CPU/RAM role running `prepare.py` extraction, profiling, cleaning,
   validation, and prepared dataset materialization.
2. `rqvae`: GPU role for RQ-VAE training plus callable semantic-ID
   assignment/inference over a selected checkpoint.
3. `sft`: CPU/RAM role for downstream LLM SFT and shared eval artifact build.
4. `llm_train`: GPU role for Unsloth/Qwen training and checkpoint writes.
5. `checkpoint_app`: web/inference role that loads checkpoints and semantic-ID
   lookup artifacts for human inspection.

Keep RQ-VAE training and semantic-ID assignment in the same Modal role at first,
but expose them as separate callable functions. Split them later only if
assignment scheduling or hardware requirements justify it.

## Research Loop

Run the agent as a Codex-driven research loop during discovery and development.
The agent should build and use Modal functions over structured artifacts:

1. Read the current campaign objective, budget, results ledger, and prior run
   manifests.
2. Inspect data profiles, validation reports, metrics, and concise log summaries.
3. If the current Modal workbench cannot answer the next question, add the
   smallest function or metric that can.
4. Form one hypothesis and one bounded experiment.
5. Launch only the required Modal functions.
6. Record config, metrics, artifact pointers, failures, and decision rationale.
7. Compare against prior runs and either continue, branch, or stop.

Expensive LLM training is gated on a prepared dataset plus successful semantic-ID
assignment. RQ-VAE validation loss alone is not enough; prefix spread and
assignability must be recorded before downstream semantic-ID LLM work.

The agent must choose the post-training LLM and Modal GPU from evidence. Before
launching an expensive LLM run, inspect the approved SFT/eval dataset size,
sequence-length distribution, semantic-token vocabulary growth, task mix,
checkpoint/eval needs, budget, and available GPU memory trade-offs. Record the
chosen model, GPU/resource config, and rationale in the run manifest. Fixed
defaults may be used for the first baseline only when the rationale states they
are a baseline starting point.

Do not let training continue blindly when metrics look wrong. Interrupt or stop
the Modal run, preserve its logs/checkpoints/metrics, and diagnose before
retrying when there are suspicious signals such as non-finite loss, clear metric
corruption, invalid recommendation outputs, a failed semantic-ID validity gate,
unexpected data collapse, or resource behavior that invalidates the experiment.

Every branch in this loop is represented by a new Modal-backed run or dataset
version. Failed trials remain inspectable for the agent and humans; they are not
discarded by reverting the working tree.

For this iteration, stop the autonomous build/research loop once there is a
testable post-trained LLM: a versioned model/checkpoint artifact plus a working
evaluation or inspection path that can load it and exercise recommendation
behavior. Record what was built, what data lineage produced it, and the next
research directions before stopping.

## Modal Artifacts

Every run should be self-describing:

```text
/vol/auto-rec-sys/<campaign_id>/<run_id>/
  manifest.json
  data/
  rqvae/
    config.json
    checkpoints/
    metrics.json
  semantic_ids/
  datasets/
    llm_sft/
    sasrec/
    eval/
  llm/
    config.json
    checkpoints/
    metrics.json
  reports/
    agent_decision.json
    summary.md
```

## Metrics And Results

Keep a local `results.tsv` mirror/header with at least:

```text
run_id	dataset_id	parent_run_id	stage	hypothesis	status	primary_metric	coverage	cost_notes	artifact_uri	decision
```

Stage-specific metrics belong in run manifests and metric JSON files:

- Data: catalog/session retained counts, drop reasons, missingness/outlier
  summaries, split counts, item/session coverage.
- RQ-VAE: validation loss, prefix spread, max prefix size, codebook usage,
  assignability, selected checkpoint.
- Semantic IDs: catalog mapping coverage, collision-resolution stats, session
  mapping coverage.
- LLM: valid semantic-ID rate, catalog-valid prediction rate, next-item ranking
  metrics on the shared eval split, checkpoint cost/time.
- SASRec if present: next-item ranking metrics on the same shared eval split and
  runtime/cost.

## Done For A First Demo

A first convincing run proves that the system can:

1. Pull and validate one catalog/session dataset version through `prepare.py`.
2. Run an RQ-VAE experiment on Modal and decide from assignability evidence.
3. Assign semantic IDs and build an LLM SFT dataset without local handoff files.
4. Train at least one LLM checkpoint on Modal.
5. Inspect a checkpoint in the Modal webapp.
6. Record the experiment trail and explain the next research decision.
