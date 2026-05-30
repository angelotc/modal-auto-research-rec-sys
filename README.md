# auto-rec-sys

`auto-rec-sys` is an autonomous recommender-systems research workspace for
Nipponhomes. The first research path uses catalog and listing-view session data
to explore a semantic-ID pipeline:

```text
DB snapshot -> paired data prep -> RQ-VAE -> semantic IDs -> LLM SFT -> LLM checkpoints
```

The control plane is a Codex CLI research loop. Modal runs data exploration,
cleansing, vector materialization, training, evaluation, and checkpoint-serving
workloads.

## Files

- `program.md`: research contract, artifacts, metrics, and Modal role plan.
- `AGENTS.md`: local Codex instructions for this workspace.
- `orchestrator.md`: principal-investigator loop for Codex.
- `agents/*.md`: focused worker briefs for data, RQ-VAE, and LLM experiments.
- `prepare.py`: starting data-lab implementation for catalog/session snapshots
  and dataset transforms.
- `modal_app.py`: starting Modal workbench that Codex extends as research needs
  new data, model, evaluation, or inspection functions.
- `pyproject.toml`: local Python dependencies for `uv`.
- `results.tsv`: local append-only ledger mirror/header.

The existing `../semantic_ids` folder is the first downstream implementation
source for RQ-VAE, semantic-ID assignment, SFT generation, and Unsloth training.
This folder defines the autonomous research layer around those stages.

## Setup

Run from this folder:

```powershell
cd random-one-off-scripts/auto-rec-sys
uv sync
```

Provide DB credentials to Modal:

```powershell
modal setup
$env:DATABASE_URL="postgresql://..."
modal secret create nipponhomes-auto-rec-sys-db DATABASE_URL="$env:DATABASE_URL"
```

## Snapshot Data

Data work is Modal-native. The starting `Prepare` step pulls whole read-only
source tables as raw CSV snapshots:

- `listings`,
- `listing_views`.

It writes versioned raw CSV artifacts and table metadata before Codex adds later
Modal functions for profiling, sampling, cleaning, vector building, validation,
dataset splits, and training inputs:

```powershell
uv run modal run modal_app.py --campaign-id demo --dataset-id snapshot-001
```

Artifacts live on the Modal Volume under:

```text
/vol/auto-rec-sys/<campaign_id>/datasets/<dataset_id>/
```

Codex is expected to add Modal functions as it progresses. Examples include
snapshot, profile, sample, clean, vector-build, validate, train, evaluate, and
checkpoint-inspection functions. Prefer a small function that answers the next
research question over moving raw data local or adding an opaque remote code
executor.

The starting cleaning path builds an `auto-rec-sys` structured listing vector
with latitude and longitude features for RQ-VAE experiments. It does not reuse
the production 9D recommendation vector.

## Log Experiments

Modal artifacts remain the source of truth for datasets, checkpoints, metric
JSON, reports, and logs.

## Start Codex

Start Codex CLI from the repo or this folder and give it the orchestrator:

```text
Read random-one-off-scripts/auto-rec-sys/orchestrator.md and start a baseline
auto-rec-sys cycle. Use Modal runs for experiments and result rows.
```

The orchestrator should first establish one prepared dataset snapshot and one
baseline result before it fans out focused workers.

## Current Modal Roles

The scaffold starts with a data role and grows through Codex edits. The planned
resource directions are:

1. `data`: DB snapshot, profiling, paired catalog/session cleaning.
2. `rqvae`: RQ-VAE training plus callable semantic-ID assignment.
3. `sft`: CPU/RAM SFT and shared eval dataset build.
4. `llm_train`: GPU LLM training and checkpoint writes.
5. `checkpoint_app`: checkpoint inference and inspection webapp.

The run lineage is artifact-first. Experiments do not depend on git
commit/revert loops.
