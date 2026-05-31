# auto-tiger Program

This is an autonomous research workspace for improving the current
Nipponhomes TIGER branch under a fixed Modal artifact contract.

The design is intentionally close to `karpathy/autoresearch`:

- one fixed benchmark/data file: `prepare.py`;
- one main editable experiment file: `train.py`;
- one simple runner surface: `modal_app.py`;
- one compact local ledger: `results.tsv`.

Unlike the original autoresearch setup, this project does not use git commits
and resets as the experiment ledger. Modal run IDs, artifacts, manifests, and
result rows are the research record.

## Setup

Before starting an autonomous run:

1. Agree on a run family tag, for example `tiger-rerank-may24`.
2. Read `README.md`, `prepare.py`, `train.py`,
   `reports/current_baseline.md`, and recent `results.tsv`.
3. Verify Modal input artifacts exist with:

   ```bash
   modal run modal_app.py::inspect_inputs \
     --campaign-id demo \
     --sft-run-id sft-raw-tables-001-eugene-curriculum-002
   ```

4. If `results.tsv` is empty except for the header, run the baseline first.
5. Start experiments only after the baseline contract is visible.

## Fixed Contract

Do not edit `prepare.py` in the normal loop. It owns:

- semantic-ID parsing;
- Modal artifact path expectations;
- history-next row loading;
- catalog trie construction;
- full-catalog constrained beam evaluation;
- metric definitions;
- checkpoint report format;
- local TSV row formatting.

The accepted primary metric is full-catalog constrained `Recall@10`.
Secondary metrics are `Recall@5`, `NDCG@5`, `NDCG@10`, train loss, runtime,
and peak GPU memory when available.

## Editable Surface

Edit `train.py` for one hypothesis at a time. Fair game:

- model architecture and hidden sizes;
- optimizer and scheduler;
- loss functions;
- item-level auxiliary losses;
- reranking and beam scoring;
- regularization;
- checkpoint cadence;
- small diagnostics that are emitted into manifests or metric JSONL.

Keep changes reviewable. If an idea needs a new artifact contract, add the
smallest explicit helper to `prepare.py` only after the human agrees that the
benchmark contract should change.

## Modal Execution

Data stays in Modal. Do not download raw training/eval data locally. Codex can
inspect summaries, metrics, manifests, and small examples.

Default artifact lineage:

```text
artifact_root = /vol/auto-rec-sys
campaign_id = demo
dataset_id = raw-tables-001
sft_run_id = sft-raw-tables-001-eugene-curriculum-002
```

Each experiment writes:

```text
<artifact_root>/<campaign>/<run_id>/tiger/config.json
<artifact_root>/<campaign>/<run_id>/tiger/checkpoint_metrics.jsonl
<artifact_root>/<campaign>/<run_id>/tiger/checkpoints/checkpoint-*/model.pt
<artifact_root>/<campaign>/<run_id>/tiger/checkpoint_eval_report.json
<artifact_root>/<campaign>/<run_id>/manifest.json
```

## Loop

LOOP until stopped:

1. Read the current best row and artifact report.
2. Form one hypothesis.
3. Patch `train.py` only.
4. Launch one Modal experiment with a unique `run_id`.
5. If it crashes, inspect the traceback and fix only simple implementation
   mistakes. Log fundamental idea failures as `crash`.
6. Run saved-checkpoint evaluation.
7. Append one local row to `results.tsv`.
8. If the result improves `Recall@10`, or ties while simplifying the code,
   promote it as the new baseline in `reports/current_baseline.md`.
9. Otherwise leave the result recorded and move to the next idea.

## Experiment Queue

Start with low-risk ideas from
`C:\Users\angel\Documents\projects\auto-rec-sys\reports\tiger_improvement_research.md`.
Keep each row as a separate run family unless the human explicitly asks for a
combined sweep.

### Phase 0: Guardrails and Short Runs

1. Keep full-catalog constrained `Recall@10` as the primary acceptance metric.
2. Prefer five-minute training probes before spending a full 5k-step run.
3. Add evaluation slices before changing the benchmark contract: train-seen vs
   unseen targets, history length buckets, target popularity deciles, SID
   prefix buckets, repeated prediction diversity.
4. Compare short-run loss/metrics only as triage; promote ideas using the
   full saved-checkpoint report.

### Phase 1: No-Retrain Reranking

1. `tiger-rerank-top100-pop-prefix`: generate top-50/top-100 constrained TIGER
   candidates, then keep top-10 after reranking.
2. Score candidates with TIGER logprob plus recent-history SID prefix
   compatibility.
3. Add mild popularity or novelty penalties with `gamma={0.01,0.03,0.1}` and
   report long-tail coverage.
4. Try a dense/SASRec reranker only after the cheap prefix/popularity reranker
   shows useful candidate ordering failures.

### Phase 2: Loss Alignment

1. `tiger-aux-softmax-sameprefix`: token CE plus item-level sampled softmax.
2. Sweep `lambda_item={0.05,0.1,0.25}` and negatives `{128,512,2048}`.
3. Compare uniform, in-batch, popularity, same-prefix, and adaptive hard
   negatives. Same-prefix depth candidates are `{1,2,3}`.
4. Add DuoRec/CL4SRec-style contrastive losses only after inspecting prediction
   diversity and representation collapse diagnostics.

### Phase 3: SID-Aware Model Mechanics

1. `tiger-disc-position`: learned item-position embeddings within each
   semantic ID token group.
2. Local intra-item attention or gated semantic/collaborative branches if the
   item-position probe helps.
3. RoPE/relative-position ablation is a candidate, but true RoPE requires
   replacing `nn.Transformer` attention internals; do not treat input embedding
   rotations as paper-equivalent RoPE.
4. Increase max history length only after checking history length slices.

### Phase 4: Optimizers and Stabilization

1. Schedule-Free AdamW at the same checkpoint cadence, with averaged weights
   evaluated correctly.
2. AdEMAMix, Lion, and Muon+AdamW hybrid as controlled optimizer ablations.
3. NormFormer-style stabilization, dropout, label smoothing, gradient clipping,
   and SAM after cheaper optimizer probes.

### Later Branches

Collaborative tokenizer lineage, longer or parallel semantic IDs, compact
FAME-style head/facet MoE, Switch/ST-MoE, Mamba4Rec, and LLM-based preference
conditioning are later branches. Do not start them before the reranking,
loss-alignment, and SID-position probes have ledger rows.
