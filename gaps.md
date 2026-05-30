# RQ-VAE Gaps From `eugeneyan/semantic-ids-llm`

Source inspected: `https://github.com/eugeneyan/semantic-ids-llm`, cloned to `C:\tmp\semantic-ids-llm`.

Scope: RQ-VAE only. This excludes downstream LLM/SFT work.

## Input Vectorization Gaps

- The reference RQ-VAE consumes item embeddings produced by `Qwen/Qwen3-Embedding-0.6B` from curated item text (`src/tokenize_items.py`, `src/embed_items.py`). Our current five-ablation sweep uses structured metadata vectors only. That is acceptable for the requested baseline, but it leaves a major RQ-VAE input gap: text-derived listing embeddings.
- The reference builds an `item_context` string before embedding. We do not yet have a Modal-native listing context artifact for Nipponhomes that records exactly which listing fields are included, redacted, normalized, or excluded.
- The reference L2-normalizes 1024D text embeddings. Our structured vectors are z-scored. Any mixed structured+embedding RQ-VAE input needs an explicit scaling contract so 1024 text dimensions do not dominate smaller structured feature groups.
- The reference tokenizes text separately from embedding generation. We do not yet split Nipponhomes text-context tokenization and embedding into separately manifest-backed Modal artifacts.

## RQ-VAE Model Gaps

- The reference model uses a deeper encoder/decoder (`1024 -> 512 -> 256 -> 128 -> 32`) and codebook embedding dimension 32. Our minimal runner sizes the model from vector dimension and is much smaller for the structured-feature sweep.
- The reference uses 3 residual quantization levels and later appends a uniqueness level. Our minimal runner uses 4 learned residual levels directly and does not add a deterministic uniqueness level.
- The reference uses codebook size 256. Our sweep uses codebook size 64 to keep 5-minute runs lightweight. That is useful for comparison but not enough to evaluate final semantic-ID capacity.
- The reference supports k-means codebook initialization. Our runner initializes codebooks randomly.
- The reference supports the rotation trick for vector-quantizer gradient flow. Our runner uses a plain straight-through estimator.
- The reference includes optional EMA vector quantization. Our runner only uses learnable codebooks.

## Training Loop Gaps

- The reference has cosine scheduling with warmup, gradient clipping, fused AdamW when available, and optional `torch.compile`. Our runner uses a simple fixed-learning-rate AdamW loop.
- The reference can reset unused codebook entries during training. Our runner records final codebook usage but does not reset dead or underused codes.
- The reference logs per-level codebook and commitment losses. Our runner logs total, reconstruction, and RQ loss, plus final codebook usage, but not per-level loss curves.
- The reference validates and checkpoints periodically. Our runner validates once at the end of the 5-minute budget and writes one checkpoint.
- The reference computes batch-level unique-ID proportion during training. Our runner computes full-catalog uniqueness and prefix metrics only after training.

## Assignment And Collision Gaps

- The reference separates semantic-ID generation/mapping in notebooks after RQ-VAE training. Our runner writes `listing_semantic_ids.parquet` as part of training, but we do not yet have a separate callable assignment stage that loads a selected checkpoint.
- Current `assignability` means every vector received a code. It does not mean IDs are unique.
- We need deterministic collision resolution, probably a uniqueness suffix level, before any RQ-VAE output should be treated as final semantic IDs.
- The current prefix metrics are useful but incomplete: we record `prefix_spread_l1`, `prefix_spread_l2`, `max_prefix_size_l2`, and full-code collision counts, but not entropy/Gini per level or per-prefix distribution summaries.

## Modal Artifact Gaps

- The reference writes local `data/output` and `checkpoints/rqvae` paths. Our implementation needs every RQ-VAE input, checkpoint, metric file, and assignment mapping under the Modal Volume lineage.
- The current runner writes minimal manifests for vectors, metrics, checkpoints, and semantic IDs. It still needs richer config capture: exact feature columns, dtype decisions, normalization statistics, codebook config, model architecture, optimizer config, and W&B run URL.
- Failed RQ-VAE attempts should leave crash manifests under the intended run path. The wrapper now returns crash summaries, but the child script should also write a structured `manifest.json` on pre-training failures.
- The current ablation vectors are written under `datasets/<dataset_id>/vectors/<ablation>`. If we later rerun with changed vectorization logic, those paths need versioned vector IDs or run-stamped subdirectories to avoid ambiguity.

## Recommended RQ-VAE Next Additions

1. Finish the five structured-feature ablations and append their result rows.
2. Add a Modal text-context builder and Qwen embedding stage as separate dataset artifacts.
3. Add `text_embedding` and `geo_plus_core_plus_text_embedding` RQ-VAE ablations.
4. Add a separate assignment callable that loads a checkpoint and writes SID mappings.
5. Add deterministic collision resolution and stronger prefix/entropy metrics.
6. Test k-means initialization and 256-code codebooks after the fast 64-code baseline identifies the best input feature family.
