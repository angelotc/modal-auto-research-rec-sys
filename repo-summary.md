# semantic-ids-llm — Repository Summary

**Author:** Eugene Yan  
**Source:** https://github.com/eugeneyan/semantic-ids-llm  
**License:** MIT  
**Stars:** 122 | **Forks:** 29

---

## What It Is

An experimental LLM-recommender hybrid that can make recommendations via conversation. Instead of using retrieval or external tools, the model natively understands items as part of its vocabulary — it's "bilingual" in English and item IDs.

**Example:**
```
INPUT = "I like animal and cute games. <|rec|>"
>>> "Animal Crossing: New Leaf", "DISNEY INFINITY Starter Pack", "Nintendogs + Cats"
```

## Core Concept: Semantic IDs

Traditional recommender systems use meaningless hash IDs (e.g., `B0040JHNQG`). **Semantic IDs** (`<|sid_0|><|sid_256|><|sid_512|><|sid_768|>`) are hierarchical tokens that encode item information, where similar items share common prefixes. This enables:

1. **Native item understanding** — items are tokens, not retrieved entities
2. **Recommendation via generation** — predict next items from user history
3. **Natural language steering** — steer recommendations through chat
4. **Explainability** — reason about why items are similar

---

## Project Structure

```
semantic-ids-llm/
├── notebooks/                          # 10 Jupyter notebooks for data pipeline
│   ├── 01-prep-items-and-sequences.ipynb
│   ├── 02-clean-descriptions.ipynb
│   ├── 03-clean-titles.ipynb
│   ├── 04-augment-metadata.ipynb
│   ├── 05-update-items-and-sequences.ipynb
│   ├── 06-get-semantic-ids-per-asin.ipynb
│   ├── 07-get-semantic-ids-to-asin-sequences.ipynb
│   ├── 08-prep-finetuning-data.ipynb
│   ├── 09-evaluate-sasrec-baseline.ipynb
│   └── 10-evaluate-sasrec-semantic.ipynb
├── src/
│   ├── __init__.py
│   ├── device_manager.py               # GPU/CPU device detection
│   ├── embed_items.py                  # Item embedding with Qwen3-0.6B
│   ├── train_rqvae.py                  # RQ-VAE for semantic ID generation (~1018 lines)
│   ├── train_sasrec.py                 # Baseline SASRec recommender
│   ├── train_sasrec_semantic_id.py     # Semantic ID SASRec variant
│   ├── finetune_qwen3_8b_vocab.py      # Stage 1: Vocabulary extension
│   ├── finetune_qwen3_8b_full.py       # Stage 2: Full model finetuning
│   ├── tokenize_items.py               # Item tokenization utilities
│   ├── test_prompts.py                 # Evaluation prompts
│   └── logger.py                       # Logging setup
├── demo.ipynb                          # Interactive demo notebook
├── pyproject.toml                      # Dependencies (uv-managed)
├── setup.sh                            # GPU instance setup script
└── README.md
```

---

## Training Pipeline

### Stage 1: Data Preparation
- **Dataset:** Amazon Reviews 2023 — Video Games category
- **Stats:** 66k products, 79k user purchase sequences (avg length: 6.5 items)
- **Cleaning:** Gemini 2.5 Flash for quality filtering
- Notebooks 01–08 handle item prep, description/title cleaning, metadata augmentation, and semantic ID assignment

### Stage 2: Semantic ID Generation (RQ-VAE)
- **File:** `src/train_rqvae.py`
- **Architecture:** Residual Quantized VAE (RQ-VAE)
  - Encoder: 1024 → [512, 256, 128] → 32 (codebook embedding dim)
  - 3 quantization levels, 256 codes per codebook
  - Decoder mirrors encoder in reverse
- **Key features:**
  - Rotation trick STE for gradient flow (from [arXiv:2410.06424](https://arxiv.org/abs/2410.06424))
  - K-means codebook initialization
  - Periodic unused code reset
  - EMA quantizer option (disabled by default — noted as not working well)
  - Cosine LR scheduler with warmup
  - WandB logging and checkpointing
- **Input:** Item embeddings (from Qwen3-0.6B, 1024-dim)
- **Output:** Hierarchical discrete codes (semantic IDs)

### Stage 3: Baseline Comparison
- **Files:** `src/train_sasrec.py`, `src/train_sasrec_semantic_id.py`
- Trains SASRec (Self-Attentive Sequential Recommendation) baselines with both regular and semantic IDs

### Stage 4: LLM Finetuning (Qwen3-8B)
- **Stage 1 — Vocabulary extension:** `src/finetune_qwen3_8b_vocab.py` — adds semantic ID tokens to the model vocabulary
- **Stage 2 — Full finetuning:** `src/finetune_qwen3_8b_full.py` — end-to-end finetuning on recommendation data

---

## Key Results

| Model | Hit@10 | NDCG@10 | MRR |
|-------|--------|---------|-----|
| Baseline SASRec | 0.281 | 0.154 | 0.130 |
| Semantic ID SASRec | 0.220 | 0.114 | 0.101 |

The semantic ID model trades some accuracy for:
- Cold-start handling via shared prefixes
- Natural language steerability
- Explainability of recommendations

---

## Trained Models (HuggingFace)

- [`eugeneyan/video-games-semantic-ids-mapping`](https://huggingface.co/datasets/eugeneyan/video-games-semantic-ids-mapping) — Item ↔ semantic ID mappings
- [`eugeneyan/semantic-id-qwen3-8b-video-games`](https://huggingface.co/eugeneyan/semantic-id-qwen3-8b-video-games) — Finetuned Qwen3-8B model

---

## Key Technical Details (train_rqvae.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `item_embedding_dim` | 1024 | Input embedding dim (Qwen3-0.6B) |
| `encoder_hidden_dims` | [512, 256, 128] | Encoder layer sizes |
| `codebook_embedding_dim` | 32 | Codebook vector dimension |
| `codebook_quantization_levels` | 3 | Hierarchical quantization depth |
| `codebook_size` | 256 | Codes per codebook (8-bit) |
| `commitment_weight` | 0.25 | VQ commitment loss weight |
| `batch_size` | 32768 | Training batch size |
| `max_lr` | 3e-4 | Peak learning rate |
| `num_epochs` | 20000 | Training epochs |
| `use_rotation_trick` | True | Better gradient flow through VQ |
| `use_kmeans_init` | True | K-means codebook initialization |
| `reset_unused_codes` | True | Periodically reset dead codes |

**Dependencies:** PyTorch, Polars, scikit-learn, WandB  
**Requirements:** Python 3.12+, CUDA GPU (48GB+ VRAM for training)

---

## References

- [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (Rajput et al., 2023)
- [Better Generalization with Semantic IDs](https://arxiv.org/abs/2306.08121) (Singh et al., 2024)
- [RQ-VAE: Residual Quantized VAE](https://arxiv.org/abs/2107.03312) (Zeghidour et al., 2021)
- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (Kang & McAuley, 2018)
- [Rotation Trick STE](https://arxiv.org/abs/2410.06424)
