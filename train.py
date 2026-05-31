#!/usr/bin/env python3
"""Train a bounded TIGER-style generative retriever on shared SID histories."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger("auto_tiger.train")
SID_PATTERN = re.compile(r"<\|sid_start\|>(?:<\|sid_\d+\|>)+<\|sid_end\|>")
SID_TOKEN_PATTERN = re.compile(r"<\|sid_(\d+)\|>")
PAD = 0
BOS = 1
EOS = 2
SID_OFFSET = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_root(artifact_root: Path, campaign_id: str, run_id: str) -> Path:
    return artifact_root / campaign_id / run_id


def semantic_tokens(semantic_id: str) -> tuple[int, ...]:
    return tuple(int(token) + SID_OFFSET for token in SID_TOKEN_PATTERN.findall(semantic_id))


def read_catalog(path: Path) -> tuple[list[str], dict[tuple[int, ...], str]]:
    semantic_ids = pd.read_parquet(path, columns=["semantic_id"])["semantic_id"].astype(str).drop_duplicates().tolist()
    tokens_to_sid = {semantic_tokens(semantic_id): semantic_id for semantic_id in semantic_ids}
    if not tokens_to_sid or any(not tokens for tokens in tokens_to_sid):
        raise ValueError(f"catalog has unusable semantic IDs: {path}")
    lengths = {len(tokens) for tokens in tokens_to_sid}
    if len(lengths) != 1:
        raise ValueError(f"TIGER requires fixed-length semantic IDs, found lengths={sorted(lengths)}")
    return semantic_ids, tokens_to_sid


def pad_history(tokens: list[int], max_history_tokens: int) -> list[int]:
    tokens = tokens[-max_history_tokens:]
    return [PAD] * (max_history_tokens - len(tokens)) + tokens


def history_rows(
    path: Path,
    catalog_tokens: set[tuple[int, ...]],
    *,
    max_history_items: int,
    sid_length: int,
    limit: int,
) -> list[tuple[list[int], tuple[int, ...], str]]:
    rows: list[tuple[list[int], tuple[int, ...], str]] = []
    max_history_tokens = max_history_items * sid_length
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("task") != "history_next":
                continue
            history = [
                semantic_tokens(semantic_id)
                for semantic_id in SID_PATTERN.findall(str(payload.get("input", "")))
            ]
            history = [tokens for tokens in history if tokens in catalog_tokens]
            target_sid = str(payload.get("output", ""))
            target = semantic_tokens(target_sid)
            if not history or target not in catalog_tokens:
                continue
            flat_history = [token for tokens in history[-max_history_items:] for token in tokens]
            rows.append((pad_history(flat_history, max_history_tokens), target, target_sid))
            if limit and len(rows) >= limit:
                break
    return rows


class HistoryNextDataset(Dataset):
    def __init__(
        self,
        rows: list[tuple[list[int], tuple[int, ...], str]],
        item_index_by_tokens: dict[tuple[int, ...], int],
    ):
        self.rows = rows
        self.item_index_by_tokens = item_index_by_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history, target, _semantic_id = self.rows[index]
        decoder_input = (BOS, *target)
        labels = (*target, EOS)
        return (
            torch.tensor(history, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(self.item_index_by_tokens[target], dtype=torch.long),
        )


@dataclass(frozen=True)
class TIGERConfig:
    vocab_size: int
    max_history_tokens: int
    target_tokens: int
    hidden_size: int
    layers: int
    heads: int
    feedforward_size: int
    dropout: float
    activation: str = "relu"
    norm_first: bool = True
    use_sid_position_embeddings: bool = False
    use_input_norm: bool = False


def normalize_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    values = dict(payload)
    values.setdefault("activation", "relu")
    values.setdefault("norm_first", True)
    values.setdefault("use_sid_position_embeddings", False)
    values.setdefault("use_input_norm", False)
    return values


class TIGER(nn.Module):
    def __init__(self, config: TIGERConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=PAD)
        self.encoder_positions = nn.Embedding(config.max_history_tokens, config.hidden_size)
        self.decoder_positions = nn.Embedding(config.target_tokens + 1, config.hidden_size)
        self.input_norm = nn.LayerNorm(config.hidden_size) if config.use_input_norm else None
        self.sid_positions = (
            nn.Embedding(config.target_tokens + 1, config.hidden_size)
            if config.use_sid_position_embeddings
            else None
        )
        self.transformer = nn.Transformer(
            d_model=config.hidden_size,
            nhead=config.heads,
            num_encoder_layers=config.layers,
            num_decoder_layers=config.layers,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.encoder_positions.weight, std=0.02)
        nn.init.normal_(self.decoder_positions.weight, std=0.02)
        if self.sid_positions is not None:
            nn.init.normal_(self.sid_positions.weight, std=0.02)
            with torch.no_grad():
                self.sid_positions.weight[0].zero_()
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()

    def sid_position_ids(self, tokens: torch.Tensor, *, decoder: bool) -> torch.Tensor:
        if decoder:
            ids = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
            return ids.clamp(max=self.config.target_tokens)
        ids = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        sid_ids = (ids % self.config.target_tokens) + 1
        return sid_ids.masked_fill(tokens.eq(PAD), 0)

    def positional(self, tokens: torch.Tensor, positions: nn.Embedding, *, decoder: bool = False) -> torch.Tensor:
        ids = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        embedded = self.embedding(tokens) * math.sqrt(self.config.hidden_size) + positions(ids)
        if self.sid_positions is not None:
            embedded = embedded + self.sid_positions(self.sid_position_ids(tokens, decoder=decoder))
        if self.input_norm is not None:
            embedded = self.input_norm(embedded)
        return embedded

    def item_embedding(self, sid_tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sid_tokens) * math.sqrt(self.config.hidden_size)
        if self.sid_positions is not None:
            ids = torch.arange(1, sid_tokens.shape[-1] + 1, device=sid_tokens.device)
            embedded = embedded + self.sid_positions(ids.expand_as(sid_tokens))
        return embedded.mean(dim=-2)

    def encode(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = history.eq(PAD)
        memory = self.transformer.encoder(
            self.positional(history, self.encoder_positions),
            src_key_padding_mask=padding_mask,
        )
        return memory, padding_mask

    def decode(
        self,
        decoder_input: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        length = decoder_input.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=decoder_input.device, dtype=torch.bool),
            diagonal=1,
        )
        decoded = self.transformer.decoder(
            self.positional(decoder_input, self.decoder_positions, decoder=True),
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output(self.output_norm(decoded))

    def forward(self, history: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        memory, memory_padding_mask = self.encode(history)
        return self.decode(decoder_input, memory, memory_padding_mask)


def trie_for_catalog(catalog_tokens: set[tuple[int, ...]]) -> dict[int, Any]:
    trie: dict[int, Any] = {}
    for token_tuple in catalog_tokens:
        node = trie
        for token in token_tuple:
            node = node.setdefault(token, {})
    return trie


def trie_node(trie: dict[int, Any], prefix: tuple[int, ...]) -> dict[int, Any]:
    node = trie
    for token in prefix:
        node = node[token]
    return node


def rank_sid_tokens(
    model: TIGER,
    history: torch.Tensor,
    trie: dict[int, Any],
    *,
    target_tokens: int,
    beam_size: int,
) -> list[tuple[tuple[int, ...], float]]:
    memory, memory_padding_mask = model.encode(history)
    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for _depth in range(target_tokens):
        decoder_inputs = torch.tensor(
            [[BOS, *prefix] for prefix, _score in beams],
            dtype=torch.long,
            device=history.device,
        )
        logits = model.decode(
            decoder_inputs,
            memory.expand(len(beams), -1, -1),
            memory_padding_mask.expand(len(beams), -1),
        )[:, -1]
        log_probs = logits.log_softmax(dim=-1)
        next_beams: list[tuple[tuple[int, ...], float]] = []
        for index, (prefix, score) in enumerate(beams):
            for token in trie_node(trie, prefix):
                next_beams.append(((*prefix, token), score + float(log_probs[index, token].item())))
        beams = sorted(next_beams, key=lambda row: row[1], reverse=True)[:beam_size]
    return beams


def history_item_tokens(history: list[int], target_tokens: int) -> list[tuple[int, ...]]:
    tokens = [token for token in history if token != PAD]
    item_count = len(tokens) // target_tokens
    return [
        tuple(tokens[index * target_tokens : (index + 1) * target_tokens])
        for index in range(item_count)
    ]


def prefix_compatibility(
    candidate: tuple[int, ...],
    history_items: list[tuple[int, ...]],
    *,
    max_depth: int = 0,
    recency_mode: str = "inverse",
) -> float:
    if not history_items:
        return 0.0
    depth = min(max_depth if max_depth > 0 else len(candidate), len(candidate))
    total = 0.0
    weight_total = 0.0
    for offset, item in enumerate(reversed(history_items), start=1):
        if recency_mode == "uniform":
            weight = 1.0
        elif recency_mode == "exp":
            weight = 0.5 ** (offset - 1)
        else:
            weight = 1.0 / offset
        matches = 0
        for candidate_token, history_token in zip(candidate[:depth], item[:depth], strict=False):
            if candidate_token != history_token:
                break
            matches += 1
        total += weight * (matches / max(depth, 1))
        weight_total += weight
    return total / max(weight_total, 1e-12)


def target_popularity(rows: list[tuple[list[int], tuple[int, ...], str]]) -> dict[tuple[int, ...], float]:
    counts: dict[tuple[int, ...], int] = {}
    for _history, target, _target_sid in rows:
        counts[target] = counts.get(target, 0) + 1
    max_count = max(counts.values(), default=0)
    if max_count <= 0:
        return {}
    denom = math.log1p(max_count)
    return {tokens: math.log1p(count) / denom for tokens, count in counts.items()}


def ranking_metrics(expected: str, ranked_sids: list[str]) -> dict[str, float]:
    metrics = {"recall_at_5": 0.0, "ndcg_at_5": 0.0, "recall_at_10": 0.0, "ndcg_at_10": 0.0}
    if expected not in ranked_sids:
        return metrics
    rank = ranked_sids.index(expected)
    if rank < 5:
        metrics["recall_at_5"] = 1.0
        metrics["ndcg_at_5"] = 1.0 / math.log2(rank + 2)
    if rank < 10:
        metrics["recall_at_10"] = 1.0
        metrics["ndcg_at_10"] = 1.0 / math.log2(rank + 2)
    return metrics


def evaluate(
    model: TIGER,
    rows: list[tuple[list[int], tuple[int, ...], str]],
    tokens_to_sid: dict[tuple[int, ...], str],
    trie: dict[int, Any],
    device: torch.device,
    *,
    target_tokens: int,
    beam_size: int,
    rerank_prefix_bonus: float,
    rerank_prefix_depth: int,
    rerank_recency_mode: str,
    rerank_popularity_penalty: float,
    popularity_by_tokens: dict[tuple[int, ...], float] | None = None,
) -> dict[str, Any]:
    training = model.training
    model.eval()
    sums = {"recall_at_5": 0.0, "ndcg_at_5": 0.0, "recall_at_10": 0.0, "ndcg_at_10": 0.0}
    examples: list[dict[str, Any]] = []
    with torch.no_grad():
        for history, _target, expected_sid in rows:
            history_tensor = torch.tensor([history], dtype=torch.long, device=device)
            ranked_tokens = rank_sid_tokens(
                model,
                history_tensor,
                trie,
                target_tokens=target_tokens,
                beam_size=beam_size,
            )
            if rerank_prefix_bonus or rerank_popularity_penalty:
                history_items = history_item_tokens(history, target_tokens)
                ranked_tokens = sorted(
                    ranked_tokens,
                    key=lambda row: (
                        row[1]
                        + rerank_prefix_bonus
                        * prefix_compatibility(
                            row[0],
                            history_items,
                            max_depth=rerank_prefix_depth,
                            recency_mode=rerank_recency_mode,
                        )
                        - rerank_popularity_penalty
                        * (popularity_by_tokens or {}).get(row[0], 0.0)
                    ),
                    reverse=True,
                )
            ranked_sids = [tokens_to_sid[tokens] for tokens, _score in ranked_tokens[:10]]
            for key, value in ranking_metrics(expected_sid, ranked_sids).items():
                sums[key] += value
            if len(examples) < 5:
                examples.append({"expected": expected_sid, "top_10": ranked_sids[:10]})
    samples = max(len(rows), 1)
    model.train(training)
    return {
        "samples": len(rows),
        **{key: value / samples for key, value in sums.items()},
        "examples": examples,
    }


def mean_pool_memory(memory: torch.Tensor, memory_padding_mask: torch.Tensor) -> torch.Tensor:
    mask = ~memory_padding_mask
    denom = mask.sum(dim=1).clamp_min(1).unsqueeze(1)
    return (memory * mask.unsqueeze(2)).sum(dim=1) / denom


def sampled_item_softmax_loss(
    model: TIGER,
    memory: torch.Tensor,
    memory_padding_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    catalog_token_tensor: torch.Tensor,
    *,
    negatives: int,
    temperature: float,
) -> torch.Tensor:
    batch_size = target_tokens.shape[0]
    negative_indices = torch.randint(
        0,
        catalog_token_tensor.shape[0],
        (batch_size, negatives),
        device=target_tokens.device,
    )
    negative_tokens = catalog_token_tensor.to(target_tokens.device)[negative_indices]
    candidate_tokens = torch.cat([target_tokens.unsqueeze(1), negative_tokens], dim=1)
    history_vectors = mean_pool_memory(memory, memory_padding_mask)
    candidate_vectors = model.item_embedding(candidate_tokens)
    logits = torch.einsum("bd,bnd->bn", history_vectors, candidate_vectors) / temperature
    labels = torch.zeros(batch_size, dtype=torch.long, device=target_tokens.device)
    return nn.functional.cross_entropy(logits, labels)


def save_checkpoint(
    model: TIGER,
    optimizer: torch.optim.Optimizer,
    config: TIGERConfig,
    checkpoint_dir: Path,
    step: int,
) -> Path:
    path = checkpoint_dir / f"checkpoint-{step}" / "model.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        path,
    )
    return path


def saved_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("checkpoint-*/model.pt"):
        try:
            step = int(path.parent.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, path))
    return sorted(checkpoints)


def load_checkpoint(
    model: TIGER,
    optimizer: torch.optim.Optimizer,
    config: TIGERConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> int:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TIGER resume checkpoint is missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if normalize_config_payload(payload.get("config", {})) != asdict(config):
        raise ValueError(f"TIGER resume checkpoint config mismatch: {checkpoint_path}")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["step"])


def report_saved_checkpoints(args: argparse.Namespace) -> dict[str, Any]:
    sft_root = run_root(args.artifact_root, args.campaign_id, args.sft_run_id)
    output_root = run_root(args.artifact_root, args.campaign_id, args.run_id)
    checkpoint_run_id = args.checkpoint_run_id or args.run_id
    checkpoint_root = run_root(args.artifact_root, args.campaign_id, checkpoint_run_id)
    eval_path = sft_root / "datasets" / "eval" / "history_next_eval.jsonl"
    train_path = sft_root / "datasets" / "llm_sft" / "history_next_train.jsonl"
    catalog_path = sft_root / "datasets" / "eval" / "catalog_semantic_ids.parquet"
    checkpoint_dir = checkpoint_root / "tiger" / "checkpoints"
    for path in (eval_path, train_path, catalog_path, checkpoint_dir):
        if not path.exists():
            raise FileNotFoundError(f"TIGER saved-checkpoint report input is missing: {path}")

    checkpoints = saved_checkpoints(checkpoint_dir)
    if args.report_checkpoint_step:
        checkpoints = [(step, path) for step, path in checkpoints if step == args.report_checkpoint_step]
    if not checkpoints:
        raise FileNotFoundError(f"TIGER run has no saved model checkpoints: {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_payload = torch.load(checkpoints[0][1], map_location=device, weights_only=False)
    config = TIGERConfig(**normalize_config_payload(first_payload["config"]))
    semantic_ids, tokens_to_sid = read_catalog(catalog_path)
    catalog_tokens = set(tokens_to_sid)
    target_tokens = len(next(iter(catalog_tokens)))
    if target_tokens != config.target_tokens or config.max_history_tokens % target_tokens:
        raise ValueError("TIGER checkpoint config does not match catalog semantic-ID shape")
    eval_rows = history_rows(
        eval_path,
        catalog_tokens,
        max_history_items=config.max_history_tokens // target_tokens,
        sid_length=target_tokens,
        limit=args.max_eval_examples,
    )
    train_rows = history_rows(
        train_path,
        catalog_tokens,
        max_history_items=config.max_history_tokens // target_tokens,
        sid_length=target_tokens,
        limit=args.max_train_examples,
    )
    if not eval_rows:
        raise ValueError("TIGER report requires non-empty shared held-out history-next examples")
    popularity_by_tokens = target_popularity(train_rows)
    trie = trie_for_catalog(catalog_tokens)

    rows: list[dict[str, Any]] = []
    for step, checkpoint_path in checkpoints:
        LOGGER.info(
            "evaluating checkpoint=%s beam_size=%s rerank_prefix_bonus=%.4f rerank_prefix_depth=%s rerank_recency_mode=%s rerank_popularity_penalty=%.4f",
            step,
            args.beam_size,
            args.rerank_prefix_bonus,
            args.rerank_prefix_depth,
            args.rerank_recency_mode,
            args.rerank_popularity_penalty,
        )
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if normalize_config_payload(payload.get("config", {})) != asdict(config):
            raise ValueError(f"TIGER checkpoint config mismatch in report set: {checkpoint_path}")
        model = TIGER(config).to(device)
        model.load_state_dict(payload["model_state_dict"])
        metrics = evaluate(
            model,
            eval_rows,
            tokens_to_sid,
            trie,
            device,
            target_tokens=target_tokens,
            beam_size=args.beam_size,
            rerank_prefix_bonus=args.rerank_prefix_bonus,
            rerank_prefix_depth=args.rerank_prefix_depth,
            rerank_recency_mode=args.rerank_recency_mode,
            rerank_popularity_penalty=args.rerank_popularity_penalty,
            popularity_by_tokens=popularity_by_tokens,
        )
        rows.append(
            {
                "checkpoint": str(checkpoint_path.parent),
                "step": step,
                "metrics": {key: value for key, value in metrics.items() if key != "examples"},
                "examples": metrics["examples"],
            }
        )
        LOGGER.info(
            "evaluated checkpoint=%s recall@5=%.4f ndcg@5=%.4f recall@10=%.4f ndcg@10=%.4f",
            step,
            metrics["recall_at_5"],
            metrics["ndcg_at_5"],
            metrics["recall_at_10"],
            metrics["ndcg_at_10"],
        )

    report_path = output_root / "tiger" / "checkpoint_eval_report.json"
    report = {
        "campaign_id": args.campaign_id,
        "dataset_id": args.dataset_id,
        "run_id": args.run_id,
        "checkpoint_run_id": checkpoint_run_id,
        "sft_run_id": args.sft_run_id,
        "stage": "tiger_saved_checkpoint_eval",
        "primary_metric_scope": "full_catalog",
        "rerank": {
            "beam_size": args.beam_size,
            "prefix_bonus": args.rerank_prefix_bonus,
            "prefix_depth": args.rerank_prefix_depth,
            "recency_mode": args.rerank_recency_mode,
            "popularity_penalty": args.rerank_popularity_penalty,
        },
        "inputs": {
            "catalog_semantic_ids": str(catalog_path),
            "eval_jsonl": str(eval_path),
        },
        "counts": {
            "catalog_semantic_ids": len(semantic_ids),
            "checkpoints": len(rows),
            "eval_history_next_examples": len(eval_rows),
            "train_history_next_examples": len(train_rows),
        },
        "evaluation_scope": "catalog-constrained generative top-10 over all catalog semantic IDs",
        "rows": rows,
        "artifacts": {"report": str(report_path)},
        "evaluated_at": utc_now(),
    }
    write_json(report_path, report)
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sft_root = run_root(args.artifact_root, args.campaign_id, args.sft_run_id)
    output_root = run_root(args.artifact_root, args.campaign_id, args.run_id)
    train_path = sft_root / "datasets" / "llm_sft" / "history_next_train.jsonl"
    eval_path = sft_root / "datasets" / "eval" / "history_next_eval.jsonl"
    catalog_path = sft_root / "datasets" / "eval" / "catalog_semantic_ids.parquet"
    for path in (train_path, eval_path, catalog_path):
        if not path.exists():
            raise FileNotFoundError(f"TIGER input artifact is missing: {path}")

    semantic_ids, tokens_to_sid = read_catalog(catalog_path)
    catalog_item_tokens = list(tokens_to_sid)
    item_index_by_tokens = {tokens: index for index, tokens in enumerate(catalog_item_tokens)}
    catalog_tokens = set(tokens_to_sid)
    target_tokens = len(next(iter(catalog_tokens)))
    train_rows = history_rows(
        train_path,
        catalog_tokens,
        max_history_items=args.max_history_items,
        sid_length=target_tokens,
        limit=args.max_train_examples,
    )
    eval_rows = history_rows(
        eval_path,
        catalog_tokens,
        max_history_items=args.max_history_items,
        sid_length=target_tokens,
        limit=args.max_eval_examples,
    )
    if not train_rows or not eval_rows:
        raise ValueError("TIGER requires non-empty shared history-next train and eval examples")
    popularity_by_tokens = target_popularity(train_rows)

    max_sid_token = max(token for token_tuple in catalog_tokens for token in token_tuple)
    config = TIGERConfig(
        vocab_size=max_sid_token + 1,
        max_history_tokens=args.max_history_items * target_tokens,
        target_tokens=target_tokens,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        feedforward_size=args.feedforward_size,
        dropout=args.dropout,
        activation=args.activation,
        norm_first=args.norm_first,
        use_sid_position_embeddings=args.use_sid_position_embeddings,
        use_input_norm=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    model = TIGER(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        HistoryNextDataset(train_rows, item_index_by_tokens),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    catalog_token_tensor = torch.tensor(catalog_item_tokens, dtype=torch.long, device=device)
    trie = trie_for_catalog(catalog_tokens)
    checkpoint_dir = output_root / "tiger" / "checkpoints"
    metrics_path = output_root / "tiger" / "checkpoint_metrics.jsonl"
    config_path = output_root / "tiger" / "config.json"
    write_json(
        config_path,
        {
            "model": asdict(config),
            "train": {
                "batch_size": args.batch_size,
                "beam_size": args.beam_size,
                "rerank_prefix_bonus": args.rerank_prefix_bonus,
                "rerank_prefix_depth": args.rerank_prefix_depth,
                "rerank_recency_mode": args.rerank_recency_mode,
                "rerank_popularity_penalty": args.rerank_popularity_penalty,
                "checkpoint_steps": args.checkpoint_steps,
                "learning_rate": args.learning_rate,
                "activation": config.activation,
                "norm_first": config.norm_first,
                "label_smoothing": args.label_smoothing,
                "item_loss_negatives": args.item_loss_negatives,
                "item_loss_temperature": args.item_loss_temperature,
                "item_loss_weight": args.item_loss_weight,
                "max_steps": args.max_steps,
                "max_minutes": args.max_minutes,
                "optimizer": "AdamW",
                "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else "",
                "seed": args.seed,
                "use_sid_position_embeddings": args.use_sid_position_embeddings,
                "weight_decay": args.weight_decay,
            },
            "inputs": {
                "catalog_semantic_ids": str(catalog_path),
                "eval_jsonl": str(eval_path),
                "train_jsonl": str(train_path),
            },
            "counts": {
                "catalog_semantic_ids": len(semantic_ids),
                "eval_history_next_examples": len(eval_rows),
                "train_history_next_examples": len(train_rows),
            },
            "evaluation_scope": "catalog-constrained generative top-10 over all catalog semantic IDs",
        },
    )

    losses: list[float] = []
    ce_losses: list[float] = []
    item_losses: list[float] = []
    step = 0
    last_checkpoint_step = 0
    started_monotonic = time.monotonic()
    deadline = started_monotonic + args.max_minutes * 60 if args.max_minutes > 0 else None
    stop_reason = "max_steps"

    def write_checkpoint_metrics() -> None:
        nonlocal last_checkpoint_step
        if step == 0 or step == last_checkpoint_step:
            return
        checkpoint_path = save_checkpoint(model, optimizer, config, checkpoint_dir, step)
        metrics = evaluate(
            model,
            eval_rows,
            tokens_to_sid,
            trie,
            device,
            target_tokens=target_tokens,
            beam_size=args.beam_size,
            rerank_prefix_bonus=args.rerank_prefix_bonus,
            rerank_prefix_depth=args.rerank_prefix_depth,
            rerank_recency_mode=args.rerank_recency_mode,
            rerank_popularity_penalty=args.rerank_popularity_penalty,
            popularity_by_tokens=popularity_by_tokens,
        )
        loss_window = losses[-max(args.checkpoint_steps, 1) :] if losses else [float("nan")]
        ce_window = ce_losses[-max(args.checkpoint_steps, 1) :] if ce_losses else []
        item_window = item_losses[-max(args.checkpoint_steps, 1) :] if item_losses else []
        append_jsonl(
            metrics_path,
            {
                "checkpoint": str(checkpoint_path.parent),
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "metrics": {key: value for key, value in metrics.items() if key != "examples"},
                "examples": metrics["examples"],
                "step": step,
                "stop_reason": stop_reason,
                "token_ce_loss": sum(ce_window) / len(ce_window) if ce_window else None,
                "item_loss": sum(item_window) / len(item_window) if item_window else None,
                "train_loss": sum(loss_window) / len(loss_window),
            },
        )
        LOGGER.info(
            "checkpoint=%s recall@5=%.4f ndcg@5=%.4f recall@10=%.4f ndcg@10=%.4f",
            step,
            metrics["recall_at_5"],
            metrics["ndcg_at_5"],
            metrics["recall_at_10"],
            metrics["ndcg_at_10"],
        )
        last_checkpoint_step = step
        model.train()

    if args.resume_checkpoint:
        step = load_checkpoint(model, optimizer, config, args.resume_checkpoint, device)
        LOGGER.info("resumed from checkpoint=%s step=%s", args.resume_checkpoint, step)
    if step >= args.max_steps:
        raise ValueError(f"TIGER resume step {step} is already at or after max steps {args.max_steps}")
    model.train()
    while step < args.max_steps:
        for histories, decoder_inputs, labels, _target_indices in train_loader:
            if deadline is not None and time.monotonic() >= deadline:
                stop_reason = "max_minutes"
                break
            histories = histories.to(device)
            decoder_inputs = decoder_inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            memory, memory_padding_mask = model.encode(histories)
            logits = model.decode(decoder_inputs, memory, memory_padding_mask)
            token_ce_loss = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                label_smoothing=args.label_smoothing,
            )
            loss = token_ce_loss
            item_loss = None
            if args.item_loss_weight > 0:
                item_loss = sampled_item_softmax_loss(
                    model,
                    memory,
                    memory_padding_mask,
                    labels[:, :target_tokens],
                    catalog_token_tensor,
                    negatives=args.item_loss_negatives,
                    temperature=args.item_loss_temperature,
                )
                loss = loss + args.item_loss_weight * item_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            step += 1
            losses.append(float(loss.item()))
            ce_losses.append(float(token_ce_loss.item()))
            if item_loss is not None:
                item_losses.append(float(item_loss.item()))
            if step == 1 or step % args.log_steps == 0:
                LOGGER.info("step=%s loss=%.6f", step, sum(losses[-args.log_steps :]) / len(losses[-args.log_steps :]))
            if (args.checkpoint_steps > 0 and step % args.checkpoint_steps == 0) or step == args.max_steps:
                write_checkpoint_metrics()
            if step >= args.max_steps:
                break
        if stop_reason == "max_minutes":
            break
    write_checkpoint_metrics()

    elapsed_seconds = time.monotonic() - started_monotonic
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if device.type == "cuda"
        else None
    )
    manifest = {
        "campaign_id": args.campaign_id,
        "dataset_id": args.dataset_id,
        "run_id": args.run_id,
        "parent_run_id": args.sft_run_id,
        "stage": "tiger_train",
        "artifacts": {
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_metrics": str(metrics_path),
            "config": str(config_path),
        },
        "elapsed_seconds": elapsed_seconds,
        "gpu_name": gpu_name,
        "max_minutes": args.max_minutes,
        "peak_vram_gb": peak_vram_gb,
        "steps_per_second": step / elapsed_seconds if elapsed_seconds > 0 else None,
        "steps_completed": step,
        "stop_reason": stop_reason,
    }
    write_json(output_root / "manifest.json", manifest)
    return {
        **manifest,
        "device": str(device),
        "finished_at": utc_now(),
        "counts": {
            "catalog_semantic_ids": len(semantic_ids),
            "eval_history_next_examples": len(eval_rows),
            "train_history_next_examples": len(train_rows),
        },
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--artifact-root", type=Path, required=True)
    root.add_argument("--campaign-id", required=True)
    root.add_argument("--dataset-id", required=True)
    root.add_argument("--sft-run-id", required=True)
    root.add_argument("--run-id", required=True)
    root.add_argument("--checkpoint-run-id", default="")
    root.add_argument("--resume-checkpoint", type=Path)
    root.add_argument("--max-history-items", type=int, default=20)
    root.add_argument("--hidden-size", type=int, default=128)
    root.add_argument("--layers", type=int, default=2)
    root.add_argument("--heads", type=int, default=4)
    root.add_argument("--feedforward-size", type=int, default=1024)
    root.add_argument("--dropout", type=float, default=0.1)
    root.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    root.add_argument("--norm-first", action=argparse.BooleanOptionalAction, default=True)
    root.add_argument("--max-steps", type=int, default=400)
    root.add_argument("--checkpoint-steps", type=int, default=100)
    root.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="Stop training after this many wall-clock minutes, then save/evaluate the current model.",
    )
    root.add_argument("--batch-size", type=int, default=256)
    root.add_argument("--beam-size", type=int, default=10)
    root.add_argument("--rerank-prefix-bonus", type=float, default=0.0)
    root.add_argument("--rerank-prefix-depth", type=int, default=0)
    root.add_argument("--rerank-recency-mode", choices=["inverse", "uniform", "exp"], default="inverse")
    root.add_argument("--rerank-popularity-penalty", type=float, default=0.0)
    root.add_argument("--learning-rate", type=float, default=0.001)
    root.add_argument("--label-smoothing", type=float, default=0.0)
    root.add_argument("--item-loss-weight", type=float, default=0.0)
    root.add_argument("--item-loss-negatives", type=int, default=256)
    root.add_argument("--item-loss-temperature", type=float, default=0.1)
    root.add_argument("--weight-decay", type=float, default=0.00001)
    root.add_argument("--max-grad-norm", type=float, default=1.0)
    root.add_argument("--max-train-examples", type=int, default=0)
    root.add_argument("--max-eval-examples", type=int, default=0)
    root.add_argument("--workers", type=int, default=2)
    root.add_argument("--log-steps", type=int, default=25)
    root.add_argument("--seed", type=int, default=42)
    root.add_argument(
        "--use-sid-position-embeddings",
        action="store_true",
        help="Add learned within-semantic-ID position embeddings to history and target tokens.",
    )
    root.add_argument(
        "--report-saved-checkpoints",
        action="store_true",
        help="Evaluate every saved checkpoint against the shared held-out split without training.",
    )
    root.add_argument("--report-checkpoint-step", type=int, default=0)
    return root


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args()
    result = report_saved_checkpoints(args) if args.report_saved_checkpoints else train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
