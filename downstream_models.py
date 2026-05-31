#!/usr/bin/env python3
"""Modal-native downstream next-listing baselines over RQ-VAE semantic IDs."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ITEM_ID_HINTS = ("listing_id", "listingid", "property_id", "propertyid", "home_id", "homeid", "item_id", "itemid", "listing", "property")
ID_HINTS = ITEM_ID_HINTS
USER_HINTS = ("user_id", "session_id", "visitor_id", "anonymous_id", "client_id", "device_id", "ip_hash")
TIME_HINTS = ("created_at", "viewed_at", "timestamp", "event_time", "time", "date", "updated_at")
PAD = 0
MIN_DIRECT_MATCH_ROWS = 1


@dataclass
class DownstreamConfig:
    campaign_id: str
    dataset_id: str
    rqvae_run_id: str
    rqvae_ablation: str
    downstream_dataset_id: str
    model: str
    run_id: str
    output_root: Path
    max_seconds: int = 300
    max_history: int = 20
    max_items: int = 50000
    min_sequence_len: int = 3
    seed: int = 17
    batch_size: int = 256
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    learning_rate: float = 1e-3
    eval_limit: int = 5000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def normalize_id(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def infer_column(columns: list[str], hints: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for hint in hints:
        if hint in lowered:
            return lowered[hint]
    normalized = {normalize_name(column): column for column in columns}
    for hint in hints:
        norm_hint = normalize_name(hint)
        if norm_hint in normalized:
            return normalized[norm_hint]
    for hint in hints:
        for column in columns:
            if normalize_name(hint) in normalize_name(column):
                return column
    return None


def id_like_columns(columns: list[str], hints: tuple[str, ...] = ID_HINTS) -> list[str]:
    selected: list[str] = []
    for column in columns:
        norm = normalize_name(column)
        if any(normalize_name(hint) in norm for hint in hints) or norm.endswith("id") or "id" in norm:
            selected.append(column)
    return selected


def item_id_like_columns(columns: list[str]) -> list[str]:
    selected: list[str] = []
    excluded = tuple(normalize_name(hint) for hint in USER_HINTS) + ("correlationid", "recommendationid")
    for column in columns:
        norm = normalize_name(column)
        if any(exclusion in norm for exclusion in excluded):
            continue
        if any(normalize_name(hint) in norm for hint in ITEM_ID_HINTS):
            selected.append(column)
    return selected


def column_match_report(frame: pd.DataFrame, columns: list[str], valid_items: set[str], max_values: int = 5) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for column in columns:
        normalized = frame[column].map(normalize_id)
        non_null = normalized.dropna()
        matched = normalized.isin(valid_items)
        matched_values = non_null[non_null.isin(valid_items)].drop_duplicates().head(max_values).tolist()
        sample_values = non_null.drop_duplicates().head(max_values).tolist()
        report.append(
            {
                "column": column,
                "non_null_rows": int(non_null.shape[0]),
                "unique_values": int(non_null.nunique()),
                "matched_rows": int(matched.sum()),
                "matched_unique_values": int(non_null[non_null.isin(valid_items)].nunique()),
                "sample_values": sample_values,
                "matched_sample_values": matched_values,
            }
        )
    return sorted(report, key=lambda row: (row["matched_rows"], row["matched_unique_values"], row["non_null_rows"]), reverse=True)


def normalize_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_id)


def select_direct_item_column(views: pd.DataFrame, valid_items: set[str]) -> tuple[str | None, list[dict[str, Any]]]:
    candidates = item_id_like_columns(list(views.columns))
    inferred = infer_column(list(views.columns), ID_HINTS)
    if inferred and inferred not in candidates:
        candidates.insert(0, inferred)
    report = column_match_report(views, candidates, valid_items)
    best = report[0] if report else None
    if best and best["matched_rows"] >= MIN_DIRECT_MATCH_ROWS:
        return str(best["column"]), report
    return None, report


def bridge_views_to_semantic_ids(
    views: pd.DataFrame,
    raw_listings: Path,
    valid_items: set[str],
) -> tuple[pd.Series | None, dict[str, Any]]:
    if not raw_listings.exists():
        return None, {"status": "missing_listings_artifact", "listings_path": str(raw_listings)}

    listings = pd.read_csv(raw_listings, low_memory=False)
    view_columns = item_id_like_columns(list(views.columns))
    listing_columns = id_like_columns(list(listings.columns))
    if not view_columns or not listing_columns:
        return None, {
            "status": "no_id_like_columns",
            "view_id_columns": view_columns,
            "listing_id_columns": listing_columns,
        }

    listing_semantic_report = column_match_report(listings, listing_columns, valid_items)
    semantic_col = listing_semantic_report[0]["column"] if listing_semantic_report and listing_semantic_report[0]["matched_rows"] > 0 else None
    if semantic_col is None:
        return None, {
            "status": "no_listing_column_matches_semantic_ids",
            "view_id_columns": view_columns,
            "listing_id_columns": listing_columns,
            "listing_semantic_match_report": listing_semantic_report,
        }

    listings_norm = {column: normalize_series(listings[column]) for column in listing_columns}
    semantic_values = normalize_series(listings[semantic_col])
    best: dict[str, Any] | None = None
    best_mapped: pd.Series | None = None
    for view_col in view_columns:
        view_values = normalize_series(views[view_col])
        for listing_key_col in listing_columns:
            listing_key_values = listings_norm[listing_key_col]
            mapping_frame = pd.DataFrame({"key": listing_key_values, "semantic_id": semantic_values})
            mapping_frame = mapping_frame[mapping_frame["key"].notna() & mapping_frame["semantic_id"].isin(valid_items)]
            mapping_frame = mapping_frame.drop_duplicates("key", keep="first")
            if mapping_frame.empty:
                continue
            mapping = dict(zip(mapping_frame["key"], mapping_frame["semantic_id"]))
            mapped = view_values.map(mapping)
            matched_rows = int(mapped.notna().sum())
            if matched_rows <= 0:
                continue
            candidate = {
                "status": "ok",
                "view_column": view_col,
                "listing_key_column": listing_key_col,
                "listing_semantic_column": semantic_col,
                "mapping_key_count": int(len(mapping)),
                "mapped_view_rows": matched_rows,
                "mapped_unique_semantic_ids": int(mapped.dropna().nunique()),
            }
            if best is None or (candidate["mapped_view_rows"], candidate["mapped_unique_semantic_ids"]) > (
                best["mapped_view_rows"],
                best["mapped_unique_semantic_ids"],
            ):
                best = candidate
                best_mapped = mapped

    if best is None or best_mapped is None or best["mapped_view_rows"] <= 0:
        return None, {
            "status": "no_view_to_listing_bridge",
            "view_id_columns": view_columns,
            "listing_id_columns": listing_columns,
            "listing_semantic_match_report": listing_semantic_report,
        }
    best["listing_semantic_match_report"] = listing_semantic_report[:10]
    return best_mapped, best


def sid_tokens(row: pd.Series) -> list[str]:
    sid_cols = sorted([c for c in row.index if c.startswith("sid_") and c != "sid_unique_suffix"])
    tokens = [f"L{i}_{int(row[col])}" for i, col in enumerate(sid_cols)]
    tokens.append(f"S_{int(row['sid_unique_suffix'])}")
    return tokens


def artifact_paths(config: DownstreamConfig) -> dict[str, Path]:
    root = config.output_root / config.campaign_id
    return {
        "dataset_root": root / "datasets" / config.dataset_id,
        "downstream_root": root / "datasets" / config.dataset_id / "downstream" / config.downstream_dataset_id,
        "rqvae_root": root / "runs" / config.rqvae_run_id,
        "run_root": root / "runs" / config.run_id,
    }


def build_or_load_dataset(config: DownstreamConfig) -> dict[str, Any]:
    paths = artifact_paths(config)
    downstream_root = paths["downstream_root"]
    manifest_path = downstream_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stats = manifest.get("stats", {})
        if int(stats.get("test_examples", 0)) <= 0 or int(stats.get("item_count", 0)) <= 0:
            raise RuntimeError(f"Existing downstream dataset manifest is unusable: {manifest_path} stats={stats}")
        return manifest

    raw_views = paths["dataset_root"] / "raw" / "listing_views.csv"
    raw_listings = paths["dataset_root"] / "raw" / "listings.csv"
    semantic_path = paths["rqvae_root"] / "semantic_ids" / "listing_semantic_ids.parquet"
    if not raw_views.exists():
        raise FileNotFoundError(f"Missing raw listing views artifact: {raw_views}")
    if not semantic_path.exists():
        raise FileNotFoundError(f"Missing semantic ID artifact: {semantic_path}")

    semantic = pd.read_parquet(semantic_path)
    semantic["listing_id"] = semantic["listing_id"].map(normalize_id)
    semantic = semantic[semantic["listing_id"].notna()].copy()
    semantic = semantic.drop_duplicates("listing_id", keep="first")
    semantic_lookup = {row["listing_id"]: sid_tokens(row) for _, row in semantic.iterrows()}
    valid_items = set(semantic_lookup)

    views = pd.read_csv(raw_views, low_memory=False)
    source_view_rows = int(len(views))
    columns = list(views.columns)
    item_col, direct_match_report = select_direct_item_column(views, valid_items)
    bridge_report: dict[str, Any] | None = None
    group_col = infer_column(columns, USER_HINTS)
    time_col = infer_column(columns, TIME_HINTS)
    if item_col is None:
        mapped_ids, bridge_report = bridge_views_to_semantic_ids(views, raw_listings, valid_items)
        if mapped_ids is None:
            diagnostics = {
                "listing_views_columns": columns,
                "semantic_item_count": len(valid_items),
                "direct_match_report": direct_match_report,
                "bridge_report": bridge_report,
            }
            raise RuntimeError(
                "No listing view rows matched semantic ID assignment artifacts. "
                f"Diagnostics: {json.dumps(diagnostics, sort_keys=True, default=str)[:3500]}"
            )
        views["_listing_id"] = mapped_ids
        item_col = str(bridge_report.get("view_column", "__bridged__")) if bridge_report else "__bridged__"
        item_join_strategy = "listings_bridge"
    else:
        views["_listing_id"] = views[item_col].map(normalize_id)
        item_join_strategy = "direct"
    if group_col is None:
        group_col = "__single_sequence__"
        views[group_col] = "all"
    if time_col is None:
        views["__order__"] = np.arange(len(views))
        time_col = "__order__"

    views = views[views["_listing_id"].notna()].copy()
    views = views[views["_listing_id"].isin(valid_items)].copy()
    if views.empty:
        diagnostics = {
            "listing_views_columns": columns,
            "semantic_item_count": len(valid_items),
            "selected_item_column": item_col,
            "item_join_strategy": item_join_strategy,
            "direct_match_report": direct_match_report,
            "bridge_report": bridge_report,
        }
        raise RuntimeError(
            "No listing view rows matched semantic ID assignment artifacts after item mapping. "
            f"Diagnostics: {json.dumps(diagnostics, sort_keys=True, default=str)[:3500]}"
        )
    if time_col != "__order__":
        views["_time"] = pd.to_datetime(views[time_col], errors="coerce")
        views["_time"] = views["_time"].fillna(pd.Timestamp("1970-01-01"))
    else:
        views["_time"] = views[time_col]
    views["_group"] = views[group_col].fillna("__missing__").astype(str)
    views = views.sort_values(["_group", "_time"])

    raw_sequences: list[list[str]] = []
    candidate_sequence_lengths: list[int] = []
    for _, group in views.groupby("_group", sort=False):
        seq = []
        prev = None
        for item in group["_listing_id"].tolist():
            if item != prev:
                seq.append(item)
            prev = item
        candidate_sequence_lengths.append(len(seq))
        if len(seq) >= config.min_sequence_len:
            raw_sequences.append(seq)

    item_counts = Counter(item for seq in raw_sequences for item in seq)
    kept_items = {item for item, _ in item_counts.most_common(config.max_items)}
    item_to_idx = {item: idx + 1 for idx, item in enumerate(sorted(kept_items))}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    examples: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    retained_sequences = 0
    for sequence_id, seq in enumerate(raw_sequences):
        seq = [item for item in seq if item in item_to_idx]
        if len(seq) < config.min_sequence_len:
            continue
        retained_sequences += 1
        indexed = [item_to_idx[item] for item in seq]
        test_pos = len(indexed) - 1
        val_pos = len(indexed) - 2
        for pos in range(1, len(indexed)):
            split = "test" if pos == test_pos else "val" if pos == val_pos else "train"
            history = indexed[max(0, pos - config.max_history) : pos]
            target = indexed[pos]
            examples[split].append(
                {
                    "sequence_id": sequence_id,
                    "history": history,
                    "target": target,
                    "target_listing_id": idx_to_item[target],
                    "history_listing_ids": [idx_to_item[i] for i in history],
                }
            )

    downstream_root.mkdir(parents=True, exist_ok=True)
    for split, rows in examples.items():
        pd.DataFrame(rows).to_json(downstream_root / f"{split}.jsonl", orient="records", lines=True)

    token_to_idx = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "<item_sep>": 4}
    item_semantic_tokens: dict[str, list[str]] = {}
    for item in sorted(kept_items):
        tokens = semantic_lookup[item]
        item_semantic_tokens[item] = tokens
        for token in tokens:
            if token not in token_to_idx:
                token_to_idx[token] = len(token_to_idx)
    idx_to_semantic_target = {
        str(item_to_idx[item]): [token_to_idx[t] for t in item_semantic_tokens[item]] + [token_to_idx["<eos>"]]
        for item in kept_items
    }

    vocab = {
        "item_to_idx": item_to_idx,
        "idx_to_item": {str(k): v for k, v in idx_to_item.items()},
        "token_to_idx": token_to_idx,
        "item_semantic_tokens": item_semantic_tokens,
        "idx_to_semantic_target": idx_to_semantic_target,
    }
    write_json(downstream_root / "vocab.json", vocab)

    sequence_lengths = [len(seq) for seq in raw_sequences]
    manifest = {
        "campaign_id": config.campaign_id,
        "dataset_id": config.dataset_id,
        "rqvae_run_id": config.rqvae_run_id,
        "rqvae_ablation": config.rqvae_ablation,
        "downstream_dataset_id": config.downstream_dataset_id,
        "created_at": utc_now(),
        "source_artifacts": {
            "listing_views": str(raw_views),
            "listings": str(raw_listings),
            "semantic_ids": str(semantic_path),
        },
        "columns": {
            "item": item_col,
            "group": group_col,
            "time": time_col,
            "item_join_strategy": item_join_strategy,
            "direct_item_match_report": direct_match_report[:10],
            "bridge_report": bridge_report,
        },
        "filters": {
            "max_history": config.max_history,
            "max_items": config.max_items,
            "min_sequence_len": config.min_sequence_len,
        },
        "stats": {
            "source_view_rows": source_view_rows,
            "matched_view_rows": int(len(views)),
            "matched_group_count": int(views["_group"].nunique()),
            "semantic_item_count": int(len(valid_items)),
            "item_count": int(len(item_to_idx)),
            "sequence_count": int(retained_sequences),
            "train_examples": int(len(examples["train"])),
            "val_examples": int(len(examples["val"])),
            "test_examples": int(len(examples["test"])),
            "sequence_length_min": int(min(sequence_lengths)) if sequence_lengths else 0,
            "sequence_length_p50": float(np.median(sequence_lengths)) if sequence_lengths else 0.0,
            "sequence_length_p95": float(np.quantile(sequence_lengths, 0.95)) if sequence_lengths else 0.0,
            "sequence_length_max": int(max(sequence_lengths)) if sequence_lengths else 0,
            "candidate_sequence_length_min": int(min(candidate_sequence_lengths)) if candidate_sequence_lengths else 0,
            "candidate_sequence_length_p50": float(np.median(candidate_sequence_lengths)) if candidate_sequence_lengths else 0.0,
            "candidate_sequence_length_p95": float(np.quantile(candidate_sequence_lengths, 0.95)) if candidate_sequence_lengths else 0.0,
            "candidate_sequence_length_max": int(max(candidate_sequence_lengths)) if candidate_sequence_lengths else 0,
            "dropped_invalid_view_rows": int(max(0, source_view_rows - len(views))),
        },
        "artifacts": {
            "train": str(downstream_root / "train.jsonl"),
            "val": str(downstream_root / "val.jsonl"),
            "test": str(downstream_root / "test.jsonl"),
            "vocab": str(downstream_root / "vocab.json"),
            "manifest": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    if int(manifest["stats"]["test_examples"]) <= 0 or int(manifest["stats"]["item_count"]) <= 0:
        raise RuntimeError(f"Downstream dataset has no held-out examples: {json.dumps(manifest['stats'], sort_keys=True)}")
    return manifest


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SequenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], max_history: int) -> None:
        self.rows = rows
        self.max_history = max_history

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        history = row["history"][-self.max_history :]
        return {"history": history, "target": int(row["target"])}


def collate_sequence(batch: list[dict[str, Any]], max_history: int) -> dict[str, torch.Tensor]:
    histories = torch.zeros((len(batch), max_history), dtype=torch.long)
    targets = torch.tensor([row["target"] for row in batch], dtype=torch.long)
    for i, row in enumerate(batch):
        hist = row["history"][-max_history:]
        histories[i, -len(hist) :] = torch.tensor(hist, dtype=torch.long)
    return {"history": histories, "target": targets}


class GRURec(nn.Module):
    def __init__(self, item_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(item_count + 1, hidden_dim, padding_idx=PAD)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, item_count + 1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = self.embedding(history)
        _, hidden = self.gru(x)
        return self.output(hidden[-1])


class SASRecSmall(nn.Module):
    def __init__(self, item_count: int, hidden_dim: int, max_history: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(item_count + 1, hidden_dim, padding_idx=PAD)
        self.pos_embedding = nn.Embedding(max_history, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, item_count + 1)
        self.max_history = max_history

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(history.shape[1], device=history.device).unsqueeze(0)
        x = self.item_embedding(history) + self.pos_embedding(positions)
        mask = history.eq(PAD)
        causal = torch.triu(torch.ones(history.shape[1], history.shape[1], device=history.device), diagonal=1).bool()
        encoded = self.encoder(x, mask=causal, src_key_padding_mask=mask)
        lengths = history.ne(PAD).sum(dim=1).clamp(min=1) - 1
        last = encoded[torch.arange(history.shape[0], device=history.device), lengths]
        return self.output(last)


class T5Dataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], vocab: dict[str, Any], max_history: int) -> None:
        self.rows = rows
        self.max_history = max_history
        self.token_to_idx = vocab["token_to_idx"]
        self.idx_to_item = {int(k): v for k, v in vocab["idx_to_item"].items()}
        self.item_semantic_tokens = vocab["item_semantic_tokens"]
        self.idx_to_semantic_target = {int(k): v for k, v in vocab["idx_to_semantic_target"].items()}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        input_ids = [self.token_to_idx["<bos>"]]
        for item_idx in row["history"][-self.max_history :]:
            item = self.idx_to_item[int(item_idx)]
            input_ids.extend(self.token_to_idx[t] for t in self.item_semantic_tokens[item])
            input_ids.append(self.token_to_idx["<item_sep>"])
        target = self.idx_to_semantic_target[int(row["target"])]
        return {"input_ids": input_ids, "labels": target, "target": int(row["target"]), "history": row["history"]}


def collate_t5(batch: list[dict[str, Any]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    max_in = max(len(row["input_ids"]) for row in batch)
    max_out = max(len(row["labels"]) for row in batch)
    input_ids = torch.full((len(batch), max_in), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_in), dtype=torch.long)
    labels = torch.full((len(batch), max_out), -100, dtype=torch.long)
    targets = torch.tensor([row["target"] for row in batch], dtype=torch.long)
    histories = [row["history"] for row in batch]
    for i, row in enumerate(batch):
        input_ids[i, : len(row["input_ids"])] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[i, : len(row["input_ids"])] = 1
        labels[i, : len(row["labels"])] = torch.tensor(row["labels"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "target": targets, "histories": histories}


def ranking_metrics(rank_lists: list[list[int]], targets: list[int], histories: list[list[int]], item_count: int) -> dict[str, float]:
    recall5 = recall10 = ndcg5 = ndcg10 = mrr10 = 0.0
    covered: set[int] = set()
    total = len(targets)
    if total == 0:
        return {
            "test_recall_at_10": 0.0,
            "test_ndcg_at_10": 0.0,
            "test_recall_at_5": 0.0,
            "test_ndcg_at_5": 0.0,
            "test_mrr_at_10": 0.0,
            "coverage_at_10": 0.0,
        }
    for recs, target in zip(rank_lists, targets):
        top10 = recs[:10]
        covered.update(top10)
        if target in top10:
            rank = top10.index(target) + 1
            recall10 += 1
            ndcg10 += 1 / math.log2(rank + 1)
            mrr10 += 1 / rank
            if rank <= 5:
                recall5 += 1
                ndcg5 += 1 / math.log2(rank + 1)
    return {
        "test_recall_at_10": recall10 / total,
        "test_ndcg_at_10": ndcg10 / total,
        "test_recall_at_5": recall5 / total,
        "test_ndcg_at_5": ndcg5 / total,
        "test_mrr_at_10": mrr10 / total,
        "coverage_at_10": len(covered) / max(1, item_count),
    }


def evaluate_logits_model(model: nn.Module, rows: list[dict[str, Any]], config: DownstreamConfig, item_count: int) -> dict[str, float]:
    device = next(model.parameters()).device
    eval_rows = rows[: config.eval_limit]
    rank_lists: list[list[int]] = []
    targets: list[int] = []
    histories: list[list[int]] = []
    loader = DataLoader(
        SequenceDataset(eval_rows, config.max_history),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_sequence(b, config.max_history),
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            logits = model(history)
            logits[:, PAD] = -1e9
            for i in range(history.shape[0]):
                seen = set(int(x) for x in history[i].detach().cpu().tolist() if int(x) != PAD)
                if seen:
                    logits[i, list(seen)] = -1e9
            top = torch.topk(logits, k=min(10, item_count), dim=1).indices.cpu().tolist()
            rank_lists.extend([[int(x) for x in row] for row in top])
            targets.extend([int(x) for x in batch["target"].tolist()])
            histories.extend([[int(x) for x in h.tolist() if int(x) != PAD] for h in history.cpu()])
    return ranking_metrics(rank_lists, targets, histories, item_count)


def train_logits_model(model_name: str, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], item_count: int, config: DownstreamConfig) -> tuple[dict[str, Any], nn.Module]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "sasrec_small":
        model = SASRecSmall(item_count, config.hidden_dim, config.max_history, config.num_heads, config.num_layers)
    elif model_name == "gru4rec_small":
        model = GRURec(item_count, config.hidden_dim)
    else:
        raise ValueError(model_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(
        SequenceDataset(train_rows, config.max_history),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sequence(b, config.max_history),
        drop_last=False,
    )
    started = time.monotonic()
    deadline = started + max(20, config.max_seconds - 20)
    steps = 0
    last_loss = None
    while time.monotonic() < deadline:
        for batch in train_loader:
            history = batch["history"].to(device)
            target = batch["target"].to(device)
            logits = model(history)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            last_loss = float(loss.detach().cpu())
            if time.monotonic() >= deadline:
                break
    val_loss = None
    if val_rows:
        losses = []
        model.eval()
        loader = DataLoader(
            SequenceDataset(val_rows[: config.eval_limit], config.max_history),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_sequence(b, config.max_history),
        )
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["history"].to(device))
                losses.append(float(F.cross_entropy(logits, batch["target"].to(device)).cpu()))
        val_loss = float(np.mean(losses)) if losses else None
    metrics = evaluate_logits_model(model, test_rows, config, item_count)
    metrics.update(
        {
            "training_steps_completed": int(steps),
            "training_seconds": round(time.monotonic() - started, 3),
            "validation_loss": val_loss,
            "train_last_loss": last_loss,
            "device": str(device),
        }
    )
    return metrics, model


def popularity_metrics(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], config: DownstreamConfig, item_count: int) -> dict[str, Any]:
    started = time.monotonic()
    counts = Counter(int(row["target"]) for row in train_rows)
    popular = [item for item, _ in counts.most_common()]
    rank_lists = []
    targets = []
    histories = []
    for row in test_rows[: config.eval_limit]:
        seen = set(int(x) for x in row["history"])
        recs = [item for item in popular if item not in seen][:10]
        rank_lists.append(recs)
        targets.append(int(row["target"]))
        histories.append([int(x) for x in row["history"]])
    metrics = ranking_metrics(rank_lists, targets, histories, item_count)
    metrics.update(
        {
            "training_steps_completed": 0,
            "training_seconds": round(time.monotonic() - started, 3),
            "validation_loss": None,
            "train_last_loss": None,
            "device": "cpu",
        }
    )
    return metrics


def train_t5(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], vocab: dict[str, Any], config: DownstreamConfig) -> tuple[dict[str, Any], nn.Module]:
    from transformers import T5Config, T5ForConditionalGeneration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_to_idx = vocab["token_to_idx"]
    idx_to_target = {int(k): tuple(v) for k, v in vocab["idx_to_semantic_target"].items()}
    target_to_idx = {v: k for k, v in idx_to_target.items()}
    model_config = T5Config(
        vocab_size=len(token_to_idx),
        d_model=config.hidden_dim,
        d_ff=config.hidden_dim * 4,
        num_layers=config.num_layers,
        num_decoder_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout_rate=0.1,
        pad_token_id=token_to_idx["<pad>"],
        eos_token_id=token_to_idx["<eos>"],
        decoder_start_token_id=token_to_idx["<bos>"],
    )
    model = T5ForConditionalGeneration(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(
        T5Dataset(train_rows, vocab, config.max_history),
        batch_size=max(16, min(64, config.batch_size)),
        shuffle=True,
        collate_fn=collate_t5,
    )
    started = time.monotonic()
    deadline = started + max(20, config.max_seconds - 20)
    steps = 0
    last_loss = None
    while time.monotonic() < deadline:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            last_loss = float(loss.detach().cpu())
            if time.monotonic() >= deadline:
                break
    val_loss = None
    if val_rows:
        losses = []
        model.eval()
        loader = DataLoader(
            T5Dataset(val_rows[: config.eval_limit], vocab, config.max_history),
            batch_size=max(16, min(64, config.batch_size)),
            shuffle=False,
            collate_fn=collate_t5,
        )
        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}
                losses.append(float(model(**inputs).loss.cpu()))
        val_loss = float(np.mean(losses)) if losses else None

    eval_rows = test_rows[: config.eval_limit]
    rank_lists = []
    targets = []
    histories = []
    model.eval()
    loader = DataLoader(
        T5Dataset(eval_rows, vocab, config.max_history),
        batch_size=max(8, min(32, config.batch_size)),
        shuffle=False,
        collate_fn=collate_t5,
    )
    with torch.no_grad():
        for batch in loader:
            generated = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_length=8,
                num_beams=10,
                num_return_sequences=10,
                early_stopping=True,
            ).cpu().tolist()
            for i in range(len(batch["target"])):
                seqs = generated[i * 10 : (i + 1) * 10]
                recs = []
                seen = set(int(x) for x in batch["histories"][i])
                for seq in seqs:
                    cleaned = tuple(x for x in seq if x not in {token_to_idx["<pad>"], token_to_idx["<bos>"]})
                    if token_to_idx["<eos>"] in cleaned:
                        cleaned = cleaned[: cleaned.index(token_to_idx["<eos>"]) + 1]
                    item = target_to_idx.get(cleaned)
                    if item is not None and item not in seen and item not in recs:
                        recs.append(item)
                rank_lists.append(recs[:10])
                targets.append(int(batch["target"][i]))
                histories.append([int(x) for x in batch["histories"][i]])
    metrics = ranking_metrics(rank_lists, targets, histories, len(vocab["item_to_idx"]))
    metrics.update(
        {
            "training_steps_completed": int(steps),
            "training_seconds": round(time.monotonic() - started, 3),
            "validation_loss": val_loss,
            "train_last_loss": last_loss,
            "device": str(device),
            "notes": "T5 evaluation uses unconstrained beam generation mapped exactly to semantic ID tokens; preliminary if beams do not map to catalog IDs.",
        }
    )
    return metrics, model


def run_downstream(config: DownstreamConfig) -> dict[str, Any]:
    started_at = utc_now()
    paths = artifact_paths(config)
    run_root = paths["run_root"]
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        dataset_manifest = build_or_load_dataset(config)
        downstream_root = paths["downstream_root"]
        train_rows = load_jsonl(downstream_root / "train.jsonl")
        val_rows = load_jsonl(downstream_root / "val.jsonl")
        test_rows = load_jsonl(downstream_root / "test.jsonl")
        vocab = json.loads((downstream_root / "vocab.json").read_text(encoding="utf-8"))
        item_count = int(dataset_manifest["stats"]["item_count"])
        if config.model == "dataset_only":
            metrics = {
                "training_steps_completed": 0,
                "training_seconds": 0.0,
                "validation_loss": None,
                "test_recall_at_10": None,
                "test_ndcg_at_10": None,
                "test_recall_at_5": None,
                "test_ndcg_at_5": None,
                "test_mrr_at_10": None,
                "coverage_at_10": None,
                "notes": "Dataset-only materialization run.",
            }
            model_path = None
        elif config.model == "popularity_baseline":
            metrics = popularity_metrics(train_rows, test_rows, config, item_count)
            model_path = None
        elif config.model in {"sasrec_small", "gru4rec_small"}:
            metrics, model = train_logits_model(config.model, train_rows, val_rows, test_rows, item_count, config)
            model_path = run_root / "model.pt"
            torch.save(model.state_dict(), model_path)
        elif config.model == "t5_tiger_small":
            metrics, model = train_t5(train_rows, val_rows, test_rows, vocab, config)
            model_path = run_root / "model.pt"
            torch.save(model.state_dict(), model_path)
        else:
            raise ValueError(f"Unknown downstream model: {config.model}")

        model_config = asdict(config)
        model_config["output_root"] = str(config.output_root)
        result_metrics = {
            **metrics,
            "train_examples": int(dataset_manifest["stats"]["train_examples"]),
            "val_examples": int(dataset_manifest["stats"]["val_examples"]),
            "test_examples": int(dataset_manifest["stats"]["test_examples"]),
            "item_count": item_count,
            "sequence_count": int(dataset_manifest["stats"]["sequence_count"]),
        }
        write_json(run_root / "metrics.json", result_metrics)
        manifest = {
            "campaign_id": config.campaign_id,
            "dataset_id": config.dataset_id,
            "rqvae_run_id": config.rqvae_run_id,
            "rqvae_ablation": config.rqvae_ablation,
            "downstream_dataset_id": config.downstream_dataset_id,
            "run_id": config.run_id,
            "model": config.model,
            "model_config": model_config,
            "status": "ok",
            "decision": "keep",
            "started_at": started_at,
            "finished_at": utc_now(),
            "metrics": result_metrics,
            "artifacts": {
                "run_root": str(run_root),
                "dataset_manifest": dataset_manifest["artifacts"]["manifest"],
                "metrics": str(run_root / "metrics.json"),
                "model": str(model_path) if model_path else None,
            },
        }
        write_json(run_root / "manifest.json", manifest)
        return {
            "run_id": config.run_id,
            "dataset_id": config.dataset_id,
            "rqvae_run_id": config.rqvae_run_id,
            "rqvae_ablation": config.rqvae_ablation,
            "downstream_dataset_id": config.downstream_dataset_id,
            "model": config.model,
            "model_config": model_config,
            "status": "ok",
            "decision": "keep",
            "artifact_uri": str(run_root),
            "metrics": result_metrics,
        }
    except Exception as exc:
        manifest = {
            "campaign_id": config.campaign_id,
            "dataset_id": config.dataset_id,
            "rqvae_run_id": config.rqvae_run_id,
            "rqvae_ablation": config.rqvae_ablation,
            "downstream_dataset_id": config.downstream_dataset_id,
            "run_id": config.run_id,
            "model": config.model,
            "status": "crash",
            "decision": "crash",
            "started_at": started_at,
            "finished_at": utc_now(),
            "error": repr(exc),
        }
        write_json(run_root / "manifest.json", manifest)
        return {
            "run_id": config.run_id,
            "dataset_id": config.dataset_id,
            "rqvae_run_id": config.rqvae_run_id,
            "rqvae_ablation": config.rqvae_ablation,
            "downstream_dataset_id": config.downstream_dataset_id,
            "model": config.model,
            "model_config": asdict(config) | {"output_root": str(config.output_root)},
            "status": "crash",
            "decision": "crash",
            "artifact_uri": str(run_root),
            "metrics": {},
            "error": repr(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--rqvae-run-id", required=True)
    parser.add_argument("--rqvae-ablation", required=True)
    parser.add_argument("--downstream-dataset-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-items", type=int, default=50000)
    parser.add_argument("--eval-limit", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DownstreamConfig(
        campaign_id=args.campaign_id,
        dataset_id=args.dataset_id,
        rqvae_run_id=args.rqvae_run_id,
        rqvae_ablation=args.rqvae_ablation,
        downstream_dataset_id=args.downstream_dataset_id,
        model=args.model,
        run_id=args.run_id,
        output_root=args.output_root,
        max_seconds=args.max_seconds,
        max_history=args.max_history,
        max_items=args.max_items,
        eval_limit=args.eval_limit,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate,
    )
    result = run_downstream(config)
    print("RESULT_JSON\t" + json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
