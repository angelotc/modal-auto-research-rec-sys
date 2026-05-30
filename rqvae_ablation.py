#!/usr/bin/env python3
"""Modal-native RQ-VAE listing-vector ablation runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ABLATIONS = {
    "core_metadata",
    "structured_metadata",
    "context_metadata",
    "price_shape_metadata",
    "no_price",
    "no_location",
    "property_shape",
    "fees_and_price",
}
ID_CANDIDATES = ("id", "listing_id", "property_id", "home_id", "item_id")
GEO_RE = re.compile(r"(latitude|longitude|\blat\b|\blon\b|\blng\b|geo)", re.I)
LOCATION_RE = re.compile(
    r"(address|street|city|ward|prefecture|province|state|country|region|"
    r"neighbou?rhood|district|station|line|nearest|postal|zipcode|zip|location|"
    r"latitude|longitude|\blat\b|\blon\b|\blng\b|geo)",
    re.I,
)
PRICE_RE = re.compile(r"(price|rent|fee|cost|deposit|key_money|yen|amount)", re.I)
TEXT_RE = re.compile(r"(title|description|summary|remark|comment|body|url|image|photo|slug)", re.I)
DATE_RE = re.compile(r"(date|time|_at$|timestamp)", re.I)
SHAPE_RE = re.compile(
    r"(size|area|rooms|bedroom|building_age|year_built|month_built|"
    r"listing_type|structure_type|coverage|floor_area|land_area)",
    re.I,
)
CONTEXT_RE = re.compile(
    r"(title|description|summary|remark|comment|amenit|feature|layout|building|"
    r"property_type|type|address|city|ward|prefecture|station|line|neighbou?rhood)",
    re.I,
)


@dataclass
class RQVAETrainConfig:
    """Small but explicit RQ-VAE config for comparable Modal sweeps."""

    max_seconds: int = 300
    seed: int = 17
    rq_levels: int = 4
    codebook_size: int = 64
    commitment_beta: float = 0.25
    learning_rate: float = 2e-3
    min_learning_rate: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_hidden_dim: int = 256
    max_latent_dim: int = 64
    gradient_clip_norm: float = 1.0
    use_rotation_trick: bool = True
    use_kmeans_init: bool = False
    reset_unused_codes: bool = True
    reset_every_steps: int = 2000
    validation_size: int = 2048


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def maybe_init_wandb(*, run_id: str, ablation: str, config: dict[str, Any]) -> Any:
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "auto-rec-sys"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        id=run_id,
        name=run_id,
        resume="allow",
        config=config,
        tags=["rqvae", "ablation", ablation],
    )


def wandb_log(run: Any, metrics: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    scalars = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if scalars:
        run.log(scalars, step=step)


def find_id_column(frame: pd.DataFrame) -> str | None:
    lowered = {column.lower(): column for column in frame.columns}
    for candidate in ID_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def safe_feature_token(value: Any) -> str:
    text = str(value)
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")[:40] or "value"
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{slug}_{digest}"


def find_geo_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    columns = list(frame.columns)
    lat = next((c for c in columns if re.search(r"(latitude|\blat\b)", c, re.I)), None)
    lon = next((c for c in columns if re.search(r"(longitude|\blon\b|\blng\b)", c, re.I)), None)
    return lat, lon


def add_numeric_feature(parts: list[pd.Series], names: list[str], series: pd.Series, name: str) -> None:
    series = series.astype("float64")
    valid = series.notna()
    if valid.mean() < 0.2 or series[valid].nunique(dropna=True) <= 1:
        return
    filled = series.fillna(series.median())
    lo, hi = filled.quantile([0.01, 0.99])
    if math.isfinite(float(lo)) and math.isfinite(float(hi)) and lo < hi:
        filled = filled.clip(lo, hi)
    parts.append(filled.astype("float32"))
    names.append(name)
    if valid.mean() < 0.98:
        parts.append((~valid).astype("float32"))
        names.append(f"{name}__missing")


def geo_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    lat_col, lon_col = find_geo_columns(frame)
    if not lat_col or not lon_col:
        raise RuntimeError("geo_only requires latitude and longitude columns in listings.csv")
    lat = numeric_series(frame, lat_col)
    lon = numeric_series(frame, lon_col)
    mask = lat.notna() & lon.notna()
    lat = lat.fillna(lat[mask].median())
    lon = lon.fillna(lon[mask].median())
    lat_rad = np.radians(lat.astype("float64"))
    lon_rad = np.radians(lon.astype("float64"))
    features = pd.DataFrame(
        {
            "geo_lat": lat.astype("float32"),
            "geo_lon": lon.astype("float32"),
            "geo_lat_sin": np.sin(lat_rad).astype("float32"),
            "geo_lat_cos": np.cos(lat_rad).astype("float32"),
            "geo_lon_sin": np.sin(lon_rad).astype("float32"),
            "geo_lon_cos": np.cos(lon_rad).astype("float32"),
        },
        index=frame.index,
    )
    return features, mask, list(features.columns)


def is_core_numeric(column: str, *, allow_location: bool, allow_price: bool) -> bool:
    if TEXT_RE.search(column):
        return False
    if DATE_RE.search(column):
        return False
    if GEO_RE.search(column) or column.lower() == "geom":
        return False
    if not allow_location and LOCATION_RE.search(column):
        return False
    if not allow_price and PRICE_RE.search(column):
        return False
    return True


def core_features(
    frame: pd.DataFrame,
    *,
    allow_location: bool,
    allow_price: bool,
    include_pattern: re.Pattern[str] | None = None,
    exclude_pattern: re.Pattern[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    parts: list[pd.Series] = []
    names: list[str] = []
    id_col = find_id_column(frame)
    id_like = {candidate.lower() for candidate in ID_CANDIDATES}
    for column in frame.columns:
        if column.lower() in id_like:
            continue
        if include_pattern is not None and not include_pattern.search(column):
            continue
        if exclude_pattern is not None and exclude_pattern.search(column):
            continue
        if column == id_col or not is_core_numeric(column, allow_location=allow_location, allow_price=allow_price):
            continue
        series = numeric_series(frame, column)
        if series.notna().mean() >= 0.2:
            add_numeric_feature(parts, names, series, f"num__{column}")
            continue
        if LOCATION_RE.search(column) and not allow_location:
            continue
        if PRICE_RE.search(column) and not allow_price:
            continue
        if TEXT_RE.search(column):
            continue
        raw = frame[column]
        unique_count = raw.nunique(dropna=True)
        if 1 < unique_count <= 64:
            values = raw.fillna("__MISSING__").astype(str)
            top_values = list(values.value_counts().head(20).index)
            for value in top_values:
                parts.append((values == value).astype("float32"))
                safe = safe_feature_token(value)
                names.append(f"cat__{column}__{safe}")
    if not parts:
        raise RuntimeError("No usable core metadata features were inferred from listings.csv")
    features = pd.concat(parts, axis=1)
    features.columns = names
    return features, names


def build_vectors(frame: pd.DataFrame, ablation: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    id_col = find_id_column(frame)
    listing_ids = frame[id_col].astype(str) if id_col else pd.Series(frame.index.astype(str), index=frame.index)
    masks: list[pd.Series] = []
    feature_frames: list[pd.DataFrame] = []
    feature_sources: dict[str, list[str]] = {}

    if ablation == "fees_and_price":
        price, price_names = core_features(
            frame,
            allow_location=False,
            allow_price=True,
            include_pattern=PRICE_RE,
        )
        feature_frames.append(price)
        feature_sources["price_and_fees"] = price_names
    elif ablation == "property_shape":
        shape, shape_names = core_features(
            frame,
            allow_location=False,
            allow_price=False,
            include_pattern=SHAPE_RE,
        )
        feature_frames.append(shape)
        feature_sources["property_shape"] = shape_names
    elif ablation == "price_shape_metadata":
        price_shape, price_shape_names = core_features(
            frame,
            allow_location=False,
            allow_price=True,
            include_pattern=re.compile(f"({PRICE_RE.pattern})|({SHAPE_RE.pattern})", re.I),
        )
        feature_frames.append(price_shape)
        feature_sources["price_shape_metadata"] = price_shape_names
    if ablation == "core_metadata":
        core, core_names = core_features(frame, allow_location=True, allow_price=True)
        feature_frames.append(core)
        feature_sources["core_metadata_without_raw_coordinates"] = core_names
    elif ablation == "structured_metadata":
        core, core_names = core_features(
            frame,
            allow_location=True,
            allow_price=True,
            exclude_pattern=CONTEXT_RE,
        )
        feature_frames.append(core)
        feature_sources["structured_metadata_without_context_or_raw_coordinates"] = core_names
    elif ablation == "context_metadata":
        core, core_names = core_features(
            frame,
            allow_location=True,
            allow_price=False,
            include_pattern=CONTEXT_RE,
        )
        feature_frames.append(core)
        feature_sources["context_metadata_without_price_or_raw_coordinates"] = core_names
    elif ablation == "no_price":
        core, core_names = core_features(frame, allow_location=False, allow_price=False)
        feature_frames.append(core)
        feature_sources["core_without_price"] = core_names
    elif ablation == "no_location":
        core, core_names = core_features(frame, allow_location=False, allow_price=True)
        feature_frames.append(core)
        feature_sources["core_without_location"] = core_names

    mask = pd.Series(True, index=frame.index)
    for part_mask in masks:
        mask &= part_mask
    vectors = pd.concat(feature_frames, axis=1).loc[mask].copy()
    ids = listing_ids.loc[mask].reset_index(drop=True)
    vectors = vectors.reset_index(drop=True)
    means = vectors.mean(axis=0)
    stds = vectors.std(axis=0).replace(0, 1).fillna(1)
    vectors = ((vectors - means) / stds).astype("float32")
    vectors.insert(0, "listing_id", ids)
    duplicate_columns = vectors.columns[vectors.columns.duplicated()].tolist()
    if duplicate_columns:
        raise RuntimeError(f"Duplicate vector columns after feature construction: {duplicate_columns}")
    manifest = {
        "ablation": ablation,
        "item_count": int(len(vectors)),
        "vector_dim": int(vectors.shape[1] - 1),
        "source_rows": int(len(frame)),
        "dropped_rows": int(len(frame) - len(vectors)),
        "id_column": id_col,
        "feature_sources": feature_sources,
        "normalization": "per-feature z-score after median imputation and 1/99 percentile clipping",
    }
    return vectors, manifest


class ResidualVQ(nn.Module):
    def __init__(
        self,
        levels: int,
        codebook_size: int,
        dim: int,
        beta: float,
        *,
        use_rotation_trick: bool,
    ) -> None:
        super().__init__()
        self.levels = levels
        self.codebook_size = codebook_size
        self.beta = beta
        self.use_rotation_trick = use_rotation_trick
        self.codebooks = nn.Parameter(torch.randn(levels, codebook_size, dim) * 0.02)
        self.register_buffer("usage_count", torch.zeros(levels, codebook_size))

    @staticmethod
    def l2norm(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return F.normalize(tensor, p=2, dim=-1, eps=eps)

    @classmethod
    def rotate_to(cls, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Rotation-trick STE from the reference implementation, simplified for 2D tensors."""
        source_norm = source.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        target_norm = target.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        u = source / source_norm
        q = target / target_norm
        w = cls.l2norm(u + q).detach()
        e = source.unsqueeze(1)
        w_col = w.unsqueeze(-1)
        w_row = w.unsqueeze(-2)
        u_col = u.unsqueeze(-1).detach()
        q_row = q.unsqueeze(-2).detach()
        rotated = e - 2 * (e @ w_col @ w_row) + 2 * (e @ u_col @ q_row)
        return rotated.squeeze(1) * (target_norm / source_norm).detach()

    def gradient_estimator(self, residual: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        if self.training and residual.requires_grad and self.use_rotation_trick:
            return self.rotate_to(residual, quantized)
        return residual + (quantized - residual).detach()

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]], torch.Tensor]:
        residual = z
        quantized_sum = torch.zeros_like(z)
        losses = []
        codes = []
        details = []
        for level in range(self.levels):
            codebook = self.codebooks[level]
            distances = torch.cdist(residual, codebook)
            indices = distances.argmin(dim=1)
            quantized = codebook[indices]
            codebook_loss = F.mse_loss(quantized, residual.detach())
            commitment_loss = F.mse_loss(residual, quantized.detach())
            level_loss = codebook_loss + self.beta * commitment_loss
            losses.append(level_loss)
            quantized_st = self.gradient_estimator(residual, quantized)
            quantized_sum = quantized_sum + quantized_st
            residual = residual - quantized.detach()
            codes.append(indices)
            if self.training:
                with torch.no_grad():
                    self.usage_count[level].scatter_add_(
                        0,
                        indices.detach(),
                        torch.ones_like(indices, dtype=torch.float32),
                    )
            details.append(
                {
                    "codebook_loss": codebook_loss,
                    "commitment_loss": commitment_loss,
                    "level_loss": level_loss,
                }
            )
        return quantized_sum, torch.stack(losses).sum(), torch.stack(codes, dim=1), details, residual

    @torch.no_grad()
    def reset_unused_codes(self, z_sample: torch.Tensor) -> list[int]:
        residual = z_sample
        reset_counts: list[int] = []
        for level in range(self.levels):
            unused = (self.usage_count[level] == 0).nonzero(as_tuple=False).flatten()
            if len(unused) and len(residual) >= len(unused):
                sample_idx = torch.randperm(len(residual), device=residual.device)[: len(unused)]
                self.codebooks[level, unused] = residual[sample_idx]
            reset_counts.append(int(len(unused)))
            codebook = self.codebooks[level]
            indices = torch.cdist(residual, codebook).argmin(dim=1)
            residual = residual - codebook[indices]
        self.usage_count.zero_()
        return reset_counts


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        levels: int,
        codebook_size: int,
        beta: float,
        *,
        use_rotation_trick: bool,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.vq = ResidualVQ(
            levels,
            codebook_size,
            latent_dim,
            beta,
            use_rotation_trick=use_rotation_trick,
        )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
        z = self.encoder(x)
        quantized, rq_loss, codes, details, _ = self.vq(z)
        reconstructed = self.decoder(quantized)
        return reconstructed, rq_loss, codes, details

    @torch.no_grad()
    def kmeans_init(self, sample: torch.Tensor) -> None:
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            return
        z = self.encoder(sample)
        residual = z
        for level in range(self.vq.levels):
            n_clusters = min(self.vq.codebook_size, len(residual))
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=17, batch_size=4096)
            centers = kmeans.fit(residual.detach().cpu().numpy()).cluster_centers_
            center_tensor = torch.as_tensor(centers, dtype=self.vq.codebooks.dtype, device=sample.device)
            self.vq.codebooks[level, :n_clusters] = center_tensor
            if n_clusters < self.vq.codebook_size:
                extra = torch.randint(0, n_clusters, (self.vq.codebook_size - n_clusters,), device=sample.device)
                self.vq.codebooks[level, n_clusters:] = center_tensor[extra]
            indices = torch.cdist(residual, self.vq.codebooks[level]).argmin(dim=1)
            residual = residual - self.vq.codebooks[level, indices]

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        chunks = []
        for start in range(0, len(x), batch_size):
            z = self.encoder(x[start : start + batch_size])
            _, _, codes, _, _ = self.vq(z)
            chunks.append(codes.cpu())
        return torch.cat(chunks, dim=0)


def code_metrics(codes: np.ndarray, item_count: int, codebook_size: int) -> dict[str, Any]:
    full = [tuple(row.tolist()) for row in codes]
    full_counts = Counter(full)
    first_prefix = [row[:1] for row in full]
    second_prefix = [row[:2] for row in full]
    second_counts = Counter(second_prefix)
    usage = []
    entropy = []
    gini = []
    for level in range(codes.shape[1]):
        level_codes = codes[:, level].tolist()
        counts = np.array(list(Counter(level_codes).values()), dtype=np.float64)
        probs = counts / counts.sum() if counts.sum() else counts
        used = int(len(counts))
        usage.append({"level": level, "used_codes": used, "usage_fraction": used / codebook_size})
        entropy.append(
            {
                "level": level,
                "entropy": float(-(probs * np.log2(probs + 1e-12)).sum()) if len(probs) else 0.0,
                "normalized_entropy": float((-(probs * np.log2(probs + 1e-12)).sum()) / np.log2(codebook_size))
                if codebook_size > 1 and len(probs)
                else 0.0,
            }
        )
        if len(counts):
            sorted_counts = np.sort(counts)
            n = len(sorted_counts)
            gini_value = (2 * np.arange(1, n + 1) @ sorted_counts) / (n * sorted_counts.sum()) - (n + 1) / n
        else:
            gini_value = 0.0
        gini.append({"level": level, "gini": float(gini_value)})
    return {
        "codebook_usage": usage,
        "codebook_entropy": entropy,
        "codebook_gini": gini,
        "unique_full_codes": int(len(full_counts)),
        "assignability": float(len(codes) / item_count) if item_count else 0.0,
        "collision_free_fraction": float(len(full_counts) / item_count) if item_count else 0.0,
        "prefix_spread_l1": int(len(set(first_prefix))),
        "prefix_spread_l2": int(len(second_counts)),
        "max_prefix_size_l2": int(max(second_counts.values())) if second_counts else 0,
        "max_full_code_size": int(max(full_counts.values())) if full_counts else 0,
    }


def semantic_id_frame(listing_ids: pd.Series, codes: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append a deterministic uniqueness suffix per duplicate code path."""
    code_tuples = [tuple(row.tolist()) for row in codes]
    seen: Counter[tuple[int, ...]] = Counter()
    suffixes = []
    for code in code_tuples:
        suffixes.append(seen[code])
        seen[code] += 1
    frame = pd.DataFrame(codes, columns=[f"sid_{i}" for i in range(codes.shape[1])])
    frame["sid_unique_suffix"] = suffixes
    frame.insert(0, "listing_id", listing_ids.astype(str).to_numpy())
    full_counts = Counter(code_tuples)
    collision_groups = sum(1 for count in full_counts.values() if count > 1)
    collided_items = sum(count for count in full_counts.values() if count > 1)
    report = {
        "base_code_unique_count": int(len(full_counts)),
        "base_code_collision_groups": int(collision_groups),
        "base_code_collided_items": int(collided_items),
        "max_base_code_size": int(max(full_counts.values())) if full_counts else 0,
        "unique_suffix_max": int(max(suffixes)) if suffixes else 0,
        "unique_after_suffix": True,
        "semantic_id_columns": [f"sid_{i}" for i in range(codes.shape[1])] + ["sid_unique_suffix"],
    }
    return frame, report


def train_rqvae(
    vectors: pd.DataFrame, *, config: RQVAETrainConfig, wandb_run: Any = None
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    x_np = vectors.drop(columns=["listing_id"]).to_numpy(dtype=np.float32).copy()
    if len(x_np) < 10:
        raise RuntimeError(f"Need at least 10 vectors for train/validation split, got {len(x_np)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x_np)
    indices = torch.randperm(len(x))
    val_size = max(1, min(len(x) // 5, config.validation_size))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    train_x = x[train_idx].to(device)
    val_x = x[val_idx].to(device)

    input_dim = x.shape[1]
    latent_dim = min(config.max_latent_dim, max(8, input_dim * 2))
    hidden_dim = min(config.max_hidden_dim, max(32, input_dim * 4))
    model = RQVAE(
        input_dim,
        latent_dim,
        hidden_dim,
        config.rq_levels,
        config.codebook_size,
        config.commitment_beta,
        use_rotation_trick=config.use_rotation_trick,
    ).to(device)
    if config.use_kmeans_init:
        model.kmeans_init(train_x[: min(len(train_x), 20000)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=max(config.min_learning_rate / config.learning_rate, 1e-8),
        total_iters=max(1, config.warmup_steps),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000000,
        eta_min=config.min_learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[max(1, config.warmup_steps)],
    )
    batch_size = min(config.batch_size, max(16, len(train_x)))
    started = time.monotonic()
    deadline = started + max(20, config.max_seconds - 20)
    steps = 0
    last_loss = None
    reset_history = []
    while time.monotonic() < deadline:
        batch_idx = torch.randint(0, len(train_x), (batch_size,), device=device)
        batch = train_x[batch_idx]
        recon, rq_loss, _, details = model(batch)
        recon_loss = F.mse_loss(recon, batch)
        loss = recon_loss + rq_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {steps}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm).detach().cpu())
        optimizer.step()
        scheduler.step()
        steps += 1
        last_loss = float(loss.detach().cpu())
        if steps == 1 or steps % 1000 == 0:
            train_log = {
                "train/total_loss": last_loss,
                "train/reconstruction_loss": float(recon_loss.detach().cpu()),
                "train/rq_loss": float(rq_loss.detach().cpu()),
                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train/gradient_norm_before_clip": grad_norm,
            }
            for level, detail in enumerate(details):
                train_log[f"train/codebook_loss_l{level}"] = float(detail["codebook_loss"].detach().cpu())
                train_log[f"train/commitment_loss_l{level}"] = float(detail["commitment_loss"].detach().cpu())
            wandb_log(wandb_run, train_log, step=steps)
        if config.reset_unused_codes and steps % config.reset_every_steps == 0:
            with torch.no_grad():
                reset_counts = model.vq.reset_unused_codes(model.encoder(batch))
            reset_history.append({"step": steps, "reset_counts": reset_counts})
    model.eval()
    with torch.no_grad():
        recon, rq_loss, _, details = model(val_x)
        val_recon_loss = float(F.mse_loss(recon, val_x).cpu())
        val_total_loss = float((F.mse_loss(recon, val_x) + rq_loss).cpu())
        all_codes = model.encode_codes(x.to(device)).numpy()
    final_loss_components = {}
    for level, detail in enumerate(details):
        final_loss_components[f"validation_codebook_loss_l{level}"] = float(detail["codebook_loss"].detach().cpu())
        final_loss_components[f"validation_commitment_loss_l{level}"] = float(detail["commitment_loss"].detach().cpu())
    metrics = {
        "training_seconds": round(time.monotonic() - started, 3),
        "training_steps_completed": int(steps),
        "train_last_total_loss": last_loss,
        "validation_reconstruction_loss": val_recon_loss,
        "validation_total_loss": val_total_loss,
        "device": str(device),
        "input_dim": int(input_dim),
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "rq_levels": config.rq_levels,
        "codebook_size": config.codebook_size,
        "commitment_beta": config.commitment_beta,
        "use_rotation_trick": config.use_rotation_trick,
        "use_kmeans_init": config.use_kmeans_init,
        "reset_unused_codes": config.reset_unused_codes,
        "codebook_reset_events": reset_history[-10:],
        "loss_formula": (
            "MSE(x, decoder(rq(encoder(x)))) + sum_l MSE(q_l, sg(residual_l)) "
            "+ beta * MSE(residual_l, sg(q_l)); beta=0.25"
        ),
        **final_loss_components,
        **code_metrics(all_codes, len(x), config.codebook_size),
    }
    wandb_log(wandb_run, {f"final/{key}": value for key, value in metrics.items()}, step=steps)
    return metrics, {"state_dict": model.state_dict(), "codes": torch.from_numpy(all_codes)}


def run_ablation(
    *,
    campaign_id: str,
    dataset_id: str,
    ablation: str,
    run_id: str,
    output_root: Path,
    train_config: RQVAETrainConfig,
    vector_id: str | None = None,
) -> dict[str, Any]:
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation {ablation!r}; expected one of {sorted(ABLATIONS)}")
    dataset_root = output_root / campaign_id / "datasets" / dataset_id
    raw_listings = dataset_root / "raw" / "listings.csv"
    if not raw_listings.exists():
        raise FileNotFoundError(f"Missing raw snapshot artifact: {raw_listings}")
    run_root = output_root / campaign_id / "runs" / run_id
    vector_id = vector_id or f"{ablation}-{run_id}"
    vector_root = dataset_root / "vectors" / vector_id
    started_at = utc_now()
    wandb_run = maybe_init_wandb(
        run_id=run_id,
        ablation=ablation,
        config={
            "campaign_id": campaign_id,
            "dataset_id": dataset_id,
            "ablation": ablation,
            "vector_id": vector_id,
            **asdict(train_config),
        },
    )
    try:
        frame = pd.read_csv(raw_listings, low_memory=False)
        vectors, vector_manifest = build_vectors(frame, ablation)
        vector_path = vector_root / "listing_vectors.parquet"
        vector_manifest_path = vector_root / "manifest.json"
        vector_root.mkdir(parents=True, exist_ok=True)
        vectors.to_parquet(vector_path, index=False)
        vector_manifest["artifact_path"] = str(vector_path)
        vector_manifest["vector_id"] = vector_id
        vector_manifest["created_at"] = utc_now()
        write_json(vector_manifest_path, vector_manifest)
    except Exception as exc:
        crash_manifest = {
            "campaign_id": campaign_id,
            "dataset_id": dataset_id,
            "run_id": run_id,
            "stage": "rqvae_ablation",
            "ablation": ablation,
            "status": "crash",
            "decision": "crash",
            "started_at": started_at,
            "finished_at": utc_now(),
            "error": repr(exc),
        }
        write_json(run_root / "manifest.json", crash_manifest)
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise

    if wandb_run is not None:
        wandb_run.config.update(
            {
                "item_count": vector_manifest["item_count"],
                "vector_dim": vector_manifest["vector_dim"],
                "feature_sources": vector_manifest["feature_sources"],
            },
            allow_val_change=True,
        )

    metrics, model_artifacts = train_rqvae(
        vectors,
        config=train_config,
        wandb_run=wandb_run,
    )
    metrics.update(
        {
            "campaign_id": campaign_id,
            "dataset_id": dataset_id,
            "run_id": run_id,
            "ablation": ablation,
            "item_count": vector_manifest["item_count"],
            "vector_dim": vector_manifest["vector_dim"],
            "vector_id": vector_id,
            "vector_artifact_uri": str(vector_path),
        }
    )
    rqvae_dir = run_root / "rqvae"
    checkpoint_dir = rqvae_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    torch.save(model_artifacts["state_dict"], checkpoint_path)
    codes = model_artifacts["codes"].numpy()
    semantic_dir = run_root / "semantic_ids"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    codes_path = semantic_dir / "listing_semantic_ids.parquet"
    codes_frame, assignment_report = semantic_id_frame(vectors["listing_id"], codes)
    codes_frame.to_parquet(codes_path, index=False)
    assignment_report_path = semantic_dir / "assignment_report.json"
    write_json(assignment_report_path, assignment_report)
    metrics.update(
        {
            "collision_groups": assignment_report["base_code_collision_groups"],
            "collided_items": assignment_report["base_code_collided_items"],
            "unique_suffix_max": assignment_report["unique_suffix_max"],
            "collision_resolved_assignability": 1.0 if assignment_report["unique_after_suffix"] else 0.0,
        }
    )

    metrics_path = rqvae_dir / "metrics.json"
    write_json(metrics_path, metrics)
    decision = "keep" if metrics["training_steps_completed"] > 0 and metrics["assignability"] >= 0.999 else "discard"
    manifest = {
        "campaign_id": campaign_id,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "parent_run_id": None,
        "stage": "rqvae_ablation",
        "ablation": ablation,
        "started_at": started_at,
        "finished_at": utc_now(),
        "status": "ok",
        "decision": decision,
        "artifacts": {
            "run_root": str(run_root),
            "vector_manifest": str(vector_manifest_path),
            "listing_vectors": str(vector_path),
            "metrics": str(metrics_path),
            "checkpoint": str(checkpoint_path),
            "semantic_ids": str(codes_path),
            "assignment_report": str(assignment_report_path),
        },
        "train_config": asdict(train_config),
        "vector_manifest": vector_manifest,
        "metrics": metrics,
    }
    manifest_path = run_root / "manifest.json"
    write_json(manifest_path, manifest)
    if wandb_run is not None:
        wandb_log(
            wandb_run,
            {
                "result/validation_reconstruction_loss": metrics["validation_reconstruction_loss"],
                "result/collision_free_fraction": metrics["collision_free_fraction"],
                "result/max_prefix_size_l2": metrics["max_prefix_size_l2"],
                "result/assignability": metrics["assignability"],
            },
            step=metrics["training_steps_completed"],
        )
        wandb_run.finish()
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "ablation": ablation,
        "status": "ok",
        "decision": decision,
        "manifest_path": str(manifest_path),
        "artifact_uri": str(run_root),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--ablation", required=True, choices=sorted(ABLATIONS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--vector-id")
    parser.add_argument("--rq-levels", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--use-kmeans-init", action="store_true")
    parser.add_argument("--no-rotation-trick", action="store_true")
    parser.add_argument("--no-reset-unused-codes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = RQVAETrainConfig(
        max_seconds=args.max_seconds,
        seed=args.seed,
        rq_levels=args.rq_levels,
        codebook_size=args.codebook_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_kmeans_init=args.use_kmeans_init,
        use_rotation_trick=not args.no_rotation_trick,
        reset_unused_codes=not args.no_reset_unused_codes,
    )
    result = run_ablation(
        campaign_id=args.campaign_id,
        dataset_id=args.dataset_id,
        ablation=args.ablation,
        run_id=args.run_id,
        output_root=args.output_root,
        train_config=train_config,
        vector_id=args.vector_id,
    )
    print("RESULT_JSON\t" + json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
