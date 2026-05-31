#!/usr/bin/env python3
"""Fixed Modal artifact contract and input inspection for auto-tiger.

This file is intentionally small. `train.py` owns the mutable TIGER experiment
logic; this module owns the stable path assumptions and lightweight checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import ARTIFACT_ROOT, DEFAULT_CAMPAIGN_ID, DEFAULT_SFT_RUN_ID


def run_root(artifact_root: Path, campaign_id: str, run_id: str) -> Path:
    return artifact_root / campaign_id / run_id


def sft_inputs(artifact_root: Path, campaign_id: str, sft_run_id: str) -> dict[str, Path]:
    root = run_root(artifact_root, campaign_id, sft_run_id)
    return {
        "train_jsonl": root / "datasets" / "llm_sft" / "history_next_train.jsonl",
        "eval_jsonl": root / "datasets" / "eval" / "history_next_eval.jsonl",
        "catalog_semantic_ids": root / "datasets" / "eval" / "catalog_semantic_ids.parquet",
    }


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _line in handle)


def inspect_inputs(artifact_root: Path, campaign_id: str, sft_run_id: str) -> dict[str, Any]:
    paths = sft_inputs(artifact_root, campaign_id, sft_run_id)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return {
            "ok": False,
            "artifact_root": str(artifact_root),
            "campaign_id": campaign_id,
            "sft_run_id": sft_run_id,
            "missing": missing,
            "paths": {key: str(path) for key, path in paths.items()},
        }

    catalog = pd.read_parquet(paths["catalog_semantic_ids"], columns=["semantic_id"])
    return {
        "ok": True,
        "artifact_root": str(artifact_root),
        "campaign_id": campaign_id,
        "sft_run_id": sft_run_id,
        "paths": {key: str(path) for key, path in paths.items()},
        "counts": {
            "train_jsonl_lines": count_jsonl(paths["train_jsonl"]),
            "eval_jsonl_lines": count_jsonl(paths["eval_jsonl"]),
            "catalog_semantic_ids": int(catalog["semantic_id"].astype(str).nunique()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--sft-run-id", default=DEFAULT_SFT_RUN_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(inspect_inputs(args.artifact_root, args.campaign_id, args.sft_run_id), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
