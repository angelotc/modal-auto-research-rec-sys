#!/usr/bin/env python3
"""Shared runtime configuration for auto-tiger.

Keep stable paths and service defaults here. Per-experiment knobs such as
`max_steps`, model width, optimizer, and beam size belong in `train.py` or the
Modal function arguments so the autonomous loop can vary them per run.
"""

from __future__ import annotations

import os
from pathlib import Path


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


APP_NAME = env_str("AUTO_TIGER_MODAL_APP_NAME", "nipponhomes-auto-tiger")
VOLUME_NAME = env_str("AUTO_TIGER_MODAL_VOLUME_NAME", "nipponhomes-auto-rec-sys")
RUNTIME_SECRET_NAME = env_str("AUTO_TIGER_RUNTIME_SECRET_NAME", "custom-secret")

APP_DIR = env_str("AUTO_TIGER_APP_DIR", "/app/auto-tiger")
VOLUME_PATH = env_str("AUTO_TIGER_VOLUME_PATH", "/vol")
ARTIFACT_ROOT = Path(env_str("AUTO_TIGER_ARTIFACT_ROOT", f"{VOLUME_PATH}/auto-rec-sys"))

DEFAULT_CAMPAIGN_ID = env_str("AUTO_TIGER_DEFAULT_CAMPAIGN_ID", "demo")
DEFAULT_DATASET_ID = env_str("AUTO_TIGER_DEFAULT_DATASET_ID", "raw-tables-001")
DEFAULT_SFT_RUN_ID = env_str(
    "AUTO_TIGER_DEFAULT_SFT_RUN_ID",
    "sft-raw-tables-001-eugene-curriculum-002",
)

DEFAULT_GPU = env_str("AUTO_TIGER_GPU", "T4")
DEFAULT_TRAIN_MEMORY_MB = env_int("AUTO_TIGER_MEMORY_MB", 32768)
DEFAULT_INSPECT_MEMORY_MB = env_int("AUTO_TIGER_INSPECT_MEMORY_MB", 8192)

EXPERIMENT_TABLE = env_str("AUTO_TIGER_EXPERIMENT_TABLE", "auto_tiger_experiment_results")
