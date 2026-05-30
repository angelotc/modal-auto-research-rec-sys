#!/usr/bin/env python3
"""Shared configuration constants for auto-rec-sys."""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PREPARE_SCRIPT = SCRIPT_DIR / "prepare.py"
LEDGER_PATH = SCRIPT_DIR / "results.tsv"

# ── Modal ───────────────────────────────────────────────────────────────────
APP_NAME = os.environ.get("AUTO_REC_SYS_MODAL_APP_NAME", "nipponhomes-auto-rec-sys")
VOLUME_NAME = os.environ.get("AUTO_REC_SYS_MODAL_VOLUME_NAME", "nipponhomes-auto-rec-sys")
DB_SECRET_NAME = os.environ.get("AUTO_REC_SYS_DB_SECRET_NAME", "nipponhomes-auto-rec-sys-db")
WANDB_SECRET_NAME = DB_SECRET_NAME

APP_DIR = "/app/auto-rec-sys"
VOLUME_PATH = "/vol"
ARTIFACT_ROOT = f"{VOLUME_PATH}/auto-rec-sys"

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_CAMPAIGN_ID = "demo"
DEFAULT_DATASET_ID = "raw-tables-001"

# ── DB snapshot tables ──────────────────────────────────────────────────────
TABLES = {
    "listings": "SELECT * FROM listings",
    "listing_views": "SELECT * FROM listing_views",
}
