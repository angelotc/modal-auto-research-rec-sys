#!/usr/bin/env python3
"""Thin wandb metrics-only logging wrapper for Modal functions."""

from __future__ import annotations

import os
from typing import Any

import wandb


def init_run(
    config: dict[str, Any],
    run_id: str,
    tags: list[str] | None = None,
) -> wandb.sdk.wandb_run.Run:
    """Initialize a wandb run. Safe to call from any Modal container."""
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "auto-rec-sys"),
        entity=os.environ.get("WANDB_ENTITY"),
        id=run_id,
        resume="allow",
        config=config,
        tags=tags,
    )


def log(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log scalar metrics to the active wandb run."""
    wandb.log(metrics, step=step)


def finish() -> None:
    """Finalize the active wandb run."""
    wandb.finish()
