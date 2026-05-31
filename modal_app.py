#!/usr/bin/env python3
"""Modal launch surface for auto-tiger experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).parent.resolve()
TRAIN_SCRIPT = SCRIPT_DIR / "train.py"
PREPARE_SCRIPT = SCRIPT_DIR / "prepare.py"
LOG_EXPERIMENT_SCRIPT = SCRIPT_DIR / "log_experiment.py"

try:
    from config import (
        APP_DIR,
        APP_NAME,
        ARTIFACT_ROOT,
        DEFAULT_CAMPAIGN_ID,
        DEFAULT_GPU,
        DEFAULT_INSPECT_MEMORY_MB,
        DEFAULT_SFT_RUN_ID,
        DEFAULT_TRAIN_MEMORY_MB,
        RUNTIME_SECRET_NAME,
        VOLUME_NAME,
        VOLUME_PATH,
    )
except ModuleNotFoundError:
    # Modal imports this entrypoint remotely before image-local files are
    # available. Keep these fallbacks in sync with config.py.
    APP_NAME = os.environ.get("AUTO_TIGER_MODAL_APP_NAME", "nipponhomes-auto-tiger")
    VOLUME_NAME = os.environ.get("AUTO_TIGER_MODAL_VOLUME_NAME", "nipponhomes-auto-rec-sys")
    RUNTIME_SECRET_NAME = os.environ.get("AUTO_TIGER_RUNTIME_SECRET_NAME", "custom-secret")
    APP_DIR = os.environ.get("AUTO_TIGER_APP_DIR", "/app/auto-tiger")
    VOLUME_PATH = os.environ.get("AUTO_TIGER_VOLUME_PATH", "/vol")
    ARTIFACT_ROOT = Path(os.environ.get("AUTO_TIGER_ARTIFACT_ROOT", f"{VOLUME_PATH}/auto-rec-sys"))
    DEFAULT_CAMPAIGN_ID = os.environ.get("AUTO_TIGER_DEFAULT_CAMPAIGN_ID", "demo")
    DEFAULT_SFT_RUN_ID = os.environ.get(
        "AUTO_TIGER_DEFAULT_SFT_RUN_ID",
        "sft-raw-tables-001-eugene-curriculum-002",
    )
    DEFAULT_GPU = os.environ.get("AUTO_TIGER_GPU", "T4")
    DEFAULT_TRAIN_MEMORY_MB = int(os.environ.get("AUTO_TIGER_MEMORY_MB", "32768"))
    DEFAULT_INSPECT_MEMORY_MB = int(os.environ.get("AUTO_TIGER_INSPECT_MEMORY_MB", "8192"))

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
runtime_secret = modal.Secret.from_name(
    RUNTIME_SECRET_NAME,
    required_keys=["DEEPLAKE_API_KEY"],
)

tiger_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy", "pandas", "pyarrow", "torch")
    .add_local_file(TRAIN_SCRIPT, f"{APP_DIR}/train.py")
    .add_local_file(PREPARE_SCRIPT, f"{APP_DIR}/prepare.py")
    .add_local_file(SCRIPT_DIR / "config.py", f"{APP_DIR}/config.py")
)

ledger_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("deeplake")
    .add_local_file(LOG_EXPERIMENT_SCRIPT, f"{APP_DIR}/log_experiment.py")
    .add_local_file(SCRIPT_DIR / "config.py", f"{APP_DIR}/config.py")
)


@app.function(
    image=tiger_image,
    cpu=2.0,
    memory=DEFAULT_INSPECT_MEMORY_MB,
    timeout=10 * 60,
    volumes={VOLUME_PATH: volume},
)
def inspect_inputs(
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
    sft_run_id: str = DEFAULT_SFT_RUN_ID,
) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            f"{APP_DIR}/prepare.py",
            "--artifact-root",
            str(ARTIFACT_ROOT),
            "--campaign-id",
            campaign_id,
            "--sft-run-id",
            sft_run_id,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "prepare.py input inspection failed "
            f"with exit code {result.returncode}:\n{result.stderr or result.stdout}"
        )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return json.loads(result.stdout)


@app.function(
    image=tiger_image,
    gpu=DEFAULT_GPU,
    cpu=4.0,
    memory=DEFAULT_TRAIN_MEMORY_MB,
    timeout=4 * 60 * 60,
    volumes={VOLUME_PATH: volume},
)
def train_tiger_experiment(
    campaign_id: str,
    dataset_id: str,
    sft_run_id: str,
    run_id: str,
    max_history_items: int = 20,
    hidden_size: int = 128,
    layers: int = 4,
    heads: int = 4,
    feedforward_size: int = 1024,
    dropout: float = 0.1,
    max_steps: int = 400,
    checkpoint_steps: int = 100,
    max_minutes: float = 0.0,
    resume_checkpoint: str = "",
    batch_size: int = 256,
    beam_size: int = 10,
    rerank_prefix_bonus: float = 0.0,
    rerank_prefix_depth: int = 0,
    rerank_recency_mode: str = "inverse",
    rerank_popularity_penalty: float = 0.0,
    learning_rate: float = 0.001,
    label_smoothing: float = 0.0,
    item_loss_weight: float = 0.0,
    item_loss_negatives: int = 256,
    item_loss_temperature: float = 0.1,
    weight_decay: float = 0.00001,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    use_sid_position_embeddings: bool = False,
) -> dict:
    command = [
        sys.executable,
        f"{APP_DIR}/train.py",
        "--artifact-root",
        str(ARTIFACT_ROOT),
        "--campaign-id",
        campaign_id,
        "--dataset-id",
        dataset_id,
        "--sft-run-id",
        sft_run_id,
        "--run-id",
        run_id,
        "--max-history-items",
        str(max_history_items),
        "--hidden-size",
        str(hidden_size),
        "--layers",
        str(layers),
        "--heads",
        str(heads),
        "--feedforward-size",
        str(feedforward_size),
        "--dropout",
        str(dropout),
        "--max-steps",
        str(max_steps),
        "--checkpoint-steps",
        str(checkpoint_steps),
        "--max-minutes",
        str(max_minutes),
        "--batch-size",
        str(batch_size),
        "--beam-size",
        str(beam_size),
        "--rerank-prefix-bonus",
        str(rerank_prefix_bonus),
        "--rerank-prefix-depth",
        str(rerank_prefix_depth),
        "--rerank-recency-mode",
        rerank_recency_mode,
        "--rerank-popularity-penalty",
        str(rerank_popularity_penalty),
        "--learning-rate",
        str(learning_rate),
        "--label-smoothing",
        str(label_smoothing),
        "--item-loss-weight",
        str(item_loss_weight),
        "--item-loss-negatives",
        str(item_loss_negatives),
        "--item-loss-temperature",
        str(item_loss_temperature),
        "--weight-decay",
        str(weight_decay),
        "--max-grad-norm",
        str(max_grad_norm),
        "--seed",
        str(seed),
    ]
    if use_sid_position_embeddings:
        command.append("--use-sid-position-embeddings")
    if resume_checkpoint:
        command.extend(["--resume-checkpoint", resume_checkpoint])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            "train.py failed "
            f"with exit code {result.returncode}:\n{result.stderr or result.stdout}"
        )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    volume.commit()
    return json.loads(result.stdout)


@app.function(
    image=tiger_image,
    gpu=DEFAULT_GPU,
    cpu=4.0,
    memory=DEFAULT_TRAIN_MEMORY_MB,
    timeout=4 * 60 * 60,
    volumes={VOLUME_PATH: volume},
)
def report_tiger_saved_checkpoints(
    campaign_id: str,
    dataset_id: str,
    sft_run_id: str,
    run_id: str,
    checkpoint_run_id: str = "",
    report_checkpoint_step: int = 0,
    beam_size: int = 10,
    rerank_prefix_bonus: float = 0.0,
    rerank_prefix_depth: int = 0,
    rerank_recency_mode: str = "inverse",
    rerank_popularity_penalty: float = 0.0,
) -> dict:
    command = [
            sys.executable,
            f"{APP_DIR}/train.py",
            "--artifact-root",
            str(ARTIFACT_ROOT),
            "--campaign-id",
            campaign_id,
            "--dataset-id",
            dataset_id,
            "--sft-run-id",
            sft_run_id,
            "--run-id",
            run_id,
            "--beam-size",
            str(beam_size),
            "--rerank-prefix-bonus",
            str(rerank_prefix_bonus),
            "--rerank-prefix-depth",
            str(rerank_prefix_depth),
            "--rerank-recency-mode",
            rerank_recency_mode,
            "--rerank-popularity-penalty",
            str(rerank_popularity_penalty),
            "--report-saved-checkpoints",
        ]
    if checkpoint_run_id:
        command.extend(["--checkpoint-run-id", checkpoint_run_id])
    if report_checkpoint_step:
        command.extend(["--report-checkpoint-step", str(report_checkpoint_step)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "train.py saved-checkpoint report failed "
            f"with exit code {result.returncode}:\n{result.stderr or result.stdout}"
        )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    volume.commit()
    return json.loads(result.stdout)


@app.function(
    image=ledger_image,
    timeout=10 * 60,
    secrets=[runtime_secret],
)
def log_experiment_remote(*args: str) -> str:
    if not args or args[0] not in {"log", "recent"}:
        raise ValueError("args must start with a supported log_experiment.py command")
    result = subprocess.run(
        [sys.executable, f"{APP_DIR}/log_experiment.py", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "log_experiment.py failed "
            f"with exit code {result.returncode}:\n{result.stderr or result.stdout}"
        )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr)
    return result.stdout
