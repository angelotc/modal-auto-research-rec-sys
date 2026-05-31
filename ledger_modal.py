#!/usr/bin/env python3
"""Modal-only Deeplake ledger entrypoint for auto-tiger."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).parent.resolve()
LOG_EXPERIMENT_SCRIPT = SCRIPT_DIR / "log_experiment.py"

try:
    from config import APP_DIR, APP_NAME, RUNTIME_SECRET_NAME
except ModuleNotFoundError:
    # Modal imports this entrypoint remotely before image-local files are
    # available. Keep these fallbacks in sync with config.py.
    APP_NAME = os.environ.get("AUTO_TIGER_MODAL_APP_NAME", "nipponhomes-auto-tiger")
    RUNTIME_SECRET_NAME = os.environ.get("AUTO_TIGER_RUNTIME_SECRET_NAME", "custom-secret")
    APP_DIR = os.environ.get("AUTO_TIGER_APP_DIR", "/app/auto-tiger")

app = modal.App(APP_NAME)
runtime_secret = modal.Secret.from_name(
    RUNTIME_SECRET_NAME,
    required_keys=["DEEPLAKE_API_KEY"],
)

ledger_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("deeplake")
    .add_local_file(LOG_EXPERIMENT_SCRIPT, f"{APP_DIR}/log_experiment.py")
    .add_local_file(SCRIPT_DIR / "config.py", f"{APP_DIR}/config.py")
)


@app.function(
    image=ledger_image,
    timeout=10 * 60,
    secrets=[runtime_secret],
)
def log_experiment_remote(*args: str) -> str:
    """Run the result logger with the runtime Deeplake secret."""
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
