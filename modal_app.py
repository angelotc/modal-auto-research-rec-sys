#!/usr/bin/env python3
"""Starting Modal workbench for auto-rec-sys.

Codex should extend this app with the next small data, model, evaluation, or
inspection function needed by the research loop.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).parent.resolve()
PREPARE_SCRIPT = SCRIPT_DIR / "prepare.py"
WANDB_LOGGING_SCRIPT = SCRIPT_DIR / "wandb_logging.py"
RQVAE_ABLATION_SCRIPT = SCRIPT_DIR / "rqvae_ablation.py"

APP_NAME = os.environ.get("AUTO_REC_SYS_MODAL_APP_NAME", "nipponhomes-auto-rec-sys")
VOLUME_NAME = os.environ.get("AUTO_REC_SYS_MODAL_VOLUME_NAME", "nipponhomes-auto-rec-sys")
DB_SECRET_NAME = os.environ.get("AUTO_REC_SYS_DB_SECRET_NAME", "nipponhomes-auto-rec-sys-db")
RUNTIME_SECRET_NAME = os.environ.get("AUTO_REC_SYS_RUNTIME_SECRET_NAME", "custom-secret")

APP_DIR = "/app/auto-rec-sys"
VOLUME_PATH = "/vol"
ARTIFACT_ROOT = f"{VOLUME_PATH}/auto-rec-sys"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
db_secret = modal.Secret.from_name(DB_SECRET_NAME)
runtime_secret = modal.Secret.from_name(RUNTIME_SECRET_NAME)
wandb_secret = modal.Secret.from_name(DB_SECRET_NAME)

prepare_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("psycopg[binary]")
    .add_local_file(PREPARE_SCRIPT, f"{APP_DIR}/prepare.py")
)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("wandb")
    .add_local_file(WANDB_LOGGING_SCRIPT, f"{APP_DIR}/wandb_logging.py")
)

rqvae_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy", "pandas", "pyarrow", "torch", "wandb")
    .add_local_file(RQVAE_ABLATION_SCRIPT, f"{APP_DIR}/rqvae_ablation.py")
    .add_local_file(WANDB_LOGGING_SCRIPT, f"{APP_DIR}/wandb_logging.py")
)


@app.function(
    image=prepare_image,
    cpu=4.0,
    memory=16384,
    timeout=60 * 60,
    volumes={VOLUME_PATH: volume},
    secrets=[db_secret],
)
def prepare_tables(campaign_id: str, dataset_id: str) -> dict:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            f"{APP_DIR}/prepare.py",
            "--campaign-id",
            campaign_id,
            "--dataset-id",
            dataset_id,
            "--output-root",
            ARTIFACT_ROOT,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    volume.commit()
    return json.loads(result.stdout)


@app.function(
    image=rqvae_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=8 * 60,
    volumes={VOLUME_PATH: volume},
    secrets=[wandb_secret],
)
def train_rqvae_ablation(
    campaign_id: str,
    dataset_id: str,
    ablation: str,
    run_id: str,
    max_seconds: int = 300,
    seed: int = 17,
    rq_levels: int = 4,
    codebook_size: int = 64,
    learning_rate: float = 2e-3,
    batch_size: int = 512,
    use_kmeans_init: bool = False,
) -> dict:
    import subprocess
    import sys

    command = [
        sys.executable,
        f"{APP_DIR}/rqvae_ablation.py",
        "--campaign-id",
        campaign_id,
        "--dataset-id",
        dataset_id,
        "--ablation",
        ablation,
        "--run-id",
        run_id,
        "--output-root",
        ARTIFACT_ROOT,
        "--max-seconds",
        str(max_seconds),
        "--seed",
        str(seed),
        "--rq-levels",
        str(rq_levels),
        "--codebook-size",
        str(codebook_size),
        "--learning-rate",
        str(learning_rate),
        "--batch-size",
        str(batch_size),
    ]
    if use_kmeans_init:
        command.append("--use-kmeans-init")

    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        volume.commit()
        return {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "ablation": ablation,
            "status": "crash",
            "decision": "crash",
            "artifact_uri": f"{ARTIFACT_ROOT}/{campaign_id}/runs/{run_id}",
            "error": {
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            },
        }
    volume.commit()
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("RESULT_JSON\t"):
            return json.loads(line.split("\t", 1)[1])
    return json.loads(result.stdout)


@app.function(
    image=rqvae_image,
    cpu=2.0,
    memory=8192,
    timeout=10 * 60,
    volumes={VOLUME_PATH: volume},
)
def inspect_listing_fields(campaign_id: str, dataset_id: str, sample_values: int = 5) -> dict:
    from pathlib import Path

    import pandas as pd

    dataset_root = Path(ARTIFACT_ROOT) / campaign_id / "datasets" / dataset_id
    raw_listings = dataset_root / "raw" / "listings.csv"
    if not raw_listings.exists():
        raise FileNotFoundError(f"Missing raw snapshot artifact: {raw_listings}")

    frame = pd.read_csv(raw_listings, low_memory=False)
    profile = {
        "campaign_id": campaign_id,
        "dataset_id": dataset_id,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "non_null": frame.notna().sum().astype(int).to_dict(),
        "null_count": frame.isna().sum().astype(int).to_dict(),
        "sample_values": {
            column: frame[column].dropna().astype(str).head(sample_values).tolist()
            for column in frame.columns
        },
    }
    profile_path = dataset_root / "profiles" / "listing_fields.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    volume.commit()
    return {"profile_path": str(profile_path), **profile}


@app.local_entrypoint()
def main(campaign_id: str = "demo", dataset_id: str = "raw-tables-001") -> None:
    result = prepare_tables.remote(campaign_id=campaign_id, dataset_id=dataset_id)
    print(f"Prepared raw Modal tables: {result}")


@app.local_entrypoint()
def rqvae_sweep(
    campaign_id: str = "demo",
    dataset_id: str = "raw-tables-001",
    run_prefix: str = "rqvae-ablation",
    max_seconds: int = 300,
) -> None:
    from datetime import datetime, timezone

    ablations = ["core_metadata", "no_price", "no_location", "property_shape", "fees_and_price"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []
    for ablation in ablations:
        run_id = f"{run_prefix}-{ablation}-{stamp}"
        result = train_rqvae_ablation.remote(
            campaign_id=campaign_id,
            dataset_id=dataset_id,
            ablation=ablation,
            run_id=run_id,
            max_seconds=max_seconds,
        )
        results.append(result)
        metrics = result.get("metrics", {})
        print(
            "\t".join(
                [
                    result["run_id"],
                    ablation,
                    str(metrics.get("item_count", "")),
                    str(metrics.get("vector_dim", "")),
                    str(metrics.get("training_steps_completed", "")),
                    f"{metrics.get('validation_reconstruction_loss', float('nan')):.6f}",
                    f"{metrics.get('collision_free_fraction', float('nan')):.4f}",
                    result["decision"],
                    result["artifact_uri"],
                ]
            )
        )
    print(json.dumps(results, indent=2, sort_keys=True))


@app.local_entrypoint()
def rqvae_core_tune(
    campaign_id: str = "demo",
    dataset_id: str = "raw-tables-001",
    run_prefix: str = "rqvae-core-tune",
    max_seconds: int = 300,
) -> None:
    from datetime import datetime, timezone

    configs = [
        {"suffix": "core_c128_l4", "codebook_size": 128, "rq_levels": 4, "use_kmeans_init": False},
        {"suffix": "core_c256_l4", "codebook_size": 256, "rq_levels": 4, "use_kmeans_init": False},
        {"suffix": "core_c128_l3", "codebook_size": 128, "rq_levels": 3, "use_kmeans_init": False},
        {"suffix": "core_c256_l3", "codebook_size": 256, "rq_levels": 3, "use_kmeans_init": False},
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []
    for config in configs:
        run_id = f"{run_prefix}-{config['suffix']}-{stamp}"
        result = train_rqvae_ablation.remote(
            campaign_id=campaign_id,
            dataset_id=dataset_id,
            ablation="core_metadata",
            run_id=run_id,
            max_seconds=max_seconds,
            codebook_size=config["codebook_size"],
            rq_levels=config["rq_levels"],
            use_kmeans_init=config["use_kmeans_init"],
        )
        results.append(result)
        metrics = result.get("metrics", {})
        print(
            "\t".join(
                [
                    result["run_id"],
                    result.get("dataset_id", dataset_id),
                    "core_metadata",
                    str(config["codebook_size"]),
                    str(config["rq_levels"]),
                    str(config["use_kmeans_init"]).lower(),
                    str(metrics.get("training_steps_completed", "")),
                    f"{metrics.get('validation_reconstruction_loss', float('nan')):.6f}",
                    f"{metrics.get('validation_total_loss', float('nan')):.6f}",
                    f"{metrics.get('collision_free_fraction', float('nan')):.6f}",
                    str(metrics.get("prefix_spread_l2", "")),
                    str(metrics.get("max_prefix_size_l2", "")),
                    result.get("decision", ""),
                    result.get("artifact_uri", ""),
                ]
            )
        )
    print("RESULT_JSON\t" + json.dumps(results, indent=2, sort_keys=True))


@app.local_entrypoint()
def rqvae_feature_source_sweep(
    campaign_id: str = "demo",
    dataset_id: str = "raw-tables-001",
    run_prefix: str = "rqvae-feature-src",
    max_seconds: int = 300,
) -> None:
    from datetime import datetime, timezone

    hypothesis = (
        "With codebook_size=256 and rq_levels=4 fixed, full core metadata quality is driven by "
        "which feature source family supplies collision-separating signal."
    )
    configs = [
        {
            "suffix": "structured",
            "ablation": "structured_metadata",
            "codebook_size": 256,
            "rq_levels": 4,
            "use_kmeans_init": False,
        },
        {
            "suffix": "context",
            "ablation": "context_metadata",
            "codebook_size": 256,
            "rq_levels": 4,
            "use_kmeans_init": False,
        },
        {
            "suffix": "price_shape",
            "ablation": "price_shape_metadata",
            "codebook_size": 256,
            "rq_levels": 4,
            "use_kmeans_init": False,
        },
    ]
    print(f"HYPOTHESIS\t{hypothesis}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []
    for config in configs:
        run_id = f"{run_prefix}-{config['suffix']}-{stamp}"
        result = train_rqvae_ablation.remote(
            campaign_id=campaign_id,
            dataset_id=dataset_id,
            ablation=config["ablation"],
            run_id=run_id,
            max_seconds=max_seconds,
            codebook_size=config["codebook_size"],
            rq_levels=config["rq_levels"],
            use_kmeans_init=config["use_kmeans_init"],
        )
        results.append(result)
        metrics = result.get("metrics", {})
        print(
            "\t".join(
                [
                    result["run_id"],
                    result.get("dataset_id", dataset_id),
                    config["ablation"],
                    str(config["codebook_size"]),
                    str(config["rq_levels"]),
                    str(config["use_kmeans_init"]).lower(),
                    str(metrics.get("item_count", "")),
                    str(metrics.get("vector_dim", "")),
                    str(metrics.get("training_steps_completed", "")),
                    f"{metrics.get('validation_reconstruction_loss', float('nan')):.6f}",
                    f"{metrics.get('validation_total_loss', float('nan')):.6f}",
                    f"{metrics.get('collision_free_fraction', float('nan')):.6f}",
                    str(metrics.get("prefix_spread_l2", "")),
                    str(metrics.get("max_prefix_size_l2", "")),
                    result.get("decision", ""),
                    result.get("artifact_uri", ""),
                ]
            )
        )
    print("RESULT_JSON\t" + json.dumps(results, indent=2, sort_keys=True))


@app.local_entrypoint()
def rqvae_one(
    campaign_id: str = "demo",
    dataset_id: str = "raw-tables-001",
    ablation: str = "core_metadata",
    run_id: str = "",
    max_seconds: int = 300,
    codebook_size: int = 256,
    rq_levels: int = 4,
    learning_rate: float = 2e-3,
    batch_size: int = 512,
    use_kmeans_init: bool = False,
) -> None:
    from datetime import datetime, timezone

    if not run_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"rqvae-one-{ablation}-{stamp}"
    result = train_rqvae_ablation.remote(
        campaign_id=campaign_id,
        dataset_id=dataset_id,
        ablation=ablation,
        run_id=run_id,
        max_seconds=max_seconds,
        codebook_size=codebook_size,
        rq_levels=rq_levels,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_kmeans_init=use_kmeans_init,
    )
    print("RESULT_JSON\t" + json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def fields(campaign_id: str = "demo", dataset_id: str = "raw-tables-001") -> None:
    result = inspect_listing_fields.remote(campaign_id=campaign_id, dataset_id=dataset_id)
    print(json.dumps(result, indent=2, sort_keys=True))
