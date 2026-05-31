#!/usr/bin/env python3
"""Log compact auto-tiger result rows to a Deeplake managed table."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import EXPERIMENT_TABLE

try:
    from deeplake import Client
except ModuleNotFoundError:
    Client = None

SCHEMA = {
    "run_id": "TEXT",
    "dataset_id": "TEXT",
    "parent_run_id": "TEXT",
    "stage": "TEXT",
    "direction": "TEXT",
    "hypothesis": "TEXT",
    "status": "TEXT",
    "primary_metric": "TEXT",
    "coverage": "TEXT",
    "cost_notes": "TEXT",
    "artifact_uri": "TEXT",
    "decision": "TEXT",
    "description": "TEXT",
    "timestamp": "TEXT",
}

TSV_COLUMNS = [
    "run_id",
    "stage",
    "status",
    "primary_metric",
    "recall_at_5",
    "ndcg_at_5",
    "recall_at_10",
    "ndcg_at_10",
    "peak_vram_gb",
    "artifact_uri",
    "decision",
    "description",
    "timestamp",
    "completed_steps",
    "train_seconds",
    "steps_per_second",
    "gpu_name",
]


def row_from_args(args: argparse.Namespace) -> dict[str, list[str]]:
    return {
        "run_id": [args.run_id],
        "dataset_id": [args.dataset_id],
        "parent_run_id": [args.parent_run_id],
        "stage": [args.stage],
        "direction": [args.direction],
        "hypothesis": [args.hypothesis],
        "status": [args.status],
        "primary_metric": [args.primary_metric],
        "coverage": [args.coverage],
        "cost_notes": [args.cost_notes],
        "artifact_uri": [args.artifact_uri],
        "decision": [args.decision],
        "description": [args.description],
        "timestamp": [datetime.now(timezone.utc).isoformat()],
    }


def ingest(row: dict[str, list[str]]) -> None:
    if Client is None:
        raise RuntimeError("deeplake is not installed; use the `tsv` command for local results.tsv logging")
    client = Client()
    tables = client.list_tables()
    if EXPERIMENT_TABLE in tables:
        client.ingest(EXPERIMENT_TABLE, row)
    else:
        client.ingest(EXPERIMENT_TABLE, row, schema=SCHEMA)


def recent(limit: int) -> list[dict[str, Any]]:
    if Client is None:
        raise RuntimeError("deeplake is not installed; use the `tsv-recent` command for local results.tsv reads")
    client = Client()
    if EXPERIMENT_TABLE not in client.list_tables():
        return []
    query_limit = int(limit)
    try:
        return client.query(f"SELECT * FROM {EXPERIMENT_TABLE} ORDER BY timestamp DESC LIMIT {query_limit}")
    except Exception as error:
        if "timestamp" not in str(error):
            raise
    return client.query(f"SELECT * FROM {EXPERIMENT_TABLE} LIMIT {query_limit}")


def append_tsv(args: argparse.Namespace) -> None:
    path = args.path
    row = {
        "run_id": args.run_id,
        "stage": args.stage,
        "status": args.status,
        "primary_metric": args.primary_metric,
        "recall_at_5": args.recall_at_5,
        "ndcg_at_5": args.ndcg_at_5,
        "recall_at_10": args.recall_at_10,
        "ndcg_at_10": args.ndcg_at_10,
        "peak_vram_gb": args.peak_vram_gb,
        "artifact_uri": args.artifact_uri,
        "decision": args.decision,
        "description": args.description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "completed_steps": args.completed_steps,
        "train_seconds": args.train_seconds,
        "steps_per_second": args.steps_per_second,
        "gpu_name": args.gpu_name,
    }
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_COLUMNS, delimiter="\t")
        if not exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def recent_tsv(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return rows[-limit:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    log_parser = subparsers.add_parser("log", help="Append one experiment row.")
    log_parser.add_argument("--run-id", required=True)
    log_parser.add_argument("--dataset-id", default="")
    log_parser.add_argument("--parent-run-id", default="")
    log_parser.add_argument("--stage", required=True)
    log_parser.add_argument("--direction", default="")
    log_parser.add_argument("--hypothesis", default="")
    log_parser.add_argument("--status", required=True)
    log_parser.add_argument("--primary-metric", default="")
    log_parser.add_argument("--coverage", default="")
    log_parser.add_argument("--cost-notes", default="")
    log_parser.add_argument("--artifact-uri", default="")
    log_parser.add_argument("--decision", default="")
    log_parser.add_argument("--description", default="")

    recent_parser = subparsers.add_parser("recent", help="Print recent result rows.")
    recent_parser.add_argument("--limit", type=int, default=20)

    tsv_parser = subparsers.add_parser("tsv", help="Append one local results.tsv row.")
    tsv_parser.add_argument("--path", type=Path, default=Path("results.tsv"))
    tsv_parser.add_argument("--run-id", required=True)
    tsv_parser.add_argument("--stage", required=True)
    tsv_parser.add_argument("--status", required=True)
    tsv_parser.add_argument("--primary-metric", default="")
    tsv_parser.add_argument("--recall-at-5", default="")
    tsv_parser.add_argument("--ndcg-at-5", default="")
    tsv_parser.add_argument("--recall-at-10", default="")
    tsv_parser.add_argument("--ndcg-at-10", default="")
    tsv_parser.add_argument("--peak-vram-gb", default="")
    tsv_parser.add_argument("--artifact-uri", default="")
    tsv_parser.add_argument("--decision", default="")
    tsv_parser.add_argument("--description", default="")
    tsv_parser.add_argument("--completed-steps", default="")
    tsv_parser.add_argument("--train-seconds", default="")
    tsv_parser.add_argument("--steps-per-second", default="")
    tsv_parser.add_argument("--gpu-name", default="")

    tsv_recent_parser = subparsers.add_parser("tsv-recent", help="Print recent local results.tsv rows.")
    tsv_recent_parser.add_argument("--path", type=Path, default=Path("results.tsv"))
    tsv_recent_parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "log":
        ingest(row_from_args(args))
        print(f"Logged {args.run_id} to {EXPERIMENT_TABLE}")
        return

    if args.command == "tsv":
        append_tsv(args)
        print(f"Logged {args.run_id} to {args.path}")
        return

    if args.command == "tsv-recent":
        for row in recent_tsv(args.path, args.limit):
            print(row)
        return

    for row in recent(args.limit):
        print(row)


if __name__ == "__main__":
    main()
