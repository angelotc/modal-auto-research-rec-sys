#!/usr/bin/env python3
"""Read-only raw table snapshots for Modal-native auto-rec-sys research."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

TABLES = {
    "listings": "SELECT * FROM listings",
    "listing_views": "SELECT * FROM listing_views",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_database_url() -> str:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required to snapshot auto-rec-sys tables.")
    return database_url


def dataset_dir(output_root: Path, campaign_id: str, dataset_id: str) -> Path:
    return output_root / campaign_id / "datasets" / dataset_id


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def snapshot_table(
    connection: psycopg.Connection[Any], *, table: str, query: str, output_path: Path
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with connection.cursor(name=f"auto_rec_sys_{table}_cursor") as cursor:
        cursor.itersize = 10000
        cursor.execute(query)
        columns = [column.name for column in cursor.description or []]
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)
            for row in cursor:
                writer.writerow(row)
                row_count += 1
    return {
        "table": table,
        "query": query,
        "csv_path": str(output_path),
        "columns": columns,
        "row_count": row_count,
    }


def snapshot_tables(*, campaign_id: str, dataset_id: str, output_root: Path) -> dict[str, Any]:
    root = dataset_dir(output_root, campaign_id, dataset_id)
    raw_dir = root / "raw"
    started_at = utc_now()

    with psycopg.connect(require_database_url()) as connection:
        table_metadata = {
            table: snapshot_table(
                connection,
                table=table,
                query=query,
                output_path=raw_dir / f"{table}.csv",
            )
            for table, query in TABLES.items()
        }

    metadata_path = raw_dir / "table_metadata.json"
    write_json(metadata_path, table_metadata)
    manifest = {
        "campaign_id": campaign_id,
        "dataset_id": dataset_id,
        "stage": "raw_table_snapshot",
        "started_at": started_at,
        "finished_at": utc_now(),
        "read_only": True,
        "tables": table_metadata,
        "artifacts": {
            "table_metadata": str(metadata_path),
            "raw_csvs": {
                table: metadata["csv_path"] for table, metadata in table_metadata.items()
            },
        },
        "next_stage": "Codex-authored Modal post-processing functions",
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "dataset_dir": str(root),
        "manifest_path": str(manifest_path),
        "tables": {
            table: {
                "row_count": metadata["row_count"],
                "csv_path": metadata["csv_path"],
            }
            for table, metadata in table_metadata.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = snapshot_tables(
        campaign_id=args.campaign_id,
        dataset_id=args.dataset_id,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
