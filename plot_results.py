#!/usr/bin/env python3
"""Plot kept TIGER baseline progress from the local results.tsv ledger."""

from __future__ import annotations

import argparse
import csv
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str) and "=" in value:
            return float(value.rsplit("=", 1)[1])
        raise


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [
            {key: (value or "") for key, value in row.items() if key is not None}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]


def compact_run_id(run_id: str) -> str:
    label = re.sub(r"-2026\d{4}-\d+$", "", run_id)
    return label.removeprefix("tiger-")


def is_kept(row: dict[str, str]) -> bool:
    return row.get("decision", "").startswith("promote")


def summarize_change(row: dict[str, str]) -> str:
    run_id = row.get("run_id", "")
    description = row.get("description", "")
    text = f"{run_id} {description}".lower()
    pieces: list[str] = []

    if "baseline rerun" in text:
        return "baseline"

    beam_match = re.search(r"beam[- ]?(\d+)", text)
    if beam_match:
        pieces.append(f"beam-{beam_match.group(1)}")

    prefix_match = re.search(r"prefix bonus ([0-9.]+)", text)
    if "without prefix" in text:
        pieces.append("no prefix")
    elif prefix_match:
        pieces.append(f"prefix {prefix_match.group(1)}")

    pop_match = re.search(r"popularity penalty ([0-9.]+)", text)
    if pop_match:
        pieces.append(f"pop penalty {pop_match.group(1)}")

    if "uniform history recency" in text:
        pieces.append("uniform recency")
    elif "exponential history recency" in text:
        pieces.append("exp recency")

    if "three-layer" in text or "layers 3" in text:
        pieces.append("3 layers")
    elif "two-layer" in text or "layers 2" in text:
        pieces.append("2 layers")
    else:
        layer_match = re.search(r"(?:layers|l)(\d)\b", text)
        if layer_match:
            pieces.append(f"{layer_match.group(1)} layers")

    hidden_match = re.search(r"hidden_size (\d+)", text)
    if hidden_match:
        pieces.append(f"hidden {hidden_match.group(1)}")

    ff_match = re.search(r"feedforward_size (\d+)", text)
    if ff_match:
        pieces.append(f"ff {ff_match.group(1)}")

    if "gelu" in text:
        pieces.append("GELU")
    if "norm_first false" in text:
        pieces.append("post-norm")

    history_match = re.search(r"max_history_items (\d+)", text)
    if history_match:
        pieces.append(f"history {history_match.group(1)}")

    smoothing_match = re.search(r"label smoothing ([0-9.]+)", text)
    if smoothing_match:
        pieces.append(f"label smoothing {smoothing_match.group(1)}")

    return " + ".join(dict.fromkeys(pieces)) or compact_run_id(run_id)


def kept_running_best(
    rows: list[dict[str, str]], metric: str
) -> tuple[list[int], list[float], list[tuple[int, float, str]]]:
    best = float("-inf")
    step_x: list[int] = []
    step_y: list[float] = []
    annotations: list[tuple[int, float, str]] = []

    for index, row in enumerate(rows):
        if not is_kept(row):
            continue

        value = parse_float(row[metric])
        if value <= best + 1e-12:
            continue

        previous = best
        best = value
        step_x.append(index)
        step_y.append(value)

        if previous == float("-inf"):
            label = "baseline"
        else:
            label = f"{summarize_change(row)}\n+{value - previous:.4f}"
        annotations.append((index, value, label))

    return step_x, step_y, annotations


def padded_ylim(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    padding = max((high - low) * 0.45, 0.002)
    return max(0.0, low - padding), min(1.0, high + padding)


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise SystemExit(f"Could not find an available output filename near {path}")


def plot_results(input_path: Path, output_path: Path, overwrite: bool = False) -> Path:
    rows = [row for row in read_rows(input_path) if row.get("status") == "complete"]
    if not rows:
        raise SystemExit(f"No complete rows found in {input_path}")

    kept_x = [index for index, row in enumerate(rows) if is_kept(row)]
    discarded_x = [index for index, row in enumerate(rows) if not is_kept(row)]
    metrics = [
        ("recall_at_10", "Recall@10", "#2ca25f"),
        ("ndcg_at_10", "NDCG@10", "#756bb1"),
    ]

    width = max(14, min(28, len(rows) * 0.42))
    fig, ax = plt.subplots(figsize=(width, 6.5), layout="constrained")
    fig.suptitle(
        f"TIGER Training Progress: {len(rows)} Experiments, {len(kept_x)} Kept Baselines",
        fontsize=16,
    )

    all_values = [parse_float(row[metric]) for metric, _, _ in metrics for row in rows]
    kept_values = [
        parse_float(rows[index][metric])
        for metric, _, _ in metrics
        for index in kept_x
    ]
    focus_values = kept_values or all_values
    ax.set_ylim(*padded_ylim(focus_values))
    y_low, y_high = ax.get_ylim()
    label_offset = (y_high - y_low) * 0.025
    markers = {
        "recall_at_10": "o",
        "ndcg_at_10": "s",
    }

    for metric_index, (metric, metric_label, color) in enumerate(metrics):
        values = [parse_float(row[metric]) for row in rows]
        kept_y = [values[index] for index in kept_x]
        discarded_y = [values[index] for index in discarded_x]
        step_x, step_y, annotations = kept_running_best(rows, metric)

        ax.scatter(
            discarded_x,
            discarded_y,
            s=20,
            marker=markers[metric],
            color=color,
            alpha=0.22,
            label=f"{metric_label} discarded",
        )
        ax.scatter(
            kept_x,
            kept_y,
            s=58,
            marker=markers[metric],
            color=color,
            edgecolor="0.20",
            linewidth=0.6,
            alpha=0.95,
            label=f"{metric_label} kept",
            zorder=3,
        )
        ax.step(
            step_x,
            step_y,
            where="post",
            color=color,
            linewidth=2.3,
            alpha=0.72,
            label=f"{metric_label} kept running best",
        )

        for index, value, annotation in annotations:
            direction = 1 if metric_index == 0 else -1
            ax.text(
                index + 0.55,
                value + (label_offset * direction),
                textwrap.fill(annotation, width=34),
                rotation=28,
                color=color,
                fontsize=8,
                ha="left",
                va="bottom" if direction > 0 else "top",
            )

    ax.set_ylabel("Recall@10 / NDCG@10")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", ncols=2)
    ax.set_xlabel("Training run #")
    ax.set_xlim(-1, len(rows))
    ax.set_xticks(range(0, len(rows), max(1, len(rows) // 12)))

    save_path = output_path if overwrite else next_available_path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results.tsv"))
    parser.add_argument("--output", type=Path, default=Path("results_metrics.png"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --output if it already exists.")
    args = parser.parse_args()
    output_path = plot_results(args.input, args.output, overwrite=args.overwrite)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
