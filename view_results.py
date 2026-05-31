#!/usr/bin/env python3
"""Interactive terminal viewer for the local auto-tiger results.tsv ledger."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import msvcrt
except ModuleNotFoundError:  # pragma: no cover - exercised on POSIX only.
    msvcrt = None

if os.name != "nt":  # pragma: no cover - exercised on POSIX only.
    import select
    import termios
    import tty
else:  # pragma: no cover - exercised on Windows only.
    select = None
    termios = None
    tty = None


DEFAULT_COLUMNS = [
    "run_id",
    "status",
    "recall_at_10",
    "ndcg_at_10",
    "recall_at_5",
    "ndcg_at_5",
    "decision",
    "description",
]
SORT_COLUMNS = ["timestamp", "recall_at_10", "ndcg_at_10", "recall_at_5", "run_id"]
NUMERIC_COLUMNS = {"recall_at_5", "ndcg_at_5", "recall_at_10", "ndcg_at_10", "peak_vram_gb"}
HEADER_LABELS = {
    "run_id": "run_id",
    "status": "status",
    "recall_at_10": "R@10",
    "ndcg_at_10": "NDCG@10",
    "recall_at_5": "R@5",
    "ndcg_at_5": "NDCG@5",
    "decision": "decision",
    "description": "description",
}
TABLE_MIN_HEIGHT = 5
DETAIL_MIN_HEIGHT = 8


class Ansi:
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    INVERT = "\x1b[7m"
    CLEAR = "\x1b[2J\x1b[H"
    HIDE_CURSOR = "\x1b[?25l"
    SHOW_CURSOR = "\x1b[?25h"
    ALT_SCREEN = "\x1b[?1049h"
    MAIN_SCREEN = "\x1b[?1049l"


@contextmanager
def terminal_mode(enabled: bool) -> Iterable[None]:
    if not enabled:
        yield
        return

    original_settings = None
    if os.name != "nt" and sys.stdin.isatty():
        original_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
    try:
        sys.stdout.write(Ansi.ALT_SCREEN + Ansi.HIDE_CURSOR)
        sys.stdout.flush()
        yield
    finally:
        if original_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_settings)
        sys.stdout.write(Ansi.SHOW_CURSOR + Ansi.MAIN_SCREEN + Ansi.RESET)
        sys.stdout.flush()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = [
            {key: (value or "") for key, value in row.items() if key is not None}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]
    return fieldnames, rows


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str) and "=" in value:
            try:
                return float(value.rsplit("=", 1)[1])
            except ValueError:
                pass
    return float("-inf")


def parse_timestamp(value: str) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def sort_value(row: dict[str, str], column: str) -> object:
    if column == "timestamp":
        return parse_timestamp(row.get(column, ""))
    if column in NUMERIC_COLUMNS:
        return parse_float(row.get(column, ""))
    return row.get(column, "").lower()


def visible_rows(
    rows: list[dict[str, str]],
    query: str,
    sort_column: str,
    descending: bool,
) -> list[dict[str, str]]:
    if query:
        terms = [term.lower() for term in query.split() if term.strip()]
        filtered = [
            row
            for row in rows
            if all(term in "\t".join(row.values()).lower() for term in terms)
        ]
    else:
        filtered = rows[:]
    return sorted(filtered, key=lambda row: sort_value(row, sort_column), reverse=descending)


def clean(value: object) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()


def fit(value: object, width: int) -> str:
    text = clean(value)
    if width <= 0:
        return ""
    if len(text) <= width:
        return text.ljust(width)
    if width == 1:
        return "."
    return text[: width - 1] + "."


def wrap(text: str, width: int) -> list[str]:
    if width <= 0:
        return [""]
    words = clean(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        if len(word) > width:
            if current:
                lines.append(current)
                current = ""
            for start in range(0, len(word), width):
                lines.append(word[start : start + width])
            continue
        next_line = f"{current} {word}".strip()
        if len(next_line) <= width:
            current = next_line
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def metric_text(row: dict[str, str], column: str) -> str:
    value = row.get(column, "")
    if not value:
        return ""
    number = parse_float(value)
    if number == float("-inf"):
        return value
    return f"{number:.6f}"


def decide_columns(width: int, fieldnames: list[str]) -> list[tuple[str, int]]:
    preferred = [column for column in DEFAULT_COLUMNS if column in fieldnames]
    if not preferred:
        preferred = fieldnames[:]

    fixed = [
        ("run_id", 32),
        ("status", 10),
        ("recall_at_10", 9),
        ("ndcg_at_10", 9),
        ("recall_at_5", 9),
        ("ndcg_at_5", 9),
        ("decision", 20),
    ]
    columns: list[tuple[str, int]] = []
    remaining = max(width - 3, 40)
    for column, desired in fixed:
        if column in preferred and remaining >= 10:
            column_width = min(desired, max(8, remaining // 2))
            columns.append((column, column_width))
            remaining -= column_width + 1

    if "description" in preferred and remaining >= 12:
        columns.append(("description", max(12, remaining)))
    elif remaining >= 12:
        for column in preferred:
            if column not in {name for name, _ in columns}:
                columns.append((column, max(12, remaining)))
                break

    while sum(size + 1 for _, size in columns) > width - 1 and len(columns) > 2:
        columns.pop()
    return columns


def border(width: int, char: str = "-") -> str:
    return char * max(width, 0)


def render(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    filtered: list[dict[str, str]],
    selected: int,
    offset: int,
    query: str,
    sort_column: str,
    descending: bool,
    message: str,
) -> str:
    size = shutil.get_terminal_size(fallback=(120, 36))
    width = max(size.columns, 60)
    height = max(size.lines, 20)
    detail_height = max(DETAIL_MIN_HEIGHT, min(12, height // 3))
    table_height = max(TABLE_MIN_HEIGHT, height - detail_height - 6)
    columns = decide_columns(width, fieldnames)

    title = f" auto-tiger results.tsv viewer  {path} "
    sort_marker = "desc" if descending else "asc"
    status = f"{len(filtered)}/{len(rows)} rows  sort {sort_column} {sort_marker}"
    if query:
        status += f"  filter: {query}"

    lines = [
        Ansi.CLEAR + Ansi.BOLD + fit(title, width) + Ansi.RESET,
        fit(status, width),
        border(width),
    ]

    if not fieldnames:
        lines.append(fit("No TSV header found. Use log_experiment.py tsv to create results.tsv.", width))
    elif not filtered:
        lines.append(fit("No rows match the current filter.", width))
    else:
        header = " ".join(fit(HEADER_LABELS.get(name, name), column_width) for name, column_width in columns)
        lines.append(Ansi.BOLD + fit(header, width) + Ansi.RESET)
        lines.append(border(width, "."))

        visible = filtered[offset : offset + table_height]
        for index, row in enumerate(visible, start=offset):
            parts = []
            for name, column_width in columns:
                value = metric_text(row, name) if name in NUMERIC_COLUMNS else row.get(name, "")
                parts.append(fit(value, column_width))
            line = fit(" ".join(parts), width)
            if index == selected:
                line = Ansi.INVERT + line + Ansi.RESET
            lines.append(line)

    while len(lines) < table_height + 5:
        lines.append("")

    lines.append(border(width))
    if filtered:
        row = filtered[selected]
        detail_keys = [
            "run_id",
            "stage",
            "status",
            "primary_metric",
            "recall_at_5",
            "ndcg_at_5",
            "recall_at_10",
            "ndcg_at_10",
            "peak_vram_gb",
            "decision",
            "timestamp",
            "artifact_uri",
            "description",
        ]
        detail_lines = []
        for key in detail_keys:
            if key in row and row.get(key, ""):
                label = f"{key}: "
                wrapped = wrap(row[key], max(1, width - len(label)))
                detail_lines.append(label + wrapped[0])
                detail_lines.extend(" " * len(label) + line for line in wrapped[1:])
        for line in detail_lines[:detail_height]:
            lines.append(fit(line, width))
    else:
        lines.append(fit("Detail pane is empty.", width))

    while len(lines) < height - 2:
        lines.append("")

    help_text = "up/down move  PgUp/PgDn page  s sort  d dir  / filter  c clear  r reload  q quit"
    if message:
        help_text = message
    lines.append(Ansi.DIM + fit(help_text, width) + Ansi.RESET)
    return "\n".join(lines[:height])


def prompt(prompt_text: str, initial: str = "") -> str:
    if os.name != "nt" and sys.stdin.isatty():
        settings = termios.tcgetattr(sys.stdin.fileno())
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, settings)
    sys.stdout.write(Ansi.SHOW_CURSOR + "\n" + prompt_text)
    sys.stdout.flush()
    try:
        entered = input()
    except EOFError:
        entered = initial
    finally:
        if os.name != "nt" and sys.stdin.isatty():
            tty.setcbreak(sys.stdin.fileno())
        sys.stdout.write(Ansi.HIDE_CURSOR)
        sys.stdout.flush()
    return entered


def read_key() -> str:
    if msvcrt is not None:
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            return {
                "H": "up",
                "P": "down",
                "K": "left",
                "M": "right",
                "I": "page_up",
                "Q": "page_down",
                "G": "home",
                "O": "end",
            }.get(code, "")
        if key == "\r":
            return "enter"
        if key == "\x1b":
            return "escape"
        return key

    key = sys.stdin.read(1)
    if key == "\x1b":
        if select.select([sys.stdin], [], [], 0.02)[0]:
            sequence = sys.stdin.read(2)
            return {
                "[A": "up",
                "[B": "down",
                "[D": "left",
                "[C": "right",
                "[5": "page_up",
                "[6": "page_down",
                "[H": "home",
                "[F": "end",
            }.get(sequence, "escape")
        return "escape"
    if key in ("\n", "\r"):
        return "enter"
    return key


def clamp_state(selected: int, offset: int, count: int, page_size: int) -> tuple[int, int]:
    if count <= 0:
        return 0, 0
    selected = min(max(selected, 0), count - 1)
    if selected < offset:
        offset = selected
    if selected >= offset + page_size:
        offset = selected - page_size + 1
    max_offset = max(0, count - page_size)
    offset = min(max(offset, 0), max_offset)
    return selected, offset


def run_viewer(path: Path) -> int:
    sort_index = 0
    sort_column = SORT_COLUMNS[sort_index]
    descending = True
    query = ""
    selected = 0
    offset = 0
    message = ""
    fieldnames, rows = read_rows(path)

    with terminal_mode(True):
        while True:
            filtered = visible_rows(rows, query, sort_column, descending)
            page_size = max(TABLE_MIN_HEIGHT, shutil.get_terminal_size(fallback=(120, 36)).lines - 18)
            selected, offset = clamp_state(selected, offset, len(filtered), page_size)
            sys.stdout.write(
                render(
                    path,
                    fieldnames,
                    rows,
                    filtered,
                    selected,
                    offset,
                    query,
                    sort_column,
                    descending,
                    message,
                )
            )
            sys.stdout.flush()
            message = ""

            key = read_key()
            if key in {"q", "Q", "escape"}:
                return 0
            if key in {"up", "k", "K"}:
                selected -= 1
            elif key in {"down", "j", "J"}:
                selected += 1
            elif key == "page_up":
                selected -= page_size
            elif key == "page_down":
                selected += page_size
            elif key in {"home", "g"}:
                selected = 0
            elif key in {"end", "G"}:
                selected = max(0, len(filtered) - 1)
            elif key in {"s", "S"}:
                sort_index = (sort_index + 1) % len(SORT_COLUMNS)
                sort_column = SORT_COLUMNS[sort_index]
                selected = 0
                offset = 0
            elif key in {"d", "D"}:
                descending = not descending
                selected = 0
                offset = 0
            elif key == "/":
                query = prompt("filter> ", query).strip()
                selected = 0
                offset = 0
            elif key in {"r", "R"}:
                fieldnames, rows = read_rows(path)
                message = f"reloaded {len(rows)} rows from {path}"
            elif key in {"c", "C"}:
                query = ""
                selected = 0
                offset = 0


def print_once(path: Path) -> int:
    fieldnames, rows = read_rows(path)
    filtered = visible_rows(rows, "", "timestamp", True)
    output = render(path, fieldnames, rows, filtered, 0, 0, "", "timestamp", True, "")
    plain = (
        output.replace(Ansi.CLEAR, "")
        .replace(Ansi.BOLD, "")
        .replace(Ansi.DIM, "")
        .replace(Ansi.INVERT, "")
        .replace(Ansi.RESET, "")
    )
    print(plain)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("results.tsv"), help="TSV ledger to view.")
    parser.add_argument("--once", action="store_true", help="Render one non-interactive screen.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.once or not sys.stdin.isatty():
        return print_once(args.path)
    return run_viewer(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
