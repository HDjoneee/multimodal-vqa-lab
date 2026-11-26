"""Convert the readable cell source to ipynb and optionally execute every cell.

Execution uses one Python namespace, captures real stdout and generated plots,
and writes the notebook after each cell so interrupted runs remain inspectable.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "experiment_source.py"
TARGET = ROOT / "notebooks" / "v1-3200-4.ipynb"
IMAGES = ("data_distribution.png", "training_curves.png",
          "top1000_training_curves.png", "top1000_by_type.png",
          "three_way_training_curves.png", "three_way_by_type.png")


def parse_cells(source: str) -> list[dict]:
    cells = []
    kind = None
    lines = []

    def append():
        if kind is None:
            return
        if kind == "markdown":
            content = "\n".join(line[2:] if line.startswith("# ") else
                                line[1:] if line.startswith("#") else line
                                for line in lines).strip() + "\n"
        else:
            content = "\n".join(lines).strip() + "\n"
        cell = {"cell_type": kind,
                "id": hashlib.sha1(content.encode("utf-8")).hexdigest()[:12],
                "metadata": {}, "source": content}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)

    for line in source.splitlines():
        if line == "# %% [markdown]":
            append()
            kind, lines = "markdown", []
        elif line == "# %%":
            append()
            kind, lines = "code", []
        elif kind is not None:
            lines.append(line)
    append()
    return cells


def attach_images(notebook: dict) -> None:
    """Embed plots already sitting in results/ into the cells that reference them."""
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if any(output.get("output_type") == "display_data" for output in cell["outputs"]):
            continue
        for filename in IMAGES:
            if f'"{filename}"' not in cell["source"]:
                continue
            image_path = ROOT / "results" / filename
            if image_path.is_file():
                encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                cell["outputs"].append({"output_type": "display_data", "metadata": {},
                                        "data": {"image/png": encoded}})


def attach_log(notebook: dict, path: Path, title: str) -> None:
    """Append a captured run log as a fenced markdown cell, so the numbers stay readable."""
    text = path.read_text(encoding="utf-8").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return
    notebook["cells"].append({
        "cell_type": "markdown",
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
        "metadata": {},
        "source": f"## {title}\n\n```text\n" + "\n".join(lines) + "\n```\n",
    })


def write_notebook(notebook: dict) -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


class Tee(io.TextIOBase):
    def __init__(self, stream, capture):
        self.stream = stream
        self.capture = capture
    def write(self, text):
        self.stream.write(text)
        self.capture.write(text)
        self.stream.flush()
        return len(text)
    def flush(self):
        self.stream.flush()
        self.capture.flush()


def execute(notebook: dict) -> None:
    os.chdir(ROOT)
    namespace = {"__name__": "__main__", "__builtins__": __builtins__}
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        cell["execution_count"] = count
        out, err = io.StringIO(), io.StringIO()
        print(f"\n--- Notebook cell {count} ---", flush=True)
        try:
            with contextlib.redirect_stdout(Tee(sys.stdout, out)), \
                 contextlib.redirect_stderr(Tee(sys.stderr, err)):
                exec(compile(cell["source"], f"<notebook cell {count}>", "exec"), namespace)
        except Exception:
            error = traceback.format_exc()
            cell["outputs"].append({"output_type": "error", "ename": "ExecutionError",
                                    "evalue": error.splitlines()[-1], "traceback": error.splitlines()})
            write_notebook(notebook)
            raise
        if out.getvalue():
            cell["outputs"].append({"output_type": "stream", "name": "stdout",
                                    "text": out.getvalue()})
        if err.getvalue():
            cell["outputs"].append({"output_type": "stream", "name": "stderr",
                                    "text": err.getvalue()})
        for filename in IMAGES:
            if f'"{filename}"' in cell["source"]:
                image_path = ROOT / "results" / filename
                if image_path.is_file():
                    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                    cell["outputs"].append({"output_type": "display_data", "metadata": {},
                                            "data": {"image/png": encoded}})
        write_notebook(notebook)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE,
                        help="Cell-marked Python source (default: historical pilot)")
    parser.add_argument("--output", type=Path, default=TARGET,
                        help="Notebook path (default: historical pilot)")
    parser.add_argument("--attach-log", type=Path, default=None,
                        help="Append this run log to the notebook as a markdown cell")
    parser.add_argument("--log-title", default="运行日志",
                        help="Heading for the appended log cell")
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    TARGET = args.output.resolve()
    notebook = {"cells": parse_cells(SOURCE.read_text(encoding="utf-8")),
                "metadata": {"kernelspec": {"display_name": "Python 3",
                                             "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    if args.execute:
        execute(notebook)
    else:
        attach_images(notebook)
    if args.attach_log:
        attach_log(notebook, args.attach_log.resolve(), args.log_title)
    write_notebook(notebook)
    print(TARGET)
