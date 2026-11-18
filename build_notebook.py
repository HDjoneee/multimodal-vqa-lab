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


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "experiment_source.py"
TARGET = ROOT / "VQA实验.ipynb"


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


def write_notebook(notebook: dict) -> None:
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
        for filename in ("data_distribution.png", "training_curves.png"):
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
    args = parser.parse_args()
    notebook = {"cells": parse_cells(SOURCE.read_text(encoding="utf-8")),
                "metadata": {"kernelspec": {"display_name": "Python 3",
                                             "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    write_notebook(notebook)
    if args.execute:
        execute(notebook)
    print(TARGET)
