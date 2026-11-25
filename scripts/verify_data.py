"""Frozen-dataset self-check for the three Top-1000 splits.

Run this before any experiment. It re-derives every headline number from the
raw official annotations and the three manifests, checks image completeness,
and verifies that the splits are disjoint and reproducible.

    python scripts/verify_data.py
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = DATA / "images"

MANIFESTS = {
    "train": DATA / "manifest_top1000.json",
    "dev": DATA / "manifest_top1000_dev.json",
    "test": DATA / "manifest_top1000_test.json",
}


def load_annotations(name: str) -> list[dict]:
    with zipfile.ZipFile(DATA / "raw" / name) as archive:
        member = next(m for m in archive.namelist() if m.endswith(".json"))
        return json.loads(archive.read(member))["annotations"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    vocabulary = json.loads(MANIFESTS["train"].read_text(encoding="utf-8"))["vocabulary"]
    vocab = set(vocabulary)
    val_annotations = load_annotations("v2_Annotations_Val_mscoco.zip")

    print(f"vocabulary: {len(vocabulary)} classes (sorted by official train frequency)")
    print()

    image_sets: dict[str, set[int]] = {}
    for split, path in MANIFESTS.items():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = manifest["rows"]
        image_ids = {row["image_id"] for row in rows}
        image_sets[split] = image_ids

        per_image = Counter(row["image_id"] for row in rows)
        counts = sorted(per_image.values())
        density = sum(counts) / len(counts)

        missing = [
            row["image"] for row in {r["image"]: r for r in rows}.values()
            if not (IMAGES / row["image"]).is_file()
        ]

        labels = {row["label_id"] for row in rows}
        families = Counter(row["question_family"] for row in rows)
        answer_types = Counter(row["answer_type"] for row in rows)
        total = len(rows)

        print(f"[{split}] {len(image_ids)} images, {total} QA, "
              f"density {density:.2f} (min {counts[0]}, max {counts[-1]})")
        print(f"  answer classes touched: {len(labels)} / {len(vocabulary)}")
        print(f"  answer_type: " + ", ".join(
            f"{k} {100 * v / total:.1f}%" for k, v in answer_types.most_common()))
        print(f"  question_family: " + ", ".join(
            f"{k} {100 * v / total:.1f}%" for k, v in families.most_common()))
        print(f"  missing images: {len(missing)}")

        if split in ("dev", "test"):
            selected = [a for a in val_annotations if a["image_id"] in image_ids]
            oov = [a for a in selected
                   if a["multiple_choice_answer"].lower().strip() not in vocab]
            print(f"  eval-time rows (unfiltered): {len(selected)} "
                  f"({len(selected) - len(oov)} in-vocab, {len(oov)} OOV)")
        print(f"  sha256: {sha256(path)[:16]}...")
        print()

    print("overlap check")
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        shared = image_sets[a] & image_sets[b]
        status = "OK" if not shared else f"LEAK {len(shared)}"
        print(f"  {a} ∩ {b}: {status}")

    pooled = image_sets["dev"] | image_sets["test"]
    print(f"  dev pool ∪ test pool = {len(pooled)} val images")


if __name__ == "__main__":
    main()
