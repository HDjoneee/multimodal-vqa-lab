"""Build a reproducible VQA v2 pilot subset and fetch only its COCO images.

The four official VQA archives are expected in data/raw. No labels from the
validation set are used to construct the answer vocabulary.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "images"
MANIFEST = ROOT / "data" / "manifest.json"
FAMILIES = ("yes/no", "count", "color", "animal")
VOCAB_SIZES = {"yes/no": 2, "count": 11, "color": 12, "animal": 10}

ARCHIVES = {
    "train_annotations": "v2_Annotations_Train_mscoco.zip",
    "val_annotations": "v2_Annotations_Val_mscoco.zip",
    "train_questions": "v2_Questions_Train_mscoco.zip",
    "val_questions": "v2_Questions_Val_mscoco.zip",
}


def read_archive(kind: str) -> dict:
    archive = RAW / ARCHIVES[kind]
    if not archive.is_file():
        raise FileNotFoundError(f"Missing {archive}. Download it from https://visualqa.org/download")
    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if name.endswith(".json")]
        if len(members) != 1:
            raise ValueError(f"Expected one JSON in {archive}, found {members}")
        with zf.open(members[0]) as stream:
            return json.load(stream)


def joined_rows(split: str) -> list[dict]:
    annotations = read_archive(f"{split}_annotations")["annotations"]
    questions = read_archive(f"{split}_questions")["questions"]
    question_by_id = {q["question_id"]: q for q in questions}
    rows = []
    for ann in annotations:
        q = question_by_id[ann["question_id"]]
        assert q["image_id"] == ann["image_id"]
        image_id = ann["image_id"]
        image_split = "train2014" if split == "train" else "val2014"
        rows.append({
            "question_id": ann["question_id"],
            "image_id": image_id,
            "question": q["question"],
            "answer": ann["multiple_choice_answer"].lower().strip(),
            "answer_type": ann["answer_type"],
            "human_answers": [a["answer"].lower().strip() for a in ann["answers"]],
            "image": f"{image_split}/COCO_{image_split}_{image_id:012d}.jpg",
        })
    return rows


def question_family(row: dict) -> str | None:
    question = row["question"].lower().strip()
    if question.startswith(("what color", "what is the color")):
        return "color"
    if question.startswith("how many"):
        return "count"
    if row["answer_type"] == "other" and re.search(
        r"^(what animal\b|what (kind|type) of animal\b|what is (this|that|the) animal\?)",
        question,
    ):
        return "animal"
    if row["answer_type"] == "yes/no" and re.match(
        r"^(is|are|does|do|can|has|have|was|were|did)\b", question
    ):
        return "yes/no"
    return None


def sample_type(rows: list[dict], count_per_type: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for family in FAMILIES:
        group = [row for row in rows if row["question_family"] == family]
        rng.shuffle(group)
        if len(group) < count_per_type:
            raise ValueError(f"Only {len(group)} rows for {family}; wanted {count_per_type}")
        selected.extend(group[:count_per_type])
    rng.shuffle(selected)
    return selected


def build_manifest(train_per_type: int = 800, eval_per_type: int = 80,
                   vocab_sizes: dict[str, int] | None = None, seed: int = 42) -> dict:
    vocab_sizes = vocab_sizes or VOCAB_SIZES
    train_all = joined_rows("train")
    val_all = joined_rows("val")
    for row in train_all + val_all:
        row["question_family"] = question_family(row)
    train = sample_type(train_all, train_per_type, seed)

    # Image-level split within the official labeled validation set.
    val_image_ids = sorted({row["image_id"] for row in val_all})
    rng = random.Random(seed)
    rng.shuffle(val_image_ids)
    dev_ids = set(val_image_ids[:len(val_image_ids) // 2])
    dev = sample_type([r for r in val_all if r["image_id"] in dev_ids],
                      eval_per_type, seed + 1)
    test = sample_type([r for r in val_all if r["image_id"] not in dev_ids],
                       eval_per_type, seed + 2)
    assert not ({r["image_id"] for r in dev} & {r["image_id"] for r in test})

    vocabulary = []
    allowed_ids = {}
    answer_to_id = {}
    for family in FAMILIES:
        counts = Counter(row["answer"] for row in train if row["question_family"] == family)
        allowed_ids[family] = []
        for answer, _ in counts.most_common(vocab_sizes[family]):
            answer_to_id[(family, answer)] = len(vocabulary)
            allowed_ids[family].append(len(vocabulary))
            vocabulary.append(answer)
    for split_name, split_rows in (("train", train), ("dev", dev), ("test", test)):
        for row in split_rows:
            row["split"] = split_name
            row["label_id"] = answer_to_id.get((row["question_family"], row["answer"]), -1)

    stats = {}
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        covered = sum(row["label_id"] >= 0 for row in rows)
        stats[name] = {"questions": len(rows),
                       "images": len({row["image"] for row in rows}),
                       "majority_answer_coverage": covered / len(rows),
                       "question_families": dict(Counter(row["question_family"] for row in rows))}
    manifest = {
        "source": "VQA v2 real images, https://visualqa.org/download",
        "images_source": "COCO 2014, https://cocodataset.org/",
        "seed": seed,
        "sampling": "Equal numbers per question family; official train for training; official val split by image_id for dev/test.",
        "target": "multiple_choice_answer (most frequent human answer); classification exact match",
        "oov_policy": "Drop OOV train targets; keep OOV dev/test targets and count them incorrect.",
        "vocab_sizes": vocab_sizes,
        "vocabulary": vocabulary,
        "allowed_ids": allowed_ids,
        "stats": stats,
        "rows": train + dev + test,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def download_one(relpath: str) -> tuple[str, str | None]:
    target = IMAGES / relpath
    if target.is_file():
        return relpath, None
    target.parent.mkdir(parents=True, exist_ok=True)
    # COCO's HTTP endpoint works where its HTTPS certificate fails on this host.
    url = f"http://images.cocodataset.org/{relpath}"
    error = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            temp = target.with_suffix(".jpg.part")
            temp.write_bytes(response.content)
            with Image.open(temp) as image:
                image.verify()
            temp.replace(target)
            return relpath, None
        except Exception as exc:
            error = str(exc)
            time.sleep(0.5 * (attempt + 1))
    return relpath, error


def download_images(manifest: dict, workers: int = 12) -> None:
    paths = sorted({row["image"] for row in manifest["rows"]})
    missing = [path for path in paths if not (IMAGES / path).is_file()]
    print(f"COCO images required: {len(paths)}; missing: {len(missing)}", flush=True)
    if not missing:
        return
    failures = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, path) for path in missing]
        for future in as_completed(futures):
            name, error = future.result()
            done += 1
            if error:
                failures.append((name, error))
            if done % 100 == 0 or done == len(missing):
                print(f"Downloaded {done}/{len(missing)}; failures: {len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"Failed to download {len(failures)} images; first: {failures[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-type", type=int, default=800)
    parser.add_argument("--eval-per-type", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-images", action="store_true")
    arguments = parser.parse_args()
    data = build_manifest(arguments.train_per_type, arguments.eval_per_type,
                          VOCAB_SIZES, arguments.seed)
    print(json.dumps(data["stats"], ensure_ascii=False, indent=2))
    if arguments.download_images:
        download_images(data)
