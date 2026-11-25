"""Build a reproducible VQA v2 subset and fetch only its COCO images.

The four official VQA archives are expected in data/raw. No labels from the
validation set are used to construct the answer vocabulary.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("VQA_DATA_ROOT", ROOT / "data")).expanduser().resolve()
RAW = DATA / "raw"
IMAGES = DATA / "images"
MANIFEST = DATA / "manifest.json"
TOP1000_MANIFEST = DATA / "manifest_top1000.json"
TOP1000_DEV_MANIFEST = DATA / "manifest_top1000_dev.json"
TOP1000_TEST_MANIFEST = DATA / "manifest_top1000_test.json"
FAMILIES = ("yes/no", "count", "color", "animal")
VOCAB_SIZES = {"yes/no": 2, "count": 11, "color": 12, "animal": 10}
DEFAULT_SEED = 42
TOP_ANSWERS = 1000
MIN_QA_PER_IMAGE = 6
# Density-stratified sampling: every image with at least MIN_QA_PER_IMAGE
# in-vocabulary QA is eligible, and each density stratum contributes the same
# sampling rate, so the subset keeps the natural QA-per-image distribution.
TOP1000_TRAIN_IMAGES = 2000
TOP1000_DEV_IMAGES = 800
TOP1000_TEST_IMAGES = 800

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


def validation_pools(rows: list[dict], seed: int) -> tuple[set[int], set[int]]:
    """Split the official val images into disjoint dev and test image pools."""
    image_ids = sorted({row["image_id"] for row in rows})
    random.Random(seed).shuffle(image_ids)
    half = len(image_ids) // 2
    return set(image_ids[:half]), set(image_ids[half:])


def validation_dev_ids(rows: list[dict], seed: int) -> set[int]:
    dev_ids, _ = validation_pools(rows, seed)
    return dev_ids


def build_manifest(train_per_type: int = 800, eval_per_type: int = 80,
                   vocab_sizes: dict[str, int] | None = None, seed: int = DEFAULT_SEED) -> dict:
    vocab_sizes = vocab_sizes or VOCAB_SIZES
    train_all = joined_rows("train")
    val_all = joined_rows("val")
    for row in train_all + val_all:
        row["question_family"] = question_family(row)
    train = sample_type(train_all, train_per_type, seed)

    # Image-level split within the official labeled validation set.
    dev_ids = validation_dev_ids(val_all, seed)
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


def allocate_strata(sizes: dict[int, int], target_images: int) -> dict[int, int]:
    """Split an image budget across density strata at a uniform rate (Hare quota)."""
    eligible = sum(sizes.values())
    if target_images > eligible:
        raise ValueError(f"Cannot draw {target_images} images from {eligible} eligible images")
    rate = target_images / eligible
    quotas: dict[int, int] = {}
    remainders = []
    for qa_count in sorted(sizes):
        exact = sizes[qa_count] * rate
        base = int(exact)
        quotas[qa_count] = base
        remainders.append((exact - base, qa_count))
    short = target_images - sum(quotas.values())
    for _, qa_count in sorted(remainders, key=lambda item: (-item[0], item[1]))[:short]:
        quotas[qa_count] += 1
    return quotas


def sample_high_frequency_images(
    rows: list[dict], answer_to_id: dict[str, int], seed: int, *,
    split: str, target_images: int,
) -> tuple[list[dict], dict]:
    by_image = defaultdict(list)
    for row in rows:
        if row["answer"] in answer_to_id:
            by_image[row["image_id"]].append(row)

    by_qa_count: dict[int, list[int]] = defaultdict(list)
    for image_id, group in by_image.items():
        if len(group) >= MIN_QA_PER_IMAGE:
            by_qa_count[len(group)].append(image_id)

    sizes = {count: len(ids) for count, ids in by_qa_count.items()}
    quotas = allocate_strata(sizes, target_images)

    rng = random.Random(seed)
    selected = []
    for qa_count in sorted(quotas):
        quota = quotas[qa_count]
        if quota == 0:
            continue
        for image_id in rng.sample(sorted(by_qa_count[qa_count]), quota):
            for row in sorted(by_image[image_id], key=lambda row: row["question_id"]):
                selected.append({
                    **row,
                    "split": split,
                    "question_family": question_family(row) or "other",
                    "label_id": answer_to_id[row["answer"]],
                })

    selected_images = sum(quotas.values())
    densities = [count for count, quota in quotas.items() for _ in range(quota)]
    return selected, {
        "questions": len(selected),
        "images": selected_images,
        "eligible_images": sum(sizes.values()),
        "sampling_rate": round(target_images / sum(sizes.values()), 4),
        "density": {
            "min": min(densities),
            "max": max(densities),
            "mean": round(len(selected) / selected_images, 2),
        },
        "eligible_images_by_qa_count": sizes,
        "selected_images_by_qa_count": {count: quota for count, quota in sorted(quotas.items()) if quota},
        "answer_types": dict(Counter(row["answer_type"] for row in selected)),
        "question_families": dict(Counter(row["question_family"] for row in selected)),
        "unique_answers": len({row["label_id"] for row in selected}),
    }


def build_top1000_manifest(seed: int = DEFAULT_SEED) -> dict:
    train_all = joined_rows("train")
    counts = Counter(row["answer"] for row in train_all)
    vocabulary = sorted(counts, key=lambda answer: (-counts[answer], answer))[:TOP_ANSWERS]
    if len(vocabulary) != TOP_ANSWERS:
        raise ValueError(f"Expected {TOP_ANSWERS} training answers; found {len(vocabulary)}")
    answer_to_id = {answer: index for index, answer in enumerate(vocabulary)}
    train, train_stats = sample_high_frequency_images(
        train_all, answer_to_id, seed, split="train", target_images=TOP1000_TRAIN_IMAGES,
    )

    manifest = {
        "source": "VQA v2 real images, https://visualqa.org/download",
        "images_source": "COCO 2014, https://cocodataset.org/",
        "seed": seed,
        "sampling": "Filter official train to its Top-1000 answers; sample whole image "
                    "groups from every density stratum with >=6 in-vocabulary QA at a "
                    "uniform rate, so the natural QA-per-image spread is preserved.",
        "target": "multiple_choice_answer (most frequent human answer)",
        "vocabulary_source": "Full official train; descending count, then alphabetical order.",
        "min_qa_per_image": MIN_QA_PER_IMAGE,
        "vocabulary": vocabulary,
        "stats": {"train": train_stats},
        "rows": train,
    }
    TOP1000_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    TOP1000_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_top1000_dev_manifest(seed: int = DEFAULT_SEED) -> dict:
    if not TOP1000_MANIFEST.is_file():
        raise FileNotFoundError("Build the training vocabulary first: "
                                "python scripts/prepare_data.py --sampling top1000")
    train_manifest = json.loads(TOP1000_MANIFEST.read_text(encoding="utf-8"))
    vocabulary = train_manifest["vocabulary"]
    answer_to_id = {answer: index for index, answer in enumerate(vocabulary)}
    if len(vocabulary) != TOP_ANSWERS or len(answer_to_id) != TOP_ANSWERS:
        raise ValueError(f"Expected {TOP_ANSWERS} distinct training answers")

    val_all = joined_rows("val")
    # Keep the same dev/test image pools as the default pilot.
    dev_ids, test_ids = validation_pools(val_all, DEFAULT_SEED)
    dev, dev_stats = sample_high_frequency_images(
        [row for row in val_all if row["image_id"] in dev_ids], answer_to_id, seed,
        split="dev", target_images=TOP1000_DEV_IMAGES,
    )
    manifest = {
        "source": train_manifest["source"],
        "images_source": train_manifest["images_source"],
        "seed": seed,
        "val_image_split_seed": DEFAULT_SEED,
        "val_image_pools": {"dev": len(dev_ids), "test": len(test_ids)},
        "sampling": "Filter the official val dev image pool to the training Top-1000 "
                    "answers; sample whole image groups from every density stratum with "
                    ">=6 in-vocabulary QA at a uniform rate.",
        "target": train_manifest["target"],
        "vocabulary_source": TOP1000_MANIFEST.name,
        "min_qa_per_image": MIN_QA_PER_IMAGE,
        "vocabulary": vocabulary,
        "stats": {"dev": dev_stats},
        "rows": dev,
    }
    TOP1000_DEV_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def build_top1000_test_manifest(seed: int = DEFAULT_SEED) -> dict:
    if not TOP1000_MANIFEST.is_file():
        raise FileNotFoundError("Build the training vocabulary first: "
                                "python scripts/prepare_data.py --sampling top1000")
    train_manifest = json.loads(TOP1000_MANIFEST.read_text(encoding="utf-8"))
    vocabulary = train_manifest["vocabulary"]
    answer_to_id = {answer: index for index, answer in enumerate(vocabulary)}
    if len(vocabulary) != TOP_ANSWERS or len(answer_to_id) != TOP_ANSWERS:
        raise ValueError(f"Expected {TOP_ANSWERS} distinct training answers")

    val_all = joined_rows("val")
    _, test_ids = validation_pools(val_all, DEFAULT_SEED)
    test, test_stats = sample_high_frequency_images(
        [row for row in val_all if row["image_id"] in test_ids], answer_to_id, seed,
        split="test", target_images=TOP1000_TEST_IMAGES,
    )
    dev_ids, _ = validation_pools(val_all, DEFAULT_SEED)
    overlap = dev_ids & {row["image_id"] for row in test}
    assert not overlap, f"test leaked {len(overlap)} dev images"
    manifest = {
        "source": train_manifest["source"],
        "images_source": train_manifest["images_source"],
        "seed": seed,
        "val_image_split_seed": DEFAULT_SEED,
        "sampling": "Filter the official val test image pool to the training Top-1000 "
                    "answers; sample whole image groups from every density stratum with "
                    ">=6 in-vocabulary QA at a uniform rate, matching the dev design.",
        "target": train_manifest["target"],
        "vocabulary_source": TOP1000_MANIFEST.name,
        "min_qa_per_image": MIN_QA_PER_IMAGE,
        "vocabulary": vocabulary,
        "stats": {"test": test_stats},
        "rows": test,
    }
    TOP1000_TEST_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
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
    parser.add_argument("--sampling", choices=("pilot", "top1000", "top1000-dev", "top1000-test"),
                        default="top1000",
                        help="Choose Top-1000 train (default), dev, test, or the historical pilot")
    parser.add_argument("--train-per-type", type=int, help="Historical pilot only (default: 800)")
    parser.add_argument("--eval-per-type", type=int, help="Historical pilot only (default: 80)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--download-images", action="store_true")
    arguments = parser.parse_args()
    if arguments.sampling != "pilot" and (
        arguments.train_per_type is not None or arguments.eval_per_type is not None
    ):
        parser.error("--train-per-type and --eval-per-type require --sampling pilot")
    if arguments.sampling == "top1000":
        data = build_top1000_manifest(arguments.seed)
        print(TOP1000_MANIFEST)
        stats = data["stats"]["train"]
        print(f"train: {stats['images']} images, {stats['questions']} QA "
              f"from {stats['eligible_images']} eligible training images")
    elif arguments.sampling == "top1000-dev":
        data = build_top1000_dev_manifest(arguments.seed)
        print(TOP1000_DEV_MANIFEST)
        stats = data["stats"]["dev"]
        print(f"dev: {stats['images']} images, {stats['questions']} QA "
              f"from {stats['eligible_images']} eligible development images")
    elif arguments.sampling == "top1000-test":
        data = build_top1000_test_manifest(arguments.seed)
        print(TOP1000_TEST_MANIFEST)
        stats = data["stats"]["test"]
        print(f"test: {stats['images']} images, {stats['questions']} QA "
              f"from {stats['eligible_images']} eligible test images")
    else:
        pilot_options = {"seed": arguments.seed}
        if arguments.train_per_type is not None:
            pilot_options["train_per_type"] = arguments.train_per_type
        if arguments.eval_per_type is not None:
            pilot_options["eval_per_type"] = arguments.eval_per_type
        data = build_manifest(**pilot_options)
        print(json.dumps(data["stats"], ensure_ascii=False, indent=2))
    if arguments.download_images:
        download_images(data)
