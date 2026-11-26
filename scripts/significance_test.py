"""Paired significance tests for the three aligned groups.

Both groups are evaluated on the same development questions, so the
comparison is paired. Two tests are reported:

1. McNemar's test on the discordant pairs (does A beat B more often than
   the reverse), reported as a chi-square with 1 degree of freedom.
2. Image-clustered bootstrap: questions from the same image are resampled
   as a block, because a single image contributes several questions and
   those questions are not independent.

Run: python scripts/significance_test.py
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results" / "three_way_predictions.json"
OUTPUT = ROOT / "results" / "significance.json"

GROUPS = ("text", "fusion", "finetune")
PAIRS = (("fusion", "text"), ("fusion", "finetune"), ("text", "finetune"))
BOOTSTRAP_ROUNDS = 10000
SEED = 42


def mcnemar(rows: list[dict], better: str, worse: str) -> dict:
    """Count discordant pairs and return McNemar's chi-square (no continuity correction)."""
    better_only = sum(1 for r in rows
                      if r["models"][better]["exact_match"] and not r["models"][worse]["exact_match"])
    worse_only = sum(1 for r in rows
                     if r["models"][worse]["exact_match"] and not r["models"][better]["exact_match"])
    discordant = better_only + worse_only
    if discordant == 0:
        chi2, p_value = 0.0, 1.0
    else:
        chi2 = (better_only - worse_only) ** 2 / discordant
        # Two-sided p from the chi-square(1) survival function: erfc(sqrt(x/2)).
        p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "better_only_correct": better_only,
        "worse_only_correct": worse_only,
        "discordant": discordant,
        "chi2": chi2,
        "p_value": p_value,
    }


def bootstrap_ci(rows: list[dict], better: str, worse: str, rounds: int, seed: int) -> dict:
    """Resample whole images, then recompute the exact-match difference."""
    by_image: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    image_ids = sorted(by_image)
    deltas: list[float] = []
    rng = random.Random(seed)
    for _ in range(rounds):
        sample = [rng.choice(image_ids) for _ in image_ids]
        correct = {better: 0, worse: 0}
        total = 0
        for image_id in sample:
            for row in by_image[image_id]:
                total += 1
                for group in correct:
                    correct[group] += int(row["models"][group]["exact_match"])
        deltas.append((correct[better] - correct[worse]) / total)
    deltas.sort()
    mean = sum(deltas) / len(deltas)
    lower = deltas[int(0.025 * len(deltas))]
    upper = deltas[int(0.975 * len(deltas))]
    return {"mean": mean, "ci95_low": lower, "ci95_high": upper,
            "excludes_zero": lower > 0 or upper < 0}


def main() -> None:
    all_rows = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    rows = [row for row in all_rows if row["label_id"] >= 0]
    print(f"全部 {len(all_rows)} 题，词表内 {len(rows)} 题，图片 {len({r['image_id'] for r in rows})} 张")

    report = {
        "filtered_questions": len(rows),
        "images": len({row["image_id"] for row in rows}),
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "accuracy": {group: sum(r["models"][group]["exact_match"] for r in rows) / len(rows)
                     for group in GROUPS},
        "pairs": {},
    }

    for group in GROUPS:
        print(f"  {group:9s} {100 * report['accuracy'][group]:.2f}%")

    print()
    for better, worse in PAIRS:
        m = mcnemar(rows, better, worse)
        b = bootstrap_ci(rows, better, worse, BOOTSTRAP_ROUNDS, SEED)
        delta = report["accuracy"][better] - report["accuracy"][worse]
        report["pairs"][f"{better}_vs_{worse}"] = {
            "difference": delta,
            "mcnemar": m,
            "bootstrap": b,
        }
        verdict = "显著" if m["p_value"] < 0.05 else "不显著"
        print(f"{better:9s} vs {worse:9s} 差值 {100 * delta:+.2f} pt  "
              f"McNemar chi2={m['chi2']:.1f} p={m['p_value']:.2e} ({verdict})  "
              f"聚类 bootstrap 95% CI [{100 * b['ci95_low']:+.2f}, {100 * b['ci95_high']:+.2f}]")

    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
