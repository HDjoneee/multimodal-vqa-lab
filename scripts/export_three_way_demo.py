"""Export a public HTML gallery comparing the three aligned VQA training groups.

COCO images keep their original uploaders' rights, so the gallery links to the
official image host instead of republishing image bytes.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEMO = ROOT / "demo"
MODELS = ("text", "fusion", "finetune")


def main() -> None:
    summary = json.loads((RESULTS / "three_way_summary.json").read_text(encoding="utf-8"))
    predictions = json.loads((RESULTS / "three_way_predictions.json").read_text(encoding="utf-8"))
    selected = summary["selected_model"]

    gallery = []
    for answer_type in ("yes/no", "number", "other"):
        group = [row for row in predictions if row["answer_type"] == answer_type]
        for correct in (True, False):
            cases = [row for row in group
                     if row["models"][selected]["exact_match"] == correct]
            cases.sort(key=lambda row: row["models"][selected]["confidence"],
                       reverse=True)
            gallery.extend(cases[:3])

    # 额外收集"仅文本答对但图文融合答错"的题，直接展示图像输入的负向作用
    disagreements = [row for row in predictions
                     if row["models"]["text"]["exact_match"]
                     and not row["models"]["fusion"]["exact_match"]]
    disagreements.sort(key=lambda row: row["models"]["fusion"]["confidence"])
    gallery.extend(disagreements[:3])

    for row in gallery:
        row["image_url"] = f"http://images.cocodataset.org/{row['image']}"
    payload = json.dumps({"summary": summary, "samples": gallery, "models": list(MODELS)},
                         ensure_ascii=False).replace("</", "<\\/")
    template = (DEMO / "three_way_template.html").read_text(encoding="utf-8")
    html = template.replace("__VQA_DATA__", payload)
    assert "data:image/" not in html
    target = DEMO / "three_way.html"
    target.write_text(html, encoding="utf-8")
    print(f"Wrote {target} with {len(gallery)} genuine dev predictions")


if __name__ == "__main__":
    main()
