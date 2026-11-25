"""Export a public HTML gallery from actual Top-1000 development predictions."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEMO = ROOT / "demo"


def main() -> None:
    summary = json.loads((RESULTS / "top1000_summary.json").read_text(encoding="utf-8"))
    predictions = json.loads((RESULTS / "top1000_predictions.json").read_text(encoding="utf-8"))
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
    for row in gallery:
        row["image_url"] = f"http://images.cocodataset.org/{row['image']}"
    payload = json.dumps({"summary": summary, "samples": gallery},
                         ensure_ascii=False).replace("</", "<\\/")
    template = (DEMO / "top1000_template.html").read_text(encoding="utf-8")
    html = template.replace("__VQA_DATA__", payload)
    assert "data:image/" not in html
    target = DEMO / "top1000.html"
    target.write_text(html, encoding="utf-8")
    print(f"Wrote {target} with {len(gallery)} genuine dev predictions")


if __name__ == "__main__":
    main()
