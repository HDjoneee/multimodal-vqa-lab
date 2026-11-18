"""Build a public HTML demo that references COCO images without bundling them.

COCO images have individual rights held by their original uploaders. This
export keeps the experiment's genuine predictions but does not republish image
bytes in the repository. The local Notebook still creates an offline demo.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FAMILIES = ("yes/no", "count", "color", "animal")


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    predictions = json.loads((RESULTS / "predictions.json").read_text(encoding="utf-8"))
    selected = summary["selected_model"]
    gallery = []
    for family in FAMILIES:
        group = [row for row in predictions if row["question_family"] == family]
        good = [row for row in group if row["models"][selected]["correct"]]
        bad = [row for row in group if not row["models"][selected]["correct"]]
        gallery.extend(good[:2] + bad[:2])
    for row in gallery:
        row["image_data"] = f"http://images.cocodataset.org/{row['image']}"
    data = {"samples": gallery, "summary": summary}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    template = (ROOT / "demo_template.html").read_text(encoding="utf-8")
    html = template.replace("__VQA_DATA__", payload)
    assert "data:image/jpeg;base64," not in html
    (ROOT / "demo.html").write_text(html, encoding="utf-8")
    print(f"Exported public demo.html with {len(gallery)} test cases and external COCO image URLs")


if __name__ == "__main__":
    main()
