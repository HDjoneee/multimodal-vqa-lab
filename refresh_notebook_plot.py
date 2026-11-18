"""Refresh the data plot and its retained Notebook output without retraining.

Use after a presentation-only edit to the data plot cell. Training and metrics
are not recomputed; their previously executed outputs remain intact.
"""

import base64
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_notebook import ROOT, SOURCE, TARGET, parse_cells


new_cells = parse_cells(SOURCE.read_text(encoding="utf-8"))
notebook = json.loads(TARGET.read_text(encoding="utf-8"))
if len(new_cells) != len(notebook["cells"]):
    raise ValueError("Notebook cell structure changed; run the full Notebook")

manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
splits = {name: [row for row in manifest["rows"] if row["split"] == name]
          for name in ("train", "dev", "test")}
result_dir = ROOT / "results"
plot_index = next(i for i, cell in enumerate(new_cells)
                  if cell["cell_type"] == "code" and "data_distribution.png" in cell["source"])
exec(new_cells[plot_index]["source"], {
    "manifest": manifest, "splits": splits, "Counter": Counter, "plt": plt,
    "RESULTS": result_dir, "show_saved_image": lambda filename: None,
})

for old, new in zip(notebook["cells"], new_cells):
    if old["cell_type"] != new["cell_type"]:
        raise ValueError("Notebook cell kinds changed; run the full Notebook")
    old["source"] = new["source"]
    old["id"] = new["id"]

encoded = base64.b64encode((result_dir / "data_distribution.png").read_bytes()).decode("ascii")
plot_outputs = notebook["cells"][plot_index]["outputs"]
image_outputs = [item for item in plot_outputs if item["output_type"] == "display_data"
                 and "image/png" in item["data"]]
if len(image_outputs) != 1:
    raise ValueError("Expected one retained distribution figure")
image_outputs[0]["data"]["image/png"] = encoded
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("Updated data figure and Notebook output; training results unchanged")
