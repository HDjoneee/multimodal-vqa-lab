"""Apply CSS/HTML template edits while preserving genuine model outputs."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
html_path = ROOT / "demo.html"
template_path = ROOT / "demo_template.html"
old = html_path.read_text(encoding="utf-8")
match = re.search(r"const DATA = (.+?);\n    const (?:samples|summary) = ", old, re.DOTALL)
if not match:
    raise ValueError("Could not locate the embedded experiment data")
payload = match.group(1)
data = json.loads(payload)
assert data["samples"] and data["summary"]["selected_model"]
updated = template_path.read_text(encoding="utf-8").replace("__VQA_DATA__", payload)
html_path.write_text(updated, encoding="utf-8")
print(f"Updated demo.html; retained {len(data['samples'])} actual test examples")
