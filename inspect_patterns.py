"""Read-only exploration of candidate question families in the official train set."""

import re
from collections import Counter

from prepare_data import joined_rows


PATTERNS = {
    "color": lambda q: q.startswith("what color") or q.startswith("what is the color"),
    "count": lambda q: q.startswith("how many"),
    "animal": lambda q: bool(re.search(r"^(what animal|what (kind|type) of animal|what is (this|the|that) animal|what (kind|type) of (dog|bird))\b", q)),
    "object": lambda q: bool(re.search(r"^(what (is|are)|which)\b", q)),
    "yesno": lambda q: bool(re.search(r"^(is|are|does|do|can|has|have|was|were|did)\b", q)),
}

rows = joined_rows("train")
for name, matches in PATTERNS.items():
    group = [r for r in rows if matches(r["question"].lower())]
    print(name, len(group), "top answers", Counter(r["answer"] for r in group).most_common(20))
    for row in group[:5]:
        print("  ", row["question"], "->", row["answer"])
