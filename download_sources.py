"""Download official VQA v2 annotations and OpenAI CLIP ViT-B/32.

Only public upstream files are downloaded. COCO images are fetched separately
after the deterministic manifest has been built.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
VENDOR = ROOT / "vendor"
VQA_ROOT = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa"
VQA_NAMES = (
    "v2_Annotations_Train_mscoco.zip",
    "v2_Annotations_Val_mscoco.zip",
    "v2_Questions_Train_mscoco.zip",
    "v2_Questions_Val_mscoco.zip",
)
CLIP_URL = ("https://openaipublic.azureedge.net/clip/models/"
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
            "ViT-B-32.pt")
CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
CLIP_SOURCE_URL = "https://github.com/openai/CLIP/archive/refs/heads/main.zip"


def download(url: str, target: Path) -> None:
    if target.is_file():
        print("Present:", target.name, flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print("Downloading:", url, flush=True)
    urllib.request.urlretrieve(url, partial)
    partial.replace(target)
    print("Saved:", target.name, flush=True)


def main() -> None:
    for name in VQA_NAMES:
        download(f"{VQA_ROOT}/{name}", RAW / name)
    weight = RAW / "ViT-B-32.pt"
    download(CLIP_URL, weight)
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    if digest != CLIP_SHA256:
        raise ValueError("CLIP weight SHA256 mismatch")
    source = RAW / "CLIP-main.zip"
    download(CLIP_SOURCE_URL, source)
    if not (VENDOR / "CLIP-main" / "clip" / "clip.py").is_file():
        VENDOR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(VENDOR)
    print("Official sources ready")


if __name__ == "__main__":
    main()
