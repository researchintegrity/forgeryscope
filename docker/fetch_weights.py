"""
Download every model the detector needs, so the image runs without network
access (ELIES starts tool containers with ``--network none``).

- Forgeryscope checkpoints: downloaded by the model zoo, which checks their
  SHA256.
- LightGlue weights for SIFT and ALIKED features: downloaded by LightGlue into
  the torch hub cache; their SHA256 is checked here.
- ALIKED weights: copied into the torch hub cache from a pinned ALIKED commit
  by the Dockerfile; this script only checks they are there.

Run by the Dockerfile with model_zoo.py copied next to it.
"""
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_zoo  # noqa: E402

LIGHTGLUE_WEIGHTS = {
    "sift_lightglue_v0-1_arxiv.pth": "5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0",
    "aliked_lightglue_v0-1_arxiv.pth": "d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503",
}
ALIKED_WEIGHTS = "aliked-n16.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for name in model_zoo.MODEL_SPECS:
        print(f"{name}: {model_zoo.get_model_path(name)}", flush=True)

    checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    if not (checkpoints / ALIKED_WEIGHTS).is_file():
        sys.exit(f"{ALIKED_WEIGHTS} is missing from {checkpoints}")

    from lightglue import ALIKED, LightGlue

    LightGlue(features="sift")
    LightGlue(features="aliked")
    ALIKED(model_name="aliked-n16")

    for filename, expected in LIGHTGLUE_WEIGHTS.items():
        actual = sha256(checkpoints / filename)
        if actual != expected:
            sys.exit(f"SHA256 mismatch for {filename}: expected {expected}, got {actual}")
        print(f"{filename}: {checkpoints / filename}", flush=True)


if __name__ == "__main__":
    main()
