"""Record exactly which data and weights the results were computed from.

    uv run python scripts/provenance.py
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import requests

DATASETS = ["tacofoundation/cloudsen12", "NickWright/OmniCloudMask-Combined-Training-Dataset"]
MODELS = ["NickWright/OmniCloudMask"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = {"packages": {p: version(p) for p in ["omnicloudmask", "torch", "tacoreader",
                                                    "segmentation-models-pytorch", "timm"]}}
    for kind, repos in (("datasets", DATASETS), ("models", MODELS)):
        for repo in repos:
            r = requests.get(f"https://huggingface.co/api/{kind}/{repo}", timeout=60).json()
            out[repo] = {"revision": r.get("sha"), "last_modified": r.get("lastModified")}
    out["ocm_weights_sha256"] = {p.name: sha256(p) for p in sorted(Path("models/ocm").glob("*.safetensors"))}
    out["ocm_model_version"] = 4.0
    out["cloudsen12_dataset_version"] = "1.1.2 (per dataset card)"
    Path("results/provenance.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
