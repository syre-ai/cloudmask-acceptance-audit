"""Run OmniCloudMask v4 (fp32) on every evaluation sample at L1C and L2A and cache
quantized softmax probabilities to data/preds/ocm/<level>/<id>.npz. Idempotent.

    uv run python scripts/run_ocm.py
"""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cmaudit.infer import load_models, ocm_input, predict_probs, quantize
from cmaudit.samples import load, manifest

OUT = Path("data/preds/ocm")
CHUNK = 32


def main() -> None:
    ids = manifest(tiers=None)["tortilla:id"].tolist()
    models = load_models(dtype=torch.float32)
    for level in ("l1c", "l2a"):
        out = OUT / level
        out.mkdir(parents=True, exist_ok=True)
        todo = [t for t in ids if not (out / f"{t}.npz").exists()]
        for i in tqdm(range(0, len(todo), CHUNK), desc=f"ocm {level}"):
            chunk = todo[i : i + CHUNK]
            imgs = [ocm_input(load(level, t)["bands"]) for t in chunk]
            probs = predict_probs(models, imgs, dtype=torch.float32)
            for t, p in zip(chunk, probs):
                np.savez_compressed(
                    out / f"{t}.npz", probs=quantize(p), pred=p.argmax(0).astype(np.uint8)
                )


if __name__ == "__main__":
    main()
