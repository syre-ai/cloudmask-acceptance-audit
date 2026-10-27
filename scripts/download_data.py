"""Download everything the audit needs: CloudSEN12+ indexes, the evaluation samples
(test + validation, high-quality labels, 509 px) at L1C and L2A plus baseline masks, and
the OmniCloudMask v4 weights. Idempotent; re-run to resume.

    uv run python scripts/download_data.py
"""

from pathlib import Path

import pandas as pd

from cmaudit.download import fetch
from cmaudit.meta import cache_index, with_footprints
from cmaudit.overlap import eval_mask, ocm_v4_training_mask, screen

DATA = Path("data")


def main() -> None:
    for mode in ("l1c", "l2a"):
        cache_index(mode, DATA / "index")
    idx = pd.read_parquet(DATA / "index" / "index_l1c.parquet")
    g = with_footprints(idx)
    train = g[ocm_v4_training_mask(g)]

    rows = []
    for split in ("test", "validation"):
        ev = g[eval_mask(g, split)]
        tiers = screen(ev, train).set_index("tortilla:id")
        part = ev.drop(columns="geometry").set_index("tortilla:id").join(tiers).reset_index()
        rows.append(part)
    manifest = pd.concat(rows, ignore_index=True)
    manifest.to_parquet(DATA / "manifest.parquet")
    print(manifest.groupby(["tortilla:data_split", "tier"]).size())

    ids = manifest["tortilla:id"].tolist()
    for mode in ("l1c", "l2a", "extra"):
        for attempt in range(3):
            failed = fetch(mode, ids, DATA / "samples" / mode)
            if not failed:
                break
            print(f"{mode}: {len(failed)} failed, retry {attempt + 1}")
        else:
            raise SystemExit(f"{mode}: {len(failed)} samples still failing")

    from omnicloudmask.download_models import get_models

    get_models(model_dir=Path("models/ocm"), source="hugging_face", model_version=4.0)
    print("done")


if __name__ == "__main__":
    main()
