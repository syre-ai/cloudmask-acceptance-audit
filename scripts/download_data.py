"""Download everything the audit needs: CloudSEN12+ indexes, the evaluation samples
(test + validation, high-quality labels, 509 px) at L1C and L2A plus baseline masks, and
the OmniCloudMask v4 weights. Idempotent; re-run to resume.

    uv run python scripts/download_data.py
"""

from pathlib import Path

import pandas as pd
from provenance import sha256

from cmaudit.download import fetch
from cmaudit.meta import cache_index, with_footprints
from cmaudit.overlap import eval_mask, ocm_v4_training_mask, screen

DATA = Path("data")
# OCM v4 weights the published results were computed with (see results/provenance.json).
OCM_WEIGHTS_SHA256 = {
    "PM_model_OCM_7.97_R_G_NIR_3_smp_edgenext_small.usi_in1k_PT_state.safetensors":
        "d5fe67ad00f6fdb73eb8382ad925849e97fb82cedb46225344afff1a734a9c1d",
    "PM_model_OCM_7.97_R_G_NIR_3_smp_regnety_004.pycls_in1k_PT_state.safetensors":
        "7f6e4202e17ee73efa4aba7abb5c34f4f90a9f7eb42480820714994dff2db660",
}


def main() -> None:
    for mode in ("l1c", "l2a", "extra"):
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
        index = pd.read_parquet(DATA / "index" / f"index_{mode}.parquet")
        for attempt in range(3):
            failed = fetch(mode, index, ids, DATA / "samples" / mode)
            if not failed:
                break
            print(f"{mode}: {len(failed)} failed, retry {attempt + 1}")
        else:
            raise SystemExit(f"{mode}: {len(failed)} samples still failing")

    from omnicloudmask.download_models import get_models

    Path("models/ocm").mkdir(parents=True, exist_ok=True)
    get_models(model_dir=Path("models/ocm"), source="hugging_face", model_version=4.0)
    for name, expected in OCM_WEIGHTS_SHA256.items():
        got = sha256(Path("models/ocm") / name)
        if got != expected:
            raise SystemExit(f"OCM weights changed upstream: {name} sha256 {got} != {expected}")
    print("done")


if __name__ == "__main__":
    main()
