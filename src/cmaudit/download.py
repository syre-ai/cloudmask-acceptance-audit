"""Fetch CloudSEN12+ evaluation samples by HTTP range request (no full-archive download).

Each sample is stored as ``data/samples/<level>/<tortilla:id>.npz`` with:
- ``bands``: uint16 (4, 512, 512) = B02, B03, B04, B8A (blue, green, red, NIR narrow)
- ``label``: uint8 (512, 512) expert label (0 clear, 1 thick, 2 thin, 3 shadow)
and ``data/samples/extra/<tortilla:id>.npz`` with the operational baseline masks, ESA
WorldCover and elevation, each (512, 512).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio as rio
from tqdm import tqdm

# 1-based band indexes in the 13-band CloudSEN12 GeoTIFF (B01..B12 incl. B8A and B10).
BAND_INDEX = {"B02": 2, "B03": 3, "B04": 4, "B8A": 9}
BANDS = list(BAND_INDEX)

EXTRA_LAYERS = [
    "elevation", "lc10", "cloudmask_qa60", "cloudmask_sen2cor", "cloudmask_s2cloudless",
    "cloudmask_cloudscore_cs_v1", "cloudmask_cloudscore_cs_cdf_v1", "cloudmask_unetmobv2_v1",
    "cloudmask_unetmobv2_v2", "cloudmask_sensei_v2",
]


def _read_level_sample(ds, row: int) -> dict:
    s = ds.read(row)
    with rio.open(s.read(0)) as src:
        bands = src.read([BAND_INDEX[b] for b in BANDS])
    with rio.open(s.read(1)) as src:
        label = src.read(1)
    return {"bands": bands.astype(np.uint16), "label": label.astype(np.uint8)}


def _read_extra_sample(ds, row: int) -> dict:
    s = ds.read(row)
    names = list(s["tortilla:id"])
    out = {}
    for name in EXTRA_LAYERS:
        with rio.open(s.read(names.index(name))) as src:
            out[name] = src.read(1)
    return out


def fetch(mode: str, ids: list[str], out_dir: Path, workers: int = 8) -> list[str]:
    """Download the given sample ids for one TACO mode; returns ids that failed."""
    import tacoreader

    from .meta import TACO_IDS

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = tacoreader.load(TACO_IDS[mode])
    pos = {tid: i for i, tid in enumerate(ds["tortilla:id"].values)}
    todo = [t for t in ids if not (out_dir / f"{t}.npz").exists()]
    reader = _read_extra_sample if mode == "extra" else _read_level_sample

    def job(tid: str) -> str | None:
        try:
            arrays = reader(ds, pos[tid])
            tmp = out_dir / f"{tid}.tmp.npz"
            np.savez_compressed(tmp, **arrays)
            tmp.rename(out_dir / f"{tid}.npz")
            return None
        except Exception as exc:  # noqa: BLE001 - network hiccups: report, caller retries
            print(f"[{mode}] {tid}: {exc}")
            return tid

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(job, t) for t in todo]
        for f in tqdm(as_completed(futs), total=len(futs), desc=mode):
            if (r := f.result()) is not None:
                failed.append(r)
    return failed
