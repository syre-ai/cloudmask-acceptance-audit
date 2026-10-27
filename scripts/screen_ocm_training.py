"""Check OCM v4's published training set against our evaluation images directly, rather than
trusting the dataset card. Reads only the small text columns of the training parquet shards
(subset, processing level, filenames), never the image bytes.

Checks:
1. CloudSEN12 lineage: no CloudSEN12 sample id used in training is one of our evaluation
   images (any variant: high, scribble, 2k, Planetary Computer re-download, super-resolution).
2. Same acquisition: no non-CloudSEN12 training scene (Kappaset, OCM hard negative, OCM
   scribble) shares a Sentinel-2 tile and sensing date with an evaluation image.

3. Location: OCM custom scenes (hard negative, scribble) name their centre lat/lon; any
   evaluation image within 2.6 km is excluded (tier "same_location"). Kappaset names only its
   MGRS tile (~110 km), so images on a Kappaset tile are flagged for a sensitivity check.

Writes data/ocm_training_meta.parquet, data/extra_overlap_flags.parquet and
results/ocm_training_screen.json.

    uv run python scripts/screen_ocm_training.py
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq
import requests

from cmaudit.samples import manifest

REPO = "NickWright/OmniCloudMask-Combined-Training-Dataset"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/"
COLS = ["subset", "processing_level", "image_filename", "label_filename"]
OUT = Path("data/ocm_training_meta.parquet")

CS12_ID = re.compile(r"(ROI_\d{5}__\d{8}T\d{6}_\d{8}T\d{6}_T\w{5})")
TILE_DATE = re.compile(r"(?<![A-Z0-9])T?(\d{2}[C-X][A-Z]{2})(?![A-Z0-9]).*?(20\d{6})|"
                       r"(20\d{6})T\d{6}.*?_T(\d{2}[C-X][A-Z]{2})")


def shard_names() -> list[str]:
    r = requests.get(API, timeout=60)
    r.raise_for_status()
    return sorted(x["path"].split("/")[-1] for x in r.json()
                  if x["path"].endswith(".parquet") and "/train-" in x["path"])


def read_meta(name: str) -> pd.DataFrame:
    fs = fsspec.filesystem("https")
    for attempt in range(6):
        try:
            with fs.open(BASE + name, "rb", block_size=2**20) as f:
                t = pq.ParquetFile(f).read(columns=COLS)
            return t.to_pandas().assign(shard=name)
        except Exception:  # noqa: BLE001 - rate limit / transient
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(name)


def tile_date(name: str) -> tuple[str, str] | None:
    m = TILE_DATE.search(name)
    if not m:
        return None
    if m.group(1):
        return m.group(1), m.group(2)
    return m.group(4), m.group(3)


def main() -> None:
    if OUT.exists():
        meta = pd.read_parquet(OUT)
    else:
        names = shard_names()
        with ThreadPoolExecutor(6) as pool:
            meta = pd.concat(pool.map(read_meta, names), ignore_index=True)
        meta.to_parquet(OUT)
    print(meta["subset"].value_counts().to_string())

    ev = manifest(tiers=None)
    ev_ids = set(ev["tortilla:id"])
    meta["cs12_id"] = meta["image_filename"].str.extract(CS12_ID, expand=False)
    cs = meta.dropna(subset=["cs12_id"])
    hits = cs[cs["cs12_id"].isin(ev_ids)]
    split_tags = cs["image_filename"].str.extract(r"_(train|validation|val|test)_", expand=False)

    ev_td = {(r["s2_id"].split("_")[5][1:], r["s2_id"].split("_")[2][:8]) for _, r in ev.iterrows()}
    other = meta[meta["cs12_id"].isna()].copy()
    other["tile_date"] = other["image_filename"].map(tile_date)
    parsed = other["tile_date"].notna()
    same_acq = other[parsed & other["tile_date"].isin(ev_td)]

    # Location (any date). OCM custom scenes name their centre lat/lon; a 509 px patch spans
    # ~5.1 km, so a centre inside an evaluation footprint buffered by 2.6 km can overlap it.
    from shapely.geometry import Point

    from cmaudit.meta import with_footprints

    fp = with_footprints(ev)
    ll = other["image_filename"].str.extract(r"_(-?\d+\.\d+)_(-?\d+\.\d+)_image").astype(float)
    pts = [Point(lon, lat) for lat, lon in ll.dropna().to_numpy()]
    buffered = fp.to_crs(3857).buffer(2600).to_crs(4326)
    near = sorted({fp["tortilla:id"].iloc[i] for p in pts
                   for i in buffered.sindex.query(p, predicate="intersects")})
    kappa_tiles = {td[0] for td in other.loc[other["subset"] == "Kappaset", "tile_date"]}
    ev_tiles = ev["s2_id"].map(lambda x: x.split("_")[5][1:])

    report = {
        "training_rows": len(meta),
        "rows_by_subset": meta["subset"].value_counts().to_dict(),
        "cloudsen12_rows": len(cs),
        "cloudsen12_split_tags": split_tags.value_counts(dropna=False).to_dict(),
        "cloudsen12_ids_matching_eval_images": sorted(hits["cs12_id"].unique().tolist()),
        "cloudsen12_matching_rows_by_subset": hits["subset"].value_counts().to_dict(),
        "non_cloudsen12_rows": len(other),
        "non_cloudsen12_rows_with_tile_and_date": int(parsed.sum()),
        "non_cloudsen12_same_tile_and_date_as_eval": sorted(
            same_acq["image_filename"].unique().tolist()),
        "ocm_custom_scenes_with_latlon": len(pts),
        "eval_images_within_2.6km_of_ocm_custom_scene_centre": near,
        "eval_images_on_a_kappaset_mgrs_tile_any_date": int(ev_tiles.isin(kappa_tiles).sum()),
        "eval_images_total": len(ev),
        "unparsed_examples": other.loc[~parsed, "image_filename"].head(10).tolist(),
    }
    flags = pd.DataFrame({"tortilla:id": ev["tortilla:id"]})
    flags["ocm_custom_near"] = flags["tortilla:id"].isin(near)
    flags["kappaset_tile"] = ev_tiles.isin(kappa_tiles).to_numpy()
    flags.to_parquet("data/extra_overlap_flags.parquet")
    Path("results").mkdir(exist_ok=True)
    Path("results/ocm_training_screen.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "unparsed_examples"}, indent=1,
                     default=str)[:3000])
    print("unparsed examples:", report["unparsed_examples"])


if __name__ == "__main__":
    main()
