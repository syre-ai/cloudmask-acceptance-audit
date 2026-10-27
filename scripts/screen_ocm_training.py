"""Check OCM v4's published training set against our evaluation images directly, rather than
trusting the dataset card. Reads only the small text columns of the training parquet shards
(subset, processing level, filenames), never the image bytes.

Checks:
1. CloudSEN12 lineage: no CloudSEN12 sample id used in training is one of our evaluation
   images (any variant: high, scribble, 2k, Planetary Computer re-download, super-resolution).
2. Same acquisition: no non-CloudSEN12 training scene (Kappaset, OCM hard negative, OCM
   scribble) shares a Sentinel-2 tile and sensing date with an evaluation image.

3. Location: OCM custom scenes (hard negative, scribble) name their centre lat/lon; any
   evaluation image whose footprint touches a conservative disc around such a scene (radius:
   half-diagonal plus coordinate rounding, true ground distance) is excluded (tier
   "same_location"). Kappaset names only its
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
# Pinned revision (the v4 training set as published 2026-03-05), so a rerun reads the same rows.
REVISION = "cf7f958be9954301848d6d08fdb420292134b0a9"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/"
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

    # Location (any date). OCM custom scenes name their centre lat/lon (rounded to 0.01 deg)
    # and span 509 px x 10 m = 5.09 km. A disc of radius half-diagonal (3.6 km) plus the
    # rounding error (~0.8 km), built in a local azimuthal-equidistant projection so the radius
    # is true ground distance at any latitude, contains the whole scene; any evaluation footprint
    # it touches may overlap. Conservative: some flagged pairs will not actually overlap.
    from pyproj import CRS, Transformer
    from shapely.geometry import Point
    from shapely.ops import transform as shp_transform

    from cmaudit.meta import with_footprints

    fp = with_footprints(ev)
    ll = other["image_filename"].str.extract(r"_(-?\d+\.\d+)_(-?\d+\.\d+)_image").astype(float)
    radius = 5090 / 2 * 2**0.5 + 800
    discs = []
    for lat, lon in ll.dropna().to_numpy():
        aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
        back = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True).transform
        discs.append(shp_transform(back, Point(0, 0).buffer(radius, 64)))
    near = sorted({fp["tortilla:id"].iloc[i] for d in discs
                   for i in fp.sindex.query(d, predicate="intersects")})
    kappa_tiles = {td[0] for td in other.loc[other["subset"] == "Kappaset", "tile_date"]}
    ev_tiles = ev["s2_id"].map(lambda x: x.split("_")[5][1:])

    # Same acquisition (tile + sensing date) between evaluation images and CloudSEN12 training
    # patches. The footprint screen already shows these patches do not overlap the evaluation
    # patches in space; this counts shared acquisitions so the wording can be exact.
    cs_td = cs["cs12_id"].str.extract(r"__(\d{8})T\d{6}_\d{8}T\d{6}_T(\w{5})")
    cs_pairs = set(zip(cs_td[1], cs_td[0]))
    raw = pd.read_parquet("data/manifest.parquet")  # not manifest(): it merges previous flags
    eval_screened = raw[(raw["tortilla:data_split"] == "test") & (raw["tier"] == "clean")
                        & ~raw["tortilla:id"].isin(near)]
    ev_scr_td = {(r.split("_")[5][1:], r.split("_")[2][:8]) for r in eval_screened["s2_id"]}
    shared = ev_scr_td & cs_pairs
    cs_shared_rows = int(sum((t, d) in shared for d, t in zip(cs_td[0], cs_td[1])))

    report = {
        "training_rows": len(meta),
        "rows_by_subset": meta["subset"].value_counts().to_dict(),
        "cloudsen12_rows": len(cs),
        "cloudsen12_split_tags": split_tags.value_counts(dropna=False).to_dict(),
        "cloudsen12_ids_matching_eval_images": sorted(hits["cs12_id"].unique().tolist()),
        "cloudsen12_matching_rows_by_subset": hits["subset"].value_counts().to_dict(),
        "test_cloudsen12_training_rows_sharing_an_acquisition": cs_shared_rows,
        "test_acquisitions_shared_with_cloudsen12_training_rows": len(shared),
        "non_cloudsen12_rows": len(other),
        "non_cloudsen12_rows_with_tile_and_date": int(parsed.sum()),
        "non_cloudsen12_same_tile_and_date_as_eval": sorted(
            same_acq["image_filename"].unique().tolist()),
        "ocm_custom_scenes_with_latlon": len(discs),
        "eval_images_possibly_overlapping_ocm_custom_scene": near,
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
