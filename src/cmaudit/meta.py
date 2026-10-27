"""CloudSEN12+ metadata: load the TACO index and attach a footprint to each sample."""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon

_HF = "https://huggingface.co/datasets/tacofoundation/cloudsen12/resolve/main"
# Shard URLs per mode (from the dataset's file listing). Passing them directly skips
# tacoreader's registry lookup, one fewer request against Hugging Face rate limits.
SHARDS = {
    "l1c": [f"{_HF}/cloudsen12-l1c.{i:04d}.part.taco" for i in range(5)],
    "l2a": [f"{_HF}/cloudsen12-l2a.{i:04d}.part.taco" for i in range(6)],
    "extra": [f"{_HF}/cloudsen12-extra.{i:04d}.part.taco" for i in range(3)],
}

# Columns we keep from the TACO index; internal:subfile holds each sample's byte range.
KEEP = [
    "internal:subfile", "tortilla:id", "tortilla:data_split", "stac:crs", "stac:geotransform", "stac:time_start",
    "rai:ele", "rai:admin0", "roi_id", "equi_zone", "label_type", "s2_id", "real_proj_shape",
    "s2_mean_solar_zenith_angle", "thick_percentage", "thin_percentage",
    "cloud_shadow_percentage", "clear_percentage",
]


def load_index(mode: str) -> pd.DataFrame:
    """Fetch the remote TACO index for one mode (a few MB of metadata, no imagery)."""
    import tacoreader

    ds = tacoreader.load(SHARDS[mode])
    return pd.DataFrame({c: ds[c].values for c in ds.columns if c in KEEP})


def footprint(geotransform, crs: str, size_px: int) -> Polygon:
    """Footprint in WGS84 lon/lat of the real (unpadded) patch.

    CloudSEN12+ pads 509 -> 512 and 2000 -> 2048; the geotransform origin is the top-left
    of the real data, so the real extent is size_px * pixel size from the origin.
    """
    x0, dx, _, y0, _, dy = geotransform
    xs = [x0, x0 + dx * size_px, x0 + dx * size_px, x0]
    ys = [y0, y0, y0 + dy * size_px, y0 + dy * size_px]
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xs, ys)
    return Polygon(zip(lon, lat))


def with_footprints(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geoms = [
        footprint(np.asarray(gt, dtype=float), crs, int(n))
        for gt, crs, n in zip(df["stac:geotransform"], df["stac:crs"], df["real_proj_shape"])
    ]
    return gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:4326")


def acquisition_key(s2_id: str) -> str:
    """Datatake + tile part of a Sentinel-2 product id: identifies the same acquisition
    regardless of processing level (L1C vs L2A) and baseline number.

    S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T143214 -> S2A_20190212T142031_R010_T19FDF
    """
    p = s2_id.split("_")
    return f"{p[0]}_{p[2]}_{p[4]}_{p[5]}"


def cache_index(mode: str, out_dir: Path) -> Path:
    out = out_dir / f"index_{mode}.parquet"
    if not out.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(6):  # the index read goes through rate-limited /resolve/
            try:
                df = load_index(mode)
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(120)
        df["stac:geotransform"] = df["stac:geotransform"].map(lambda g: list(map(float, g)))
        df.to_parquet(out)
    return out
