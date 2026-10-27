"""A3: where does the mask fail? Real subgroups of the clean test set (no synthetic data).

1. Pixel level, by ESA WorldCover land cover under the pixel (bright surfaces such as
   snow/ice and bare ground are the classic false-cloud cases).
2. Image level, by scene regime: true cloud cover band, continent, sun elevation, season.

Every row has an ROI-cluster bootstrap 95% CI. Writes results/subgroups_landcover.csv and
results/subgroups_scene.csv.

    uv run python scripts/subgroups.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cmaudit.baselines import BASELINES
from cmaudit.evaluate import ocm_pred, stack_cms
from cmaudit.metrics import confusion, summarize
from cmaudit.samples import load, manifest, valid_mask

WORLDCOVER = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
              50: "Built-up", 60: "Bare / sparse", 70: "Snow and ice", 80: "Water",
              90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen"}
METHODS = ["OmniCloudMask v4", "Sen2Cor SCL (L2A)", "Cloud Score+ cs"]
STATS = ["leak_rate", "discard_rate", "miss_rate", "boa"]


def landcover_cms(man: pd.DataFrame) -> pd.DataFrame:
    path = Path("data/results/landcover_cms.parquet")
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for tid, roi in tqdm(list(zip(man["tortilla:id"], man["roi_id"])), desc="lc",
                         mininterval=10):
        lv = load("l1c", tid)
        ex = load("extra", tid)
        valid = valid_mask(lv["bands"])
        lc = ex["lc10"]
        preds = {"OmniCloudMask v4": ocm_pred("l1c", tid)}
        for name in METHODS[1:]:
            layer, decode, _ = BASELINES[name]
            preds[name] = decode(ex[layer])
        for code in np.unique(lc[valid]):
            if int(code) not in WORLDCOVER:
                continue
            m = valid & (lc == code)
            for name, p in preds.items():
                rows.append({"tortilla:id": tid, "roi_id": roi, "method": name,
                             "landcover": WORLDCOVER[int(code)],
                             "cm": confusion(lv["label"], p, m).ravel()})
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    return df


def scene_groups(man: pd.DataFrame) -> pd.DataFrame:
    g = man[["tortilla:id", "roi_id", "equi_zone", "s2_mean_solar_zenith_angle",
             "stac:time_start"]].copy()
    per = pd.read_parquet("data/results/per_image_test.parquet")
    lab_cover = per[(per["method"] == "OmniCloudMask v4") & (per["level"] == "l1c")]
    g = g.merge(lab_cover[["tortilla:id", "true_cover"]], on="tortilla:id")
    g["Cloud regime"] = pd.cut(g["true_cover"], [-0.01, 0.01, 0.30, 0.70, 0.99, 1.01],
                               labels=["clear (<1%)", "light (1-30%)", "mixed (30-70%)",
                                       "heavy (70-99%)", "overcast (>99%)"]).astype(str)
    zones = {"AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America",
             "OC": "Oceania", "SA": "South America", "AN": "Antarctica"}
    g["Continent"] = g["equi_zone"].map(zones).fillna(g["equi_zone"])
    elev = 90 - g["s2_mean_solar_zenith_angle"]
    g["Sun elevation"] = pd.cut(elev, [0, 25, 45, 90],
                                labels=["low (<25°)", "mid (25-45°)", "high (>45°)"]).astype(str)
    # Meteorological season, flipped for the southern hemisphere (UTM south = EPSG:327xx).
    month = pd.to_datetime(g["stac:time_start"], unit="s").dt.month
    south = man["stac:crs"].str.startswith("EPSG:327").to_numpy()
    month = np.where(south, (month + 5) % 12 + 1, month)
    names = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
             6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    g["Season"] = pd.Series(month).map(names).to_numpy()
    return g


def main() -> None:
    man = manifest("test")
    Path("results").mkdir(exist_ok=True)

    lc = landcover_cms(man)
    out = []
    for name in METHODS:
        d = lc[lc["method"] == name].reset_index(drop=True)
        t = summarize(stack_cms(d), d["landcover"], d["roi_id"].to_numpy(), STATS, n_boot=500)
        t["pixels_M"] = [stack_cms(d[d["landcover"] == grp]).sum() / 1e6 for grp in t["group"]]
        t.insert(0, "method", name)
        out.append(t)
    lc_tab = pd.concat(out)
    lc_tab.to_csv("results/subgroups_landcover.csv", index=False, float_format="%.4f")

    g = scene_groups(man)
    per = pd.read_parquet("data/results/per_image_test.parquet")
    rows = []
    for name in METHODS:
        level = "l1c" if name == "OmniCloudMask v4" else "product"
        d = per[(per["method"] == name) & (per["level"] == level)]
        d = d.merge(g.drop(columns=["roi_id"]), on="tortilla:id").reset_index(drop=True)
        for col in ["Cloud regime", "Continent", "Sun elevation", "Season"]:
            t = summarize(stack_cms(d), d[col], d["roi_id"].to_numpy(), STATS, n_boot=500)
            t.insert(0, "dimension", col)
            t.insert(0, "method", name)
            rows.append(t)
    sc = pd.concat(rows)
    sc.to_csv("results/subgroups_scene.csv", index=False, float_format="%.4f")

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        f = lambda x: f"{x:.3f}"
        cols = ["method", "group", "images", "rois", "pixels_M", "leak_rate", "leak_rate_lo",
                "leak_rate_hi", "discard_rate", "discard_rate_lo", "discard_rate_hi"]
        print(lc_tab[cols].to_string(index=False, float_format=f))
        cols = ["method", "dimension", "group", "images", "rois", "leak_rate", "leak_rate_lo",
                "leak_rate_hi", "discard_rate", "discard_rate_lo", "discard_rate_hi"]
        print(sc[cols].to_string(index=False, float_format=f))


if __name__ == "__main__":
    main()
