import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from cmaudit.meta import acquisition_key, footprint
from cmaudit.overlap import ocm_v4_training_mask, screen

ACQ_A = "S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T143214"
ACQ_A_L2A = "S2A_MSIL2A_20190212T142031_N0211_R010_T19FDF_20190212T170000"
ACQ_B = "S2B_MSIL1C_20200301T142031_N0209_R010_T19FDF_20200301T150000"


def gdf(rows):
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=[r.pop("g") for r in rows], crs=4326)


def test_acquisition_key_ignores_processing_level_and_baseline():
    assert acquisition_key(ACQ_A) == acquisition_key(ACQ_A_L2A)
    assert acquisition_key(ACQ_A) != acquisition_key(ACQ_B)


def test_screen_assigns_tiers():
    evals = gdf([
        {"tortilla:id": "e_same", "s2_id": ACQ_A, "g": box(0, 0, 1, 1)},
        {"tortilla:id": "e_loc", "s2_id": ACQ_A, "g": box(10, 0, 11, 1)},
        {"tortilla:id": "e_clean", "s2_id": ACQ_A, "g": box(20, 0, 21, 1)},
    ])
    train = gdf([
        {"tortilla:id": "t1", "s2_id": ACQ_A_L2A, "g": box(0.5, 0, 1.5, 1)},  # same acq, 50%
        {"tortilla:id": "t2", "s2_id": ACQ_B, "g": box(10.9, 0, 12, 1)},  # other date, 10%
        {"tortilla:id": "t3", "s2_id": ACQ_B, "g": box(21, 0, 22, 1)},  # touches edge only
    ])
    out = screen(evals, train).set_index("tortilla:id")
    assert out.loc["e_same", "tier"] == "same_acquisition"
    assert abs(out.loc["e_same", "max_overlap_frac"] - 0.5) < 1e-9
    assert out.loc["e_loc", "tier"] == "same_location"
    assert out.loc["e_clean", "tier"] == "clean"  # zero-area touch is not overlap
    # threshold drops the 10% overlap
    out2 = screen(evals, train, min_overlap_frac=0.2).set_index("tortilla:id")
    assert out2.loc["e_loc", "tier"] == "clean"


def test_training_mask_matches_ocm_card():
    df = pd.DataFrame({
        "label_type": ["high", "high", "high", "scribble", "high", "nolabel"],
        "real_proj_shape": [509, 509, 509, 509, 2000, 509],
        "tortilla:data_split": ["train", "test", "validation", "test", "test", "train"],
    })
    assert ocm_v4_training_mask(df).tolist() == [True, False, False, True, True, False]


def test_footprint_utm_square():
    # 509 px at 10 m in UTM 19S near the CloudSEN12 sample ROI_00001
    p = footprint([440030.0, 10.0, 0.0, 4573910.0, 0.0, -10.0], "EPSG:32719", 509)
    lon0, lat0, lon1, lat1 = p.bounds
    assert -70.0 < lon0 < lon1 < -69.6 and -49.1 < lat0 < lat1 < -48.9
    assert 0.04 < lat1 - lat0 < 0.05  # 5.09 km ~ 0.0458 deg latitude
