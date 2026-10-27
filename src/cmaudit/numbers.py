"""Every number quoted in the README, site write-up and PDF report, read from results/.

Documents never contain typed-in results: scripts/build_docs.py formats them from here, so a
re-run of the pipeline updates every document consistently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

R = Path("results")


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def _r(x: float, d: int) -> float:
    return round(100 * x, d) + 0.0  # + 0.0 turns -0.0 into 0.0


def pp(x: float, d: int = 1) -> str:
    return f"{_r(x, d):+.{d}f} pp"


def ci(lo: float, hi: float, d: int = 1) -> str:
    return f"[{_r(lo, d):.{d}f}, {_r(hi, d):.{d}f}]"


def load() -> dict:
    N: dict = {}
    ov = pd.read_csv(R / "overall_test.csv").set_index("group")
    o = ov.loc["OmniCloudMask v4 | l1c"]
    N["images"], N["sites"] = int(o["images"]), int(o["rois"])
    N["boa"] = f"{o['boa']:.3f}"
    N["_leak"], N["_discard"] = float(o["leak_rate"]), float(o["discard_rate"])
    N["boa_ci"] = f"[{o['boa_lo']:.3f}, {o['boa_hi']:.3f}]"
    for s in ("leak_rate", "discard_rate"):
        N[s] = pct(o[s])
        N[f"{s}_ci"] = ci(o[f"{s}_lo"], o[f"{s}_hi"])
    for s in ("leak_thick", "leak_thin", "leak_shadow"):
        N[s] = pct(o[s])
    for key, short in (("Sen2Cor SCL (L2A) | product", "scl"), ("QA60 (L1C bitmask) | product", "qa60"),
                       ("s2cloudless (upstream defaults) | product", "s2c"),
                       ("Cloud Score+ cs | product", "csp"), ("SEnSeI v2 | product", "sensei"),
                       ("UnetMobV2 v2 | product", "unet2")):
        r = ov.loc[key]
        N[f"{short}_leak"], N[f"{short}_discard"] = pct(r["leak_rate"]), pct(r["discard_rate"])
        N[f"{short}_boa"] = f"{r['boa']:.3f}"

    pdiff = pd.read_csv(R / "paired_vs_ocm_test.csv").set_index("method")
    for key, short in (("SEnSeI v2 | product", "sensei"), ("OmniCloudMask v4 | l2a", "l2a"),
                       ("Sen2Cor SCL (L2A) | product", "scl"), ("UnetMobV2 v2 | product", "unet2")):
        r = pdiff.loc[key]
        for s in ("leak_rate", "discard_rate", "boa"):
            # stored as OCM minus method; report method minus OCM
            N[f"{short}_d_{s}"] = pp(-r[f"d_{s}"])
            N[f"{short}_d_{s}_ci"] = ci(-r[f"d_{s}_hi"], -r[f"d_{s}_lo"])

    sp = pd.read_csv(R / "scene_policy_test.csv")
    sp10 = sp[sp["threshold"].round(2) == 0.10].set_index("key")
    N["scl_gate_false_accept"] = pct(sp10.loc["Sen2Cor SCL (L2A) | product", "false_accept_rate"], 0)
    N["scl_cover_bias"] = f"{100 * sp10.loc['Sen2Cor SCL (L2A) | product', 'cover_bias']:+.0f} pp"
    N["ocm_gate_false_accept"] = pct(sp10.loc["OmniCloudMask v4 | l1c", "false_accept_rate"])

    op = pd.read_csv(R / "operating_points_test.csv")
    def opv(method, rule):
        return op[(op["method"] == method) & op["rule"].str.startswith(rule)].iloc[0]
    r = opv("OmniCloudMask v4 | l1c", "leak <= 2%")
    N["ocm_2pct_threshold"] = f"{r['threshold_native']:.2f}"
    N["ocm_2pct_leak"], N["ocm_2pct_leak_ci"] = pct(r["test_leak"]), ci(r["test_leak_lo"], r["test_leak_hi"])
    N["ocm_2pct_discard"] = pct(r["test_discard"])
    N["ocm_2pct_discard_ci"] = ci(r["test_discard_lo"], r["test_discard_hi"])
    N["csp_2pct_discard"] = pct(opv("Cloud Score+ cs | product", "leak <= 2%")["test_discard"], 0)
    N["s2c_2pct_achievable"] = bool(opv("s2cloudless | product", "leak <= 5%")["achievable"])
    cv = pd.read_csv(R / "curves_test.csv")
    N["s2c_leak_floor"] = pct(cv[cv["method"] == "s2cloudless | product"]["leak_rate"].min())

    lc = pd.read_csv(R / "subgroups_landcover.csv")
    lc = lc[lc["method"] == "OmniCloudMask v4"].set_index("group")
    conc = pd.read_csv(R / "subgroups_landcover_concentration.csv").set_index("landcover")
    for name, short in (("Water", "water"), ("Snow and ice", "snow"), ("Grassland", "grass"),
                        ("Tree cover", "tree")):
        N[f"{short}_leak"] = pct(lc.loc[name, "leak_rate"])
        N[f"{short}_leak_ci"] = ci(lc.loc[name, "leak_rate_lo"], lc.loc[name, "leak_rate_hi"])
        N[f"{short}_sites"] = int(lc.loc[name, "rois"])
        N[f"{short}_top2"] = pct(conc.loc[name, "top2_site_share_of_leaked_px"], 0)
        N[f"{short}_leak_thick"] = pct(conc.loc[name, "leak_thick"])
        N[f"{short}_leak_thin"] = pct(conc.loc[name, "leak_thin"])

    sc = pd.read_csv(R / "subgroups_scene.csv")
    sc = sc[(sc["method"] == "OmniCloudMask v4") & (sc["dimension"] == "Cloud regime")].set_index("group")
    N["light_miss"] = pct(sc.loc["light (1-30%)", "miss_rate"])
    N["light_miss_ci"] = ci(sc.loc["light (1-30%)", "miss_rate_lo"], sc.loc["light (1-30%)", "miss_rate_hi"])
    N["heavy_discard"] = pct(sc.loc["heavy (70-99%)", "discard_rate"])
    N["heavy_discard_ci"] = ci(sc.loc["heavy (70-99%)", "discard_rate_lo"],
                               sc.loc["heavy (70-99%)", "discard_rate_hi"])
    N["overcast_miss"] = pct(sc.loc["overcast (>99%)", "miss_rate"])

    sens = pd.read_csv(R / "sensitivity_kappaset.csv")
    N["kappa_images"] = int(sens.iloc[1]["images"])
    N["kappa_leak"] = pct(sens.iloc[1]["leak_rate"])

    scr = json.loads((R / "ocm_training_screen.json").read_text())
    N["train_rows"] = f"{scr['training_rows']:,}"
    N["train_cs12_rows"] = f"{scr['cloudsen12_rows']:,}"
    tags = scr["cloudsen12_split_tags"]
    N["train_cs12_heldout_rows"] = f"{tags.get('test', 0) + tags.get('validation', 0):,}"
    N["train_id_hits"] = len(scr["cloudsen12_ids_matching_eval_images"])
    N["train_acq_hits"] = len(scr["non_cloudsen12_same_tile_and_date_as_eval"])
    N["train_near_images"] = len(scr["eval_images_within_2.6km_of_ocm_custom_scene_centre"])

    if (R / "mitigation_buffer.csv").exists():
        b = pd.read_csv(R / "mitigation_buffer.csv").set_index("k_px")
        kmax = b.index.max()
        N["buffer_kmax"] = int(kmax)
        N["buffer_kmax_leak"], N["buffer_kmax_discard"] = (pct(b.loc[kmax, "test_leak_rate"]),
                                                           pct(b.loc[kmax, "test_discard_rate"]))
        g = pd.read_csv(R / "mitigation_scene_gate.csv").set_index("gate_cover_from")
        for name, short in (("OmniCloudMask", "ocm"), ("Sen2Cor SCL", "scl")):
            N[f"gate_{short}_kept"] = int(g.loc[name, "scenes_kept"])
            N[f"gate_{short}_over"] = pct(g.loc[name, "kept_truly_over_threshold"])
            N[f"gate_{short}_rejected_ok"] = pct(g.loc[name, "truly_ok_scenes_rejected"])
            N[f"gate_{short}_leak"] = pct(g.loc[name, "pixel_leak_in_kept_after_ocm_mask"])
            N[f"gate_{short}_leak_ci"] = ci(g.loc[name, "leak_lo"], g.loc[name, "leak_hi"])
    if (R / "gain_offset_check.json").exists():
        N["gain_agreement"] = pct(json.loads((R / "gain_offset_check.json").read_text())
                                  ["pixel_agreement_with_clean"], 2)
    if (R / "perturbations_test.csv").exists():
        t = pd.read_csv(R / "perturbations_test.csv")
        N["perturb"] = t

        def d(name: str, sev: float, stat: str) -> float:
            row = t[(t["perturbation"] == name) & (t["severity"].round(3) == round(sev, 3))]
            return float(row.iloc[0][f"d_{stat}"])

        for key, name, sev in (("fine30", "Coarse data on a 10 m grid", 3),
                               ("fine20", "Coarse data on a 10 m grid", 2),
                               ("native30", "Coarser resolution", 3),
                               ("native20", "Coarser resolution", 2),
                               ("clip03", "8-bit lossless (clip level)", 0.3),
                               ("clip06", "8-bit lossless (clip level)", 0.6),
                               ("clip10", "8-bit lossless (clip level)", 1.0),
                               ("jpeg95", "8-bit JPEG delivery", 95),
                               ("jpeg50", "8-bit JPEG delivery", 50),
                               ("noise005", "Sensor noise", 0.005),
                               ("noise01", "Sensor noise", 0.01),
                               ("blur1", "Optical blur", 1.0),
                               ("mis4", "NIR misregistration", 4)):
            dl, dd = d(name, sev, "leak_rate"), d(name, sev, "discard_rate")
            N[f"{key}_dleak"], N[f"{key}_ddiscard"] = pp(dl), pp(dd)
            N[f"{key}_leak"], N[f"{key}_discard"] = pct(N["_leak"] + dl), pct(N["_discard"] + dd)
        N["jpeg95_extra"] = pp(d("8-bit JPEG delivery", 95, "leak_rate")
                               - d("8-bit lossless (clip level)", 0.6, "leak_rate"))
        N["jpeg50_extra"] = pp(d("8-bit JPEG delivery", 50, "leak_rate")
                               - d("8-bit lossless (clip level)", 0.6, "leak_rate"))
        mis = t[t["perturbation"] == "NIR misregistration"]
        N["mis_max_abs"] = pp(float(mis[["d_leak_rate", "d_discard_rate"]].abs().max().max()))
    return N
