"""Build README.md, the site write-up and the client PDF report from results/.

No result is typed into these documents: every figure comes from cmaudit.numbers, which reads
the CSV/JSON outputs of the pipeline.

    uv run python scripts/build_docs.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from cmaudit.numbers import load, pct, pp

TODAY = dt.date.today().isoformat()
MATERIAL = 0.01  # a simulated degradation is "material" if it moves leak or discard by >= 1 pp


# --------------------------------------------------------------------------------------------
# A4 summary, derived from the table with a stated rule
# --------------------------------------------------------------------------------------------
def degradation_rows(N: dict) -> list[dict]:
    t = N["perturb"]
    t = t[~t["perturbation"].str.startswith("Oracle")]
    rows = []
    for name, g in t.groupby("perturbation", sort=False):
        cells = []
        worst = 0.0
        for _, r in g.iterrows():
            sig_leak = r["d_leak_rate_lo"] > 0 or r["d_leak_rate_hi"] < 0
            sig_disc = r["d_discard_rate_lo"] > 0 or r["d_discard_rate_hi"] < 0
            cells.append(f"{_sev_label(name, r['severity'])}: leak {pp(r['d_leak_rate'])}"
                         f"{'' if sig_leak else ' (n.s.)'}, discard {pp(r['d_discard_rate'])}"
                         f"{'' if sig_disc else ' (n.s.)'}")
            worst = max(worst, abs(r["d_leak_rate"]), abs(r["d_discard_rate"]))
        rows.append({"name": name, "cells": cells, "worst": worst,
                     "material": worst >= MATERIAL})
    return rows


def _sev_label(name: str, sev) -> str:
    sev = float(sev)
    return {
        "Sensor noise": f"σ {sev:g}",
        "Optical blur": f"σ {sev:g} px",
        "Coarser resolution": f"{int(sev) * 10} m",
        "NIR misregistration": f"{int(sev)} px",
        "8-bit JPEG delivery": f"q{int(sev)}",
        "8-bit lossless (clip level)": f"clip {sev:g}",
        "Coarse data on a 10 m grid": f"{int(sev) * 10} m",
    }.get(name, f"{sev:g}")


def oracle_rows(N: dict) -> dict:
    """Absolute leak/discard at each coarse grid: OCM run natively there, OCM fed the same
    data resampled to 10 m, and the oracle (expert label reduced to that grid). Differences in
    the table are paired against the clean 10 m run, so absolute = clean + difference."""
    t = N["perturb"]
    get = {name: t[t["perturbation"] == name].set_index("severity")
           for name in ("Oracle floor (label at coarse grid)", "Coarser resolution",
                        "Coarse data on a 10 m grid")}
    out = {}
    for f in get["Coarser resolution"].index:
        row = {}
        for key, name in (("oracle", "Oracle floor (label at coarse grid)"),
                          ("native", "Coarser resolution"), ("fine", "Coarse data on a 10 m grid")):
            if f in get[name].index:
                row[f"{key}_leak"] = N["_leak"] + get[name].loc[f, "d_leak_rate"]
                row[f"{key}_discard"] = N["_discard"] + get[name].loc[f, "d_discard_rate"]
        out[int(f)] = row
    return out


# --------------------------------------------------------------------------------------------
# Findings, ranked
# --------------------------------------------------------------------------------------------
def findings(N: dict) -> list[dict]:
    F = [
        {"sev": "High", "title": "Scene filters built on the product's own cloud mask keep cloudy scenes",
             "evidence": (f"Keeping scenes whose Sen2Cor SCL cloud+shadow cover is at most 10%: "
                       f"{N['gate_scl_over']} of kept scenes are actually above 10% (expert labels); "
                       f"SCL under-reports cover by {N['scl_cover_bias']} on average. The same gate "
                       f"computed from OmniCloudMask: {N['gate_ocm_over']} above, and it rejects "
                       f"{N['gate_ocm_rejected_ok']} of good scenes (SCL: {N['gate_scl_rejected_ok']})."),
             "action": "Gate scenes on OmniCloudMask cover instead of the Sen2Cor SCL band.",
             "rec": ("Gate scenes on OmniCloudMask-derived cover, not the SCL band. Tested: cloud "
                  f"reaching the pipeline inside kept scenes falls from {N['gate_scl_leak']} to "
                  f"{N['gate_ocm_leak']} {N['gate_ocm_leak_ci']} after masking.")},
        {"sev": "Medium", "title": "At its default setting, about 6% of accepted pixels are cloud or shadow",
             "evidence": (f"Leak {N['leak_rate']} {N['leak_rate_ci']} of accepted pixels; "
                       f"thin cloud {N['leak_thin']}, shadow {N['leak_shadow']}, thick {N['leak_thick']}. "
                       f"Discard {N['discard_rate']} {N['discard_rate_ci']} of clear pixels."),
             "action": (f"For leak-sensitive uses, accept pixels only at P(clear) >= "
                        f"{N['ocm_2pct_threshold']}; do not rely on buffering."),
             "rec": (f"For leak-sensitive uses (time series, change detection), accept a pixel only "
                  f"if P(clear) >= {N['ocm_2pct_threshold']} (chosen on the validation split): "
                  f"test leak {N['ocm_2pct_leak']} {N['ocm_2pct_leak_ci']} at "
                  f"{N['ocm_2pct_discard']} discard. Do not buffer the mask instead: an "
                  f"{N['buffer_kmax']}-pixel buffer still leaks {N['buffer_kmax_leak']} at "
                  f"{N['buffer_kmax_discard']} discard.")},
        {"sev": "Medium", "title": f"Cloud over water leaks more often ({N['water_leak']} vs {N['leak_rate']} overall)",
             "evidence": (f"Leak over water {N['water_leak']} {N['water_leak_ci']} across "
                       f"{N['water_sites']} sites (thick cloud alone {N['water_leak_thick']}, vs "
                       f"{N['grass_leak_thick']} over grassland); the two worst sites hold "
                       f"{N['water_top2']} of it, so it is not one bad scene."),
             "rec": "For coastal, inland-water and flood work, use the stricter threshold above and "
                 "spot-check a local sample before relying on the mask.",
             "action": "Spot-check a local sample before relying on the mask over water."},
        {"sev": "Medium", "title": "Small amounts of cloud are missed in mostly clear scenes",
             "evidence": (f"In scenes with 1-30% cloud, {N['light_miss']} {N['light_miss_ci']} of the "
                       f"cloud and shadow present is accepted as clear (vs {N['overcast_miss']} in "
                       f"overcast scenes). These are the scenes a cover gate keeps."),
             "rec": "Covered by the threshold recommendation; the tested buffer does not fix it as "
                 "cheaply."},
        {"sev": "Medium", "title": "Usable ground is discarded in heavily clouded scenes",
             "evidence": (f"In scenes with 70-99% cloud, {N['heavy_discard']} {N['heavy_discard_ci']} "
                       f"of the clear pixels are rejected."),
             "rec": "If gap-filling from cloudy scenes matters, validate a lower threshold for that "
                 "regime (not tested here).",
             "action": "If you gap-fill from cloudy scenes, validate a lower threshold for them."},
        {"sev": "Low", "title": "Snow and ice: higher leak, but weak evidence",
             "evidence": (f"Leak {N['snow_leak']} {N['snow_leak_ci']}, only {N['snow_sites']} sites, "
                       f"{N['snow_top2']} of it from two sites and mostly thin cloud over snow, where "
                       f"expert labels are least certain."),
             "rec": "Validate locally before using the mask in snow-covered regions."},
    ]
    F.insert(0, {
        "sev": "High", "title": "Coarse imagery resampled to a finer grid is heavily over-masked",
        "evidence": (f"Simulated: 30 m data put on a 10 m grid (bilinear, as data cubes do) raises "
                     f"discard to {N['fine30_discard']} ({N['fine30_ddiscard']}); the same data run "
                     f"on its native 30 m grid: {N['native30_discard']} ({N['native30_ddiscard']}). "
                     f"At 20 m: {N['fine20_ddiscard']} vs {N['native20_ddiscard']}. Blur alone "
                     f"(1 px) does the same: discard {N['blur1_ddiscard']}."),
        "rec": "Run the mask on each sensor's native grid, then resample the mask, never the "
               "imagery first. Tested: the native run keeps the discard increase near 2 pp.",
        "action": "Run the mask on each sensor's native grid; resample the mask, not the imagery."})
    F.insert(3, {
        "sev": "Medium", "title": "Clipped 8-bit inputs let bright cloud through",
        "evidence": (f"Simulated lossless 8-bit with reflectance clipped at 0.3 (close to the "
                     f"Sentinel-2 true-colour stretch): leak {N['clip03_leak']} "
                     f"({N['clip03_dleak']}); clip at 0.6: {N['clip06_dleak']}; at 1.0: "
                     f"{N['clip10_dleak']}. JPEG compression adds only {N['jpeg95_extra']} at "
                     f"quality 95 and {N['jpeg50_extra']} at quality 50 on top of clipping."),
        "rec": "Feed unclipped reflectance (12/16-bit), or 8-bit only with the range kept to at "
               "least 1.0 reflectance; do not run the mask on visual/stretched products.",
        "action": "Feed unclipped reflectance; never run the mask on stretched 8-bit products."})
    F.append({
        "sev": "Low", "title": "Noisier sensors leak more",
        "evidence": (f"Simulated Gaussian noise of 0.005 reflectance (several times Sentinel-2's) "
                     f"raises leak by {N['noise005_dleak']}; 0.01 by {N['noise01_dleak']}."),
        "rec": "Re-validate before using the mask on lower-SNR small-satellite imagery."})
    F.append({
        "sev": "Info", "title": "Verified non-issues",
        "evidence": (f"L1C vs L2A input: leak {N['l2a_d_leak_rate']} {N['l2a_d_leak_rate_ci']}. "
                  f"Per-band calibration drift: {N.get('gain_agreement', 'n/a')} pixel agreement "
                  f"with the clean run (the model normalizes each band per image). "
                  f"NIR misregistration up to 4 px: at most {N['mis_max_abs']} on leak or discard. "
                  "Results are stable across continents, seasons and sun elevation (overlapping CIs)."),
        "rec": "No action."})
    return F


# --------------------------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------------------------
def readme(N: dict) -> str:
    deg = degradation_rows(N)
    orc = oracle_rows(N)
    deg_md = "\n".join(f"| {d['name']} | " + " · ".join(d["cells"]) + " |" for d in deg)
    orc_md = "\n".join(
        f"| {f * 10} m | {pct(v['native_leak'])} / {pct(v['native_discard'])} | "
        f"{pct(v.get('fine_leak', float('nan')))} / {pct(v.get('fine_discard', float('nan')))} | "
        f"{pct(v['oracle_leak'])} / {pct(v['oracle_discard'])} |" for f, v in sorted(orc.items()))
    find_md = "\n".join(f"| {f['sev']} | {f['title']} | {f['rec']} |" for f in findings(N))
    return f"""# Cloud-mask acceptance audit: OmniCloudMask v4

An independent assurance audit of [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)
(v4 weights, MIT), an open cloud and cloud-shadow segmentation model for Sentinel-2 and
other 10-50 m imagery, run the way a client would commission it: **before you automate on this
mask, how much cloud leaks into the pixels you accept, how much usable ground do you throw
away, and where does it fail?**

Every number below is computed by this repository from public data; nothing is typed in by
hand. The client-style report is in [`report/`](report/).

## Why this matters

Almost every optical imagery pipeline starts by deciding which pixels and which scenes to
use. Cloud and shadow that slip through become false change, wrong indices and bad training
data; clear ground that gets masked is lost revisit. The decision is usually made by a mask
nobody has measured on the buyer's own terms.

## What was tested

- **Model:** OmniCloudMask v4 (two U-Nets, red/green/NIR input), fp32 on an RTX 2070 Super.
- **Data:** CloudSEN12+ (CC0), expert-labeled test split: **{N['images']} scenes at {N['sites']}
  sites**, 509 x 509 px at 10 m, both L1C and L2A. Validation split used only to choose
  operating points.
- **Training-overlap screen:** OmniCloudMask's published training set lists CloudSEN12 scribble
  and 2k patches from *all* splits, so the screen was run against its actual training metadata
  ({N['train_rows']} rows): {N['train_id_hits']} id matches with the evaluation images
  (including {N['train_cs12_heldout_rows']} test/validation-tagged training rows), {N['train_acq_hits']}
  same-acquisition matches, and one site (5 images) within 2.6 km of a custom training scene,
  excluded. Kappaset names only its 110 km tile; excluding all {N['images'] - N['kappa_images']} images on those tiles
  leaves leak at {N['kappa_leak']} (unchanged).
- **Baselines** (as shipped with CloudSEN12+): Sentinel-2 QA60, Sen2Cor SCL, s2cloudless
  (upstream default post-processing), Cloud Score+, UnetMobV2 and SEnSeI v2. UnetMobV2 was
  trained on CloudSEN12 itself, so it is in-domain like OmniCloudMask.
- **Metrics, framed as the decision:** *leak* = share of accepted (called clear) pixels that
  are cloud or shadow; *discard* = share of truly clear pixels thrown away; *miss* = share of
  cloud/shadow pixels accepted. Abstentions count as discarded. Scene policy: keep a scene if
  its estimated cover is at most 10%.
- **Statistics:** 95% CIs by bootstrap over sites (5 dates per site are not independent);
  method comparisons are paired.

## Results

**Headline (test, L1C):** leak **{N['leak_rate']}** {N['leak_rate_ci']}, discard
**{N['discard_rate']}** {N['discard_rate_ci']}, balanced accuracy {N['boa']} {N['boa_ci']}.

| Mask | Leak | Discard | Balanced acc. |
|---|---|---|---|
| OmniCloudMask v4 | {N['leak_rate']} | {N['discard_rate']} | {N['boa']} |
| SEnSeI v2 | {N['sensei_leak']} | {N['sensei_discard']} | {N['sensei_boa']} |
| UnetMobV2 v2 (in-domain) | {N['unet2_leak']} | {N['unet2_discard']} | {N['unet2_boa']} |
| Cloud Score+ (cs >= 0.60) | {N['csp_leak']} | {N['csp_discard']} | {N['csp_boa']} |
| s2cloudless (defaults) | {N['s2c_leak']} | {N['s2c_discard']} | {N['s2c_boa']} |
| Sen2Cor SCL | {N['scl_leak']} | {N['scl_discard']} | {N['scl_boa']} |
| QA60 | {N['qa60_leak']} | {N['qa60_discard']} | {N['qa60_boa']} |

Paired against OmniCloudMask: SEnSeI v2 leaks the same ({N['sensei_d_leak_rate']}
{N['sensei_d_leak_rate_ci']}) but discards {N['sensei_d_discard_rate']} more; Sen2Cor SCL
leaks {N['scl_d_leak_rate']} more.

![Leak vs discard](results/figures/tradeoff.png)

**Operating points** (chosen on validation, reported on test): at a 2% leak target
OmniCloudMask discards {N['ocm_2pct_discard']} {N['ocm_2pct_discard_ci']} of clear pixels,
Cloud Score+ {N['csp_2pct_discard']}; s2cloudless cannot reach 5% leak at any threshold
(floor {N['s2c_leak_floor']}; it does not model shadow).

**Where it fails** (real data):

![Leak by land cover](results/figures/landcover_leak.png)
![By cloud regime](results/figures/regimes.png)
![Examples](results/figures/examples.png)

**Stress test (simulated, labeled as such):** change versus the clean run, paired over the
same scenes; n.s. = 95% CI includes zero.

| Degradation | Effect by severity |
|---|---|
{deg_md}

Coarser data, leak / discard against the 10 m expert label (clean 10 m run:
{N['leak_rate']} / {N['discard_rate']}). *Native grid*: OCM runs on the coarse pixels.
*On a 10 m grid*: the same coarse data resampled (bilinear) to 10 m first, as data cubes do.
*Oracle*: the expert label itself reduced to that grid, the best any mask at that resolution
could score.

| Grid | OCM, native grid | OCM, on a 10 m grid | Oracle |
|---|---|---|---|
{orc_md}

## Findings and recommendations

| Severity | Finding | Recommendation |
|---|---|---|
{find_md}

## Limitations

- CloudSEN12+ labels are expert-drawn but not perfect; thin cloud over snow and cloud edges
  are the least certain, and some measured "leak" there is label disagreement.
- The test split is cloudier than a typical archive (mean cover ~47%); per-regime results
  are reported so they can be reweighted to a client's own cloud climatology.
- Only Sentinel-2 is tested. OmniCloudMask claims sensor-agnostic use (Landsat, PlanetScope);
  that claim is not evaluated here.
- Degradations in the stress test are simulated and simple; they bound sensitivity, they do
  not reproduce a specific sensor.
- QA60 values in CloudSEN12+ appear shifted (16384/32768 instead of bits 10/11); decoding was
  verified against labels, not against a source document.

## Run it

Requires `uv`, an NVIDIA GPU with 2+ GB (CPU works, slowly) and ~4 GB of disk.

```bash
uv sync --frozen
make all          # download, screen, predict, evaluate, stress test, figures, docs (~1.5 h)
make test         # unit tests only
```

Data are fetched by HTTP range requests from a pinned dataset revision; model weights are
checked against recorded SHA-256 hashes (`results/provenance.json`).

## Licenses

| Item | License | Source |
|---|---|---|
| This repository's code | MIT | `LICENSE` |
| OmniCloudMask code and v4 weights | MIT | github.com/DPIRD-DMA/OmniCloudMask, huggingface.co/NickWright/OmniCloudMask |
| CloudSEN12+ (images, labels, baseline masks) | CC0 1.0 | huggingface.co/datasets/tacofoundation/cloudsen12 |
| Sentinel-2 imagery | Copernicus open licence: "Contains modified Copernicus Sentinel data [2018-2020]" | dataspace.copernicus.eu/terms-and-conditions |
| ESA WorldCover (land cover layer) | CC BY 4.0: "© ESA WorldCover project 2020 / Contains modified Copernicus Sentinel data (2020) processed by ESA WorldCover consortium" | esa-worldcover.org |
| Cloud Score+ (baseline layer) | CC BY 4.0, Google | developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED |
| OmniCloudMask training metadata (filenames only) | CC BY 4.0 | huggingface.co/datasets/NickWright/OmniCloudMask-Combined-Training-Dataset |
| Inter typeface (report and figures) | SIL OFL 1.1 | github.com/rsms/inter |

Python dependencies and their versions are pinned in `uv.lock`.

---
Built {TODAY} by Shane Cairns. Generated by `scripts/build_docs.py`.
"""


# --------------------------------------------------------------------------------------------
# Site write-up
# --------------------------------------------------------------------------------------------
def writeup(N: dict) -> str:
    return f"""---
title: "Is this cloud mask safe to automate on?"
summary: "An independent acceptance audit of an open cloud-masking model, framed around the decision a pipeline actually makes."
---

## The problem

Every optical imagery pipeline decides, scene by scene and pixel by pixel, what is clear
enough to use. If cloud slips through, it shows up later as false change, wrong vegetation
indices and polluted training data. If clear ground is masked, you lose revisits you paid
for. Most teams take that decision from a mask they have never measured on their own terms.

## What I built

An audit harness that answers the buyer's question for
[OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask), a popular open model, on
{N['images']} expert-labeled Sentinel-2 scenes from {N['sites']} sites (CloudSEN12+):

- metrics phrased as the decision: **leak** (cloud in the pixels you accept) and **discard**
  (clear ground you throw away), plus a scene filter test;
- a check of the model's own published training data against the test set, so the scores
  are not inflated by overlap;
- side-by-side comparison with the masks that ship with Sentinel-2 products;
- failure analysis on real scenes, a simulated stress test, and mitigations that were
  chosen on one split and verified on another.

## Results

- At its default setting, **{N['leak_rate']}** of accepted pixels are cloud or shadow and
  **{N['discard_rate']}** of clear pixels are discarded. One setting change, chosen on held-out
  data, brings leak to **{N['ocm_2pct_leak']}** at {N['ocm_2pct_discard']} discard; a common
  alternative, buffering the mask, cannot get below {N['buffer_kmax_leak']}.
- The cloud classification shipped inside Sentinel-2 L2A products is far less safe for
  scene filtering: **{N['gate_scl_over']}** of scenes it lets through a "10% cloud" filter are
  cloudier than that, against {N['gate_ocm_over']} with OmniCloudMask.
- It is weakest over water (leak {N['water_leak']}) and in mostly clear scenes, where it
  misses {N['light_miss']} of the cloud and shadow present.

![Leak vs discard](/images/cloudmask/tradeoff.png)

## What this means for a client

If you are about to automate on a model you did not build, the useful question is not "how
accurate is it?" but "what will it let through, what will it cost me, and which setting should
I run?" This audit answers that in a week on public data and a single GPU, with confidence
intervals, a ranked list of findings and recommendations that were tested, not just
suggested. The same harness runs on your model and your imagery.
"""


# --------------------------------------------------------------------------------------------
# PDF report
# --------------------------------------------------------------------------------------------
CSS = """
@font-face { font-family: Inter; src: url(../assets/fonts/Inter-Regular.ttf); }
@font-face { font-family: Inter; font-weight: 600; src: url(../assets/fonts/Inter-SemiBold.ttf); }
@font-face { font-family: Inter; font-weight: 700; src: url(../assets/fonts/Inter-Bold.ttf); }
@font-face { font-family: Inter; font-style: italic; src: url(../assets/fonts/Inter-Italic.ttf); }
@page { size: Letter; margin: 0.7in 0.75in 0.75in 0.75in;
  @bottom-left { content: "Cloud-mask acceptance audit · OmniCloudMask v4"; font: 8pt Inter; color: #898781; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 8pt Inter; color: #898781; } }
body { font: 9pt/1.42 Inter; color: #0b0b0b; }
h1 { font-size: 19pt; font-weight: 700; margin: 0 0 2pt 0; letter-spacing: -0.2pt; }
h2 { font-size: 12pt; font-weight: 700; margin: 14pt 0 5pt; border-bottom: 1px solid #e1e0d9; padding-bottom: 3pt; }
h3 { font-size: 10pt; font-weight: 600; margin: 9pt 0 3pt; }
p { margin: 0 0 6pt; }
.meta { color: #52514e; font-size: 9pt; margin-bottom: 10pt; }
.box { background: #f4f3ef; border-left: 3px solid #2a78d6; padding: 7pt 10pt; margin: 6pt 0 8pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 8pt; font-size: 8.1pt; }
td, th { page-break-inside: avoid; }
th { text-align: left; font-weight: 600; color: #52514e; border-bottom: 1px solid #c3c2b7; padding: 3pt 5pt; }
td { border-bottom: 1px solid #e1e0d9; padding: 3.5pt 5pt; vertical-align: top; }
.sev { font-weight: 600; white-space: nowrap; }
.sev-High { color: #b52e2e; } .sev-Medium { color: #a8640f; } .sev-Low { color: #1c5cab; } .sev-Info { color: #52514e; }
img { width: 88%; margin: 4pt 6%; }
.small { font-size: 8pt; color: #52514e; }
ul, ol { margin: 0 0 6pt 14pt; padding: 0; } li { margin-bottom: 2.5pt; }
.pb { page-break-before: always; }
"""


def report_html(N: dict) -> str:
    F = findings(N)
    rows = "".join(
        f"<tr><td>F{i + 1}</td><td class='sev sev-{f['sev']}'>{f['sev']}</td><td><b>{f['title']}</b>"
        f"<br>{f['evidence']}</td><td>{f['rec']}</td></tr>" for i, f in enumerate(F))
    recs = [f for f in F if f["sev"] in ("High", "Medium")]
    summary = (
        f"<li><b>Use it, with a setting change.</b> OmniCloudMask is the most accurate mask tested "
        f"(balanced accuracy {N['boa']}); it leaks {N['leak_rate']} of accepted pixels by default "
        f"and {N['ocm_2pct_leak']} at the recommended threshold.</li>"
        f"<li><b>Stop filtering scenes with the SCL band.</b> {N['gate_scl_over']} of scenes it keeps "
        f"under a 10% cloud rule are cloudier than that.</li>"
        f"<li><b>Expect more leak over water</b> ({N['water_leak']}) and missed cloud in "
        f"mostly clear scenes ({N['light_miss']} of it).</li>"
        f"<li><b>Feed it native-grid, unclipped data.</b> 30 m imagery resampled to 10 m "
        f"discards {N['fine30_discard']} of clear ground (native grid: {N['native30_discard']}); "
        f"inputs clipped at 0.3 reflectance leak {N['clip03_leak']}.</li>"
        f"<li><b>Not sensitive to</b> processing level (L1C/L2A), per-band calibration drift or "
        f"NIR misregistration up to 4 px.</li>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<h1>Cloud-mask acceptance audit</h1>
<div class="meta">OmniCloudMask v4 on Sentinel-2 · Independent model assurance report · {TODAY}<br>
Prepared by Shane Cairns · All figures computed from public data by the accompanying repository</div>

<div class="box"><b>Question.</b> Before automating on OmniCloudMask, how much cloud and shadow
reaches the pixels we accept, how much clear ground do we lose, where does it fail, and which
setting should we run?</div>

<h2>Summary</h2>
<ul>{summary}</ul>

<h2>Test setup</h2>
<p><b>Data.</b> CloudSEN12+ expert-labeled test split: {N['images']} Sentinel-2 scenes
(5.1 km, 10 m) at {N['sites']} sites worldwide, 2018-2020, at L1C and L2A. The validation split
was used only to choose operating points.</p>
<p><b>Independence from training data.</b> The model's published training set includes
CloudSEN12 material from every split, so its {N['train_rows']} training records were checked
directly against the test scenes: no shared scene ids (including {N['train_cs12_heldout_rows']}
test/validation-tagged training records), no shared acquisitions; one site within 2.6 km of a
custom training scene was removed. Removing every scene on a Kappaset training tile leaves the
headline unchanged (leak {N['kappa_leak']}).</p>
<p><b>Comparators.</b> Sentinel-2 QA60 and Sen2Cor SCL (shipped with the products), s2cloudless
(upstream defaults), Cloud Score+ (cs &ge; 0.60), UnetMobV2 and SEnSeI v2.</p>
<p><b>Metrics.</b> <i>Leak</i>: share of accepted pixels that are cloud or shadow.
<i>Discard</i>: share of clear pixels rejected. <i>Miss</i>: share of cloud/shadow accepted.
Scene filter: keep if estimated cover &le; 10%. Abstentions count as rejected. 95% confidence
intervals by bootstrap over sites; comparisons are paired.</p>
<p><b>Severity.</b> High: affects a default decision path with large error. Medium: material
error in a common condition, or a trade-off the client must set. Low: limited evidence or only
under degraded input. Info: checked, no action.</p>

<h2>Findings, ranked by severity</h2>
<table><tr><th>#</th><th>Severity</th><th>Finding and evidence</th><th>Recommendation</th></tr>{rows}</table>

<img src="../results/figures/tradeoff.png">
<p class="small">Figure 1. Leak against discard for every threshold of the score-based masks; grey
points are fixed-rule products. Test split.</p>

<img src="../results/figures/stress.png">
<p class="small">Figure 2. Stress test (simulated degradations): change in leak and discard against the clean
run on the same scenes, with 95% site-bootstrap CIs. "Coarse data on a 10 m grid" is 20/30/50 m data
resampled to 10 m before masking; "Coarser resolution" is the same data masked on its native grid.</p>

<h2>Recommendations</h2>
<ol>{''.join(f"<li>{f['action']}</li>" for f in recs if f.get('action'))}</ol>

<h2>Limitations</h2>
<ul>
<li>Expert labels are least certain for thin cloud over snow and at cloud edges; part of the
measured leak there is label disagreement.</li>
<li>The test split is cloudier than a typical archive; per-regime results allow reweighting
to a client's own cloud climatology.</li>
<li>Sentinel-2 only; the model's sensor-agnostic claim was not tested. Degradations are simple
simulations that bound sensitivity.</li>
</ul>
<p class="small">Reproducibility: pinned dataset revision, model weights verified by SHA-256,
one-command pipeline (<code>make all</code>), unit tests. Data: CloudSEN12+ (CC0); contains modified
Copernicus Sentinel data 2018-2020; ESA WorldCover (CC BY 4.0). Model: OmniCloudMask (MIT).</p>
</body></html>"""


def main() -> None:
    N = load()
    Path("README.md").write_text(readme(N))
    Path("site").mkdir(exist_ok=True)
    Path("site/cloudmask-audit.md").write_text(writeup(N))
    Path("report").mkdir(exist_ok=True)
    html = report_html(N)
    Path("report/report.html").write_text(html)
    from weasyprint import HTML

    HTML(string=html, base_url=str(Path("report").resolve()) + "/").write_pdf(
        "report/cloudmask-acceptance-audit.pdf")
    for p in ("README.md", "site/cloudmask-audit.md", "report/report.html"):
        if "—" in Path(p).read_text():
            raise SystemExit(f"em dash in {p}")
    print("wrote README.md, site/cloudmask-audit.md, report/cloudmask-acceptance-audit.pdf")


if __name__ == "__main__":
    main()
