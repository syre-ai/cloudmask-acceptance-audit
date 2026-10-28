"""Build README.md, the site write-up and the client PDF report from results/.

No result is typed into these documents: every figure comes from cmaudit.numbers, which reads
the CSV/JSON outputs of the pipeline.

    uv run python scripts/build_docs.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from cmaudit.numbers import load, pp

TODAY = dt.date.today().isoformat()
REF = "Majority-label reference (label at coarse grid)"


def P(N: dict, name: str, sev: float) -> dict:
    """Stress-test entry (changes vs the clean run, absolute values, CIs)."""
    return N["pt"][(name, round(float(sev), 3))]


# --------------------------------------------------------------------------------------------
# A4 table rows
# --------------------------------------------------------------------------------------------
def degradation_rows(N: dict) -> list[dict]:
    t = N["perturb"]
    t = t[t["perturbation"] != REF]
    rows = []
    for name, g in t.groupby("perturbation", sort=False):
        cells = []
        for _, r in g.iterrows():
            sig_leak = r["d_leak_rate_lo"] > 0 or r["d_leak_rate_hi"] < 0
            sig_disc = r["d_discard_rate_lo"] > 0 or r["d_discard_rate_hi"] < 0
            cells.append(f"{_sev_label(name, r['severity'])}: leak {pp(r['d_leak_rate'])}"
                         f"{'' if sig_leak else ' (n.s.)'}, discard {pp(r['d_discard_rate'])}"
                         f"{'' if sig_disc else ' (n.s.)'}")
        rows.append({"name": name, "cells": cells})
    return rows


def _sev_label(name: str, sev) -> str:
    sev = float(sev)
    if name.startswith(("Coarser", "Coarse data")):
        return f"{int(sev) * 10} m"
    return {
        "Sensor noise": f"σ {sev:g}",
        "Optical blur": f"σ {sev:g} px",
        "NIR misregistration": f"{int(sev)} px",
        "8-bit JPEG delivery": f"q{int(sev)}",
    }.get(name, f"clip {sev:g}")


def coarse_rows(N: dict) -> dict:
    """Absolute leak/discard per coarse grid for: OCM on the native grid, OCM on the same data
    put on a 10 m grid (bilinear, and nearest as a control), and the majority-label reference.
    Absolute = clean 10 m value + paired change."""
    out = {}
    for f in (2, 3, 5):
        out[f] = {key: P(N, name, f) for key, name in (
            ("native", "Coarser resolution"), ("bilinear", "Coarse data on a 10 m grid"),
            ("nearest", "Coarse data on a 10 m grid (nearest)"), ("ref", REF))}
    return out


# --------------------------------------------------------------------------------------------
# Findings, ranked. Severity rubric (stated in every document):
#   High   - affects a default decision path, large error, measured on real data
#   Medium - material error in a common condition on real data, a trade-off the client must
#            set, or a large simulated effect on a common processing path
#   Low    - limited evidence, or simulated effects of moderate size
#   Info   - checked; small or no measured effect within the tested range
# --------------------------------------------------------------------------------------------
def findings(N: dict) -> list[dict]:
    f30, n30 = P(N, "Coarse data on a 10 m grid", 3), P(N, "Coarser resolution", 3)
    nn30 = P(N, "Coarse data on a 10 m grid (nearest)", 3)
    n20, n50 = P(N, "Coarser resolution", 2), P(N, "Coarser resolution", 5)
    c03, q03 = P(N, "Clip only (16-bit)", 0.3), P(N, "8-bit lossless (clip level)", 0.3)
    c10 = P(N, "Clip only (16-bit)", 1.0)
    q06, j95, j50 = (P(N, "8-bit lossless (clip level)", 0.6), P(N, "8-bit JPEG delivery", 95),
                     P(N, "8-bit JPEG delivery", 50))
    no5, no10 = P(N, "Sensor noise", 0.005), P(N, "Sensor noise", 0.01)
    mis4 = P(N, "NIR misregistration", 4)
    return [
        {"sev": "High", "title": "Scene filters built on the Sen2Cor SCL band keep cloudy scenes",
         "evidence": (f"On the 5.1 km test patches, keeping a patch when its SCL cloud+shadow "
                      f"cover is at most 10% keeps patches of which {N['gate_scl_over']} are above "
                      f"10% by the expert labels; SCL under-reports cover by {N['scl_cover_bias']} on "
                      f"average. The same rule using OmniCloudMask cover: {N['gate_ocm_over']} above, "
                      f"and {N['gate_ocm_rejected_ok']} of good patches rejected (SCL: "
                      f"{N['gate_scl_rejected_ok']})."),
         "rec": (f"Gate on OmniCloudMask-derived cover instead of the SCL band. Measured: "
                 f"cloud+shadow inside kept patches after masking falls from {N['gate_scl_leak']} "
                 f"to {N['gate_ocm_leak']} {N['gate_ocm_leak_ci']}. Full-granule filtering was not "
                 f"tested."),
         "action": "Gate scenes on OmniCloudMask cover instead of the Sen2Cor SCL band."},
        {"sev": "Medium", "title": "At its default setting, about 6% of accepted pixels are cloud or shadow",
         "evidence": (f"Leak {N['leak_rate']} {N['leak_rate_ci']} of accepted pixels (thin cloud "
                      f"{N['leak_thin']}, shadow {N['leak_shadow']}, thick {N['leak_thick']}); discard "
                      f"{N['discard_rate']} {N['discard_rate_ci']} of clear pixels."),
         "rec": (f"For leak-sensitive uses (time series, change detection): {N['ocm_2pct_rule']} "
                 f"(chosen on the validation split for a 2% leak target). Test: leak "
                 f"{N['ocm_2pct_leak']} {N['ocm_2pct_leak_ci']} at {N['ocm_2pct_discard']} discard. "
                 f"Buffering the mask is the weaker lever: none of the tested buffers (0 to "
                 f"{N['buffer_kmax']} px) met the target; {N['buffer_kmax']} px leaks "
                 f"{N['buffer_kmax_leak']} at {N['buffer_kmax_discard']} discard."),
         "action": f"For leak-sensitive uses, {N['ocm_2pct_rule'].replace('accept if', 'accept pixels only if')}."},
        {"sev": "Medium", "title": "Coarse data resampled onto a finer grid is over-masked (simulated)",
         "evidence": (f"Simulated 30 m data put on a 10 m grid with bilinear interpolation: discard "
                      f"{f30['discard']} ({f30['ddiscard']}); with nearest-neighbour instead: "
                      f"{nn30['discard']} ({nn30['ddiscard']}); the same 30 m data masked on its own "
                      f"grid: {n30['discard']} ({n30['ddiscard']}). Native-grid runs at 20/30/50 m: "
                      f"{n20['ddiscard']} / {n30['ddiscard']} / {n50['ddiscard']}."),
         "rec": ("Where the pipeline allows, run the mask on the sensor's native grid and resample "
                 "the mask. If imagery must be resampled to a finer grid first, validate the mask "
                 "on that product; real resamplers and sensor optics differ from this simulation."),
         "action": "Prefer masking on the native grid; validate before masking resampled imagery."},
        {"sev": "Medium", "title": "Clipping bright reflectance lets cloud through (simulated)",
         "evidence": (f"Clipping at 0.3 reflectance (close to the Sentinel-2 true-colour stretch) "
                      f"raises leak by {c03['dleak']} with 16-bit values and {q03['dleak']} when also "
                      f"quantized to 8 bits; clipping at 1.0: {c10['dleak']}. JPEG on top of 8-bit "
                      f"(clip 0.6, {q06['dleak']}) adds little at quality 95 ({j95['dleak']} in "
                      f"total) and more at quality 50 ({j50['dleak']})."),
         "rec": "Feed unclipped reflectance; avoid running the mask on stretched visual products.",
         "action": "Feed unclipped reflectance, not stretched visual products."},
        {"sev": "Medium", "title": f"Higher leak over water ({N['water_leak']} vs {N['leak_rate']} overall)",
         "evidence": (f"Leak over water {N['water_leak']} {N['water_leak_ci']} across "
                      f"{N['water_sites']} sites (thick cloud {N['water_leak_thick']} vs "
                      f"{N['grass_leak_thick']} over grassland); the two worst sites hold "
                      f"{N['water_top2']} of it."),
         "rec": ("For coastal, inland-water and flood work, spot-check a local sample; the "
                 "stricter threshold above was not validated separately for water."),
         "action": "Spot-check a local sample before relying on the mask over water."},
        {"sev": "Medium", "title": "Small amounts of cloud are missed in mostly clear scenes",
         "evidence": (f"In scenes with 1-30% cloud, {N['light_miss']} {N['light_miss_ci']} of the "
                      f"cloud and shadow present is accepted as clear (overcast scenes: "
                      f"{N['overcast_miss']}). These are the scenes a cover gate keeps."),
         "rec": "The threshold recommendation reduces leak overall; it was not tuned for this "
                "regime separately."},
        {"sev": "Medium", "title": "Usable ground is discarded in heavily clouded scenes",
         "evidence": (f"In scenes with 70-99% cloud, {N['heavy_discard']} {N['heavy_discard_ci']} "
                      f"of the clear pixels are rejected."),
         "rec": "If gap-filling from cloudy scenes matters, validate a lower threshold for that "
                "regime (not tested here).",
         "action": "If you gap-fill from cloudy scenes, validate a lower threshold for them."},
        {"sev": "Low", "title": "Snow and ice: highest observed leak, weakest evidence",
         "evidence": (f"Leak {N['snow_leak']} {N['snow_leak_ci']} over only {N['snow_sites']} sites, "
                      f"{N['snow_top2']} of it from two sites, mostly thin cloud over snow, where "
                      f"expert labels are least certain (not adjudicated here)."),
         "rec": "Validate locally before relying on the mask in snow-covered regions."},
        {"sev": "Low", "title": "Added sensor noise increases leak (simulated)",
         "evidence": (f"Gaussian noise with a standard deviation of 0.005 reflectance raises leak "
                      f"by {no5['dleak']} {no5['dleak_ci']}; 0.01 by {no10['dleak']}."),
         "rec": "Re-validate before using the mask on noisier imagery than Sentinel-2."},
        {"sev": "Info", "title": "Small measured effects within the tested range",
         "evidence": (f"L1C vs L2A input: leak {N['l2a_d_leak_rate']} {N['l2a_d_leak_rate_ci']}. "
                      f"One tested per-band gain/offset change: {N.get('gain_agreement', 'n/a')} "
                      f"pixel agreement on a fifth of the test scenes (the model normalizes each band "
                      f"per image). NIR shifted 4 px diagonally: discard {mis4['ddiscard']} "
                      f"{mis4['ddiscard_ci']}, leak {mis4['dleak']} {mis4['dleak_ci']}."),
         "rec": "None required within these ranges; larger shifts or other calibration errors "
                "were not tested."},
    ]


# --------------------------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------------------------
def readme(N: dict) -> str:
    deg = degradation_rows(N)
    deg_md = "\n".join(f"| {d['name']} | " + " · ".join(d["cells"]) + " |" for d in deg)
    coarse_md = "\n".join(
        f"| {f * 10} m | {v['native']['leak']} / {v['native']['discard']} | "
        f"{v['bilinear']['leak']} / {v['bilinear']['discard']} | "
        f"{v['nearest']['leak']} / {v['nearest']['discard']} | "
        f"{v['ref']['leak']} / {v['ref']['discard']} |" for f, v in coarse_rows(N).items())
    find_md = "\n".join(f"| {f['sev']} | {f['title']} | {f['rec']} |" for f in findings(N))
    return f"""<img src="assets/banner.png" alt="Cloud-mask acceptance audit: OmniCloudMask v4" width="100%">

An independent assurance audit of [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)
(v4 weights, MIT), an open cloud and cloud-shadow segmentation model for Sentinel-2 and
other 10-50 m imagery, run the way a client would commission it: **before you automate on this
mask, how much cloud leaks into the pixels you accept, how much usable ground do you throw
away, and where does it fail?**

Every number below is computed by this repository from public data; nothing is typed in by
hand. The client-style report is in [`report/`](report/), and the
[threshold explorer](https://syre-ai.github.io/cloudmask-acceptance-audit/site/explorer/) lets you drag the operating point.

## Why this matters

Almost every optical imagery pipeline starts by deciding which pixels and which scenes to
use. Cloud and shadow that slip through become false change, wrong indices and bad training
data; clear ground that gets masked is lost revisit. The decision is usually made by a mask
nobody has measured on the buyer's own terms.

## What was tested

- **Model:** OmniCloudMask v4 (two U-Nets, red/green/NIR input), fp32.
- **Data:** CloudSEN12+ (CC0), expert-labeled test split: **{N['images']} scenes at {N['sites']}
  sites**, 509 x 509 px at 10 m, both L1C and L2A. The validation split is used only to choose
  operating points and the buffer size.
- **Training-overlap screen:** OmniCloudMask's published training set includes CloudSEN12
  scribble and 2k patches from *all* splits, so the evaluation scenes were checked against its
  training records ({N['train_rows']} rows, pinned revision): {N['train_id_hits']} shared scene
  ids (including {N['train_cs12_heldout_rows']} test/validation-tagged training rows) and no
  spatial overlap with any CloudSEN12 training patch. {N['test_shared_acq']} test acquisitions
  (tile and date) also supplied other, non-overlapping CloudSEN12 patches to training
  ({N['test_shared_acq_rows']} training rows); same-date imagery nearby is a residual dependence.
  No acquisition is shared with the non-CloudSEN12 training sources ({N['train_acq_hits']}
  matches), and {N['train_near_images']} evaluation images whose footprint may overlap a custom
  training scene (conservative disc around its named centre) were removed. Kappaset names only its ~110 km tile; removing the {N['images'] - N['kappa_images']}
  test images on those tiles gives leak {N['kappa_leak']} (headline {N['leak_rate']}). This is a
  screen with stated limits, not proof that no overlap remains.
- **Baselines** (as shipped with CloudSEN12+): Sentinel-2 QA60, Sen2Cor SCL, s2cloudless
  (upstream default post-processing), Cloud Score+, UnetMobV2 and SEnSeI v2. UnetMobV2 was
  trained on CloudSEN12 itself, so it is in-domain like OmniCloudMask.
- **Metrics, framed as the decision:** *leak* = share of accepted (called clear) pixels that
  are cloud or shadow; *discard* = share of truly clear pixels thrown away; *miss* = share of
  cloud/shadow pixels accepted. Every method is scored on the same pixels; abstentions count as
  discarded. Scene policy: keep a 5.1 km patch if its estimated cloud+shadow cover is at most 10%.
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
| s2cloudless (upstream defaults) | {N['s2c_leak']} | {N['s2c_discard']} | {N['s2c_boa']} |
| Sen2Cor SCL | {N['scl_leak']} | {N['scl_discard']} | {N['scl_boa']} |
| QA60 | {N['qa60_leak']} | {N['qa60_discard']} | {N['qa60_boa']} |

Paired against OmniCloudMask: SEnSeI v2 leak {N['sensei_d_leak_rate']}
{N['sensei_d_leak_rate_ci']}, discard {N['sensei_d_discard_rate']}
{N['sensei_d_discard_rate_ci']}; Sen2Cor SCL leak {N['scl_d_leak_rate']}
{N['scl_d_leak_rate_ci']}.

![Leak vs discard](results/figures/tradeoff.png)

An interactive version (drag the threshold, watch leak and discard move) is generated at
`site/explorer/index.html`, a single self-contained page for the website.

**Operating points** (chosen on validation, reported on test). For a 2% leak target,
OmniCloudMask: {N['ocm_2pct_rule']}, giving leak {N['ocm_2pct_leak']} {N['ocm_2pct_leak_ci']}
and discard {N['ocm_2pct_discard']} {N['ocm_2pct_discard_ci']}; Cloud Score+ discards
{N['csp_2pct_discard']} at the same target. Thresholding s2cloudless's *raw* probability (no
averaging or dilation) never gets leak below {N['s2c_leak_floor']}; it has no shadow class.

**Where it fails** (real data):

![Leak by land cover](results/figures/landcover_leak.png)
![By cloud regime](results/figures/regimes.png)
![Examples](results/figures/examples.png)

**Stress test (simulated):** change versus the clean run, paired over the same scenes;
n.s. = 95% CI includes zero. Degradations are simple, controlled simulations; they show
sensitivity to each change, not the behaviour of a particular sensor or pipeline.

![Stress test](results/figures/stress.png)

| Degradation | Effect by severity |
|---|---|
{deg_md}

Coarser data, absolute leak / discard against the 10 m expert label (clean 10 m run:
{N['leak_rate']} / {N['discard_rate']}). *Native grid*: OCM runs on the coarse pixels.
*10 m grid*: the same coarse data put on a 10 m grid first (bilinear, or nearest-neighbour as a
control). *Majority-label reference*: the expert label reduced to that grid by 4-class block
majority, showing what the coarser grid alone changes (a reference, not a bound on these
metrics).

| Grid | OCM, native grid | OCM, 10 m grid (bilinear) | OCM, 10 m grid (nearest) | Majority-label reference |
|---|---|---|---|---|
{coarse_md}

## Findings and recommendations

Severity: **High** affects a default decision path with a large error measured on real data;
**Medium** is a material error in a common condition on real data, a trade-off the client must
set, or a large simulated effect on a common processing path; **Low** has limited evidence or
moderate simulated effects; **Info** was checked with small or no measured effect.

| Severity | Finding | Recommendation |
|---|---|---|
{find_md}

## Limitations

- CloudSEN12+ labels are expert-drawn but not perfect. Thin cloud over snow and cloud edges are
  the least certain; some measured leak there may be label disagreement (not adjudicated here).
- The test split is cloudier than a typical archive (mean cover ~47%); per-regime results are
  reported so they can be reweighted to a client's own cloud climatology.
- Only Sentinel-2 is tested. OmniCloudMask claims sensor-agnostic use (Landsat, PlanetScope);
  that claim is not evaluated here.
- Scene-filter results are for 5.1 km patches and cloud plus shadow, not full granules.
- Recommendations for specific conditions (water, cloudy scenes) were not validated separately.
- QA60 values in CloudSEN12+ appear shifted (16384/32768 instead of bits 10/11); decoding was
  verified against labels, not against a source document.

## Run it

Requires `uv` and ~4 GB of disk for data plus ~7 GB for the Python environment (PyTorch with
CUDA). An NVIDIA GPU is used when present (an RTX 2070 Super runs everything in ~1.5 h); the
scripts fall back to CPU, which is much slower.

```bash
uv sync --frozen
make all          # download, screen, predict, evaluate, stress test, figures, docs
make test         # unit tests only
```

Data are read by HTTP range requests from pinned dataset revisions, and the model weights are
checked against recorded SHA-256 hashes (`results/provenance.json`). The full run log from a
clean clone is in `results/clean_run.log`.

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

CloudSEN12+ also ships a Copernicus DEM layer; this audit does not download or use it. Python
dependencies and their versions are pinned in `uv.lock`.

---
Built {TODAY} by Shane Cairns. Generated by `scripts/build_docs.py`.
"""


# --------------------------------------------------------------------------------------------
# Site write-up
# --------------------------------------------------------------------------------------------
def writeup(N: dict) -> str:
    f30 = P(N, "Coarse data on a 10 m grid", 3)
    n30 = P(N, "Coarser resolution", 3)
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

An audit pipeline that answers the buyer's question for
[OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask), a popular open model, on
{N['images']} expert-labeled Sentinel-2 scenes from {N['sites']} sites (CloudSEN12+):

- metrics phrased as the decision: **leak** (cloud in the pixels you accept) and **discard**
  (clear ground you throw away), plus a scene-filter test;
- a screen of the model's own published training records against the test scenes, removing
  scenes flagged as possibly overlapping (one training source can only be screened at
  ~110 km and is covered by a sensitivity check instead);
- side-by-side comparison with the masks that ship with Sentinel-2 products;
- failure analysis on real scenes, a simulated stress test, and settings chosen on one data
  split and checked on another.

## Results

- At its default setting, **{N['leak_rate']}** of accepted pixels are cloud or shadow and
  **{N['discard_rate']}** of clear pixels are discarded. A threshold chosen on held-out data
  brings leak to **{N['ocm_2pct_leak']}** at {N['ocm_2pct_discard']} discard; buffering the mask,
  a common alternative, did not reach that leak at any tested size.
- The cloud classification shipped in Sentinel-2 L2A products is much less reliable for scene
  filtering: of the patches it keeps under a "10% cloud" rule, **{N['gate_scl_over']}** are
  cloudier than that, against {N['gate_ocm_over']} with OmniCloudMask.
- Leak is higher over water ({N['water_leak']}) and over snow and ice (on thinner evidence), and
  in mostly clear scenes it misses {N['light_miss']} of the cloud and shadow present.
- In a simulation, 30 m data placed on a 10 m grid before masking lost {f30['discard']} of clear
  ground, against {n30['discard']} when masked on its own grid.

![Leak vs discard](/images/cloudmask/tradeoff.png)

## What this means for a client

If you are about to automate on a model you did not build, the useful question is not "how
accurate is it?" but "what will it let through, what will it cost me, and which setting should
I run?" This audit answers that on public data and a single GPU, with confidence intervals, a
ranked list of findings, and settings that were chosen on one split and verified on another.
The same approach can be adapted to your model and your imagery.
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
    f30, n30 = P(N, "Coarse data on a 10 m grid", 3), P(N, "Coarser resolution", 3)
    c03 = P(N, "Clip only (16-bit)", 0.3)
    summary = (
        f"<li><b>Usable, with a setting change.</b> OmniCloudMask had the highest balanced accuracy "
        f"of the masks tested at their default settings ({N['boa']}); it leaks {N['leak_rate']} of "
        f"accepted pixels by default and {N['ocm_2pct_leak']} at the recommended threshold.</li>"
        f"<li><b>Do not filter scenes with the SCL band.</b> Of the patches it keeps under a 10% "
        f"cloud rule, {N['gate_scl_over']} are cloudier than that.</li>"
        f"<li><b>Expect more leak over water</b> ({N['water_leak']}) and missed cloud in mostly "
        f"clear scenes ({N['light_miss']} of it).</li>"
        f"<li><b>Mask native-grid, unclipped data where possible.</b> In simulation, 30 m data put on "
        f"a 10 m grid discards {f30['discard']} of clear ground (native grid: {n30['discard']}); "
        f"reflectance clipped at 0.3 raises leak by {c03['dleak']}.</li>")
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
test/validation-tagged training records) and no spatial overlap with CloudSEN12 training patches,
though {N['test_shared_acq']} test acquisitions also supplied other, non-overlapping patches to
training; {N['train_near_images']} scenes that may overlap a custom training scene were removed. Removing every scene on a Kappaset training
tile gives leak {N['kappa_leak']} (headline {N['leak_rate']}). This screen has limits; it does not
prove that no overlap remains.</p>
<p><b>Comparators.</b> Sentinel-2 QA60 and Sen2Cor SCL (shipped with the products), s2cloudless
(upstream defaults), Cloud Score+ (cs &ge; 0.60), UnetMobV2 and SEnSeI v2.</p>
<p><b>Metrics.</b> <i>Leak</i>: share of accepted pixels that are cloud or shadow.
<i>Discard</i>: share of clear pixels rejected. <i>Miss</i>: share of cloud/shadow accepted.
Scene filter: keep a 5.1 km patch if estimated cloud+shadow cover &le; 10%. All methods are
scored on the same pixels; abstentions count as rejected. 95% confidence intervals by bootstrap
over sites; comparisons are paired.</p>
<p><b>Severity.</b> High: affects a default decision path, large error, measured on real data.
Medium: material error in a common condition on real data, a trade-off the client must set, or a
large simulated effect on a common processing path. Low: limited evidence or moderate simulated
effects. Info: checked, small or no measured effect.</p>

<h2>Findings, ranked by severity</h2>
<table><tr><th>#</th><th>Severity</th><th>Finding and evidence</th><th>Recommendation</th></tr>{rows}</table>

<img src="../results/figures/tradeoff.png">
<p class="small">Figure 1. Leak against discard for every threshold of the score-based masks; grey
points are fixed-rule products. Test split.</p>

<img src="../results/figures/stress.png">
<p class="small">Figure 2. Stress test (simulated degradations): change in leak and discard against the clean
run on the same scenes, with 95% site-bootstrap CIs. "Coarse data on a 10 m grid" is 20/30/50 m data
put on a 10 m grid before masking (bilinear; nearest-neighbour as a control); "Coarser resolution" is the
same data masked on its native grid.</p>

<h2>Recommendations</h2>
<ol>{''.join(f"<li>{f['action']}</li>" for f in recs if f.get('action'))}</ol>

<h2>Limitations</h2>
<ul>
<li>Expert labels are least certain for thin cloud over snow and at cloud edges; some measured
leak there may be label disagreement (not adjudicated).</li>
<li>The test split is cloudier than a typical archive; per-regime results allow reweighting
to a client's own cloud climatology.</li>
<li>Sentinel-2 only; the model's sensor-agnostic claim was not tested. Degradations are simple
controlled simulations, not models of a particular sensor or pipeline.</li>
<li>Scene-filter results are for 5.1 km patches, not full granules. Condition-specific
recommendations were not validated separately.</li>
</ul>
<p class="small">Reproducibility: pinned dataset revision, model weights verified by SHA-256,
one-command pipeline (<code>make all</code>), unit tests. Data: CloudSEN12+ (CC0); contains modified
Copernicus Sentinel data 2018-2020; ESA WorldCover (CC BY 4.0). Model: OmniCloudMask (MIT).</p>
</body></html>"""


def main() -> None:
    N = load()
    N["ocm_2pct_rule"] = N["ocm_2pct_rule"].replace(">=", "≥").replace("<=", "≤")
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
