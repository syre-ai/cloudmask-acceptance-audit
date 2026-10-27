# Cloud-mask acceptance audit: OmniCloudMask v4

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
- **Data:** CloudSEN12+ (CC0), expert-labeled test split: **965 scenes at 193
  sites**, 509 x 509 px at 10 m, both L1C and L2A. Validation split used only to choose
  operating points.
- **Training-overlap screen:** OmniCloudMask's published training set lists CloudSEN12 scribble
  and 2k patches from *all* splits, so the screen was run against its actual training metadata
  (100,528 rows): 0 id matches with the evaluation images
  (including 2,754 test/validation-tagged training rows), 0
  same-acquisition matches, and one site (5 images) within 2.6 km of a custom training scene,
  excluded. Kappaset names only its 110 km tile; excluding all 50 images on those tiles
  leaves leak at 6.2% (unchanged).
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

**Headline (test, L1C):** leak **6.1%** [5.3, 7.0], discard
**4.6%** [4.0, 5.2], balanced accuracy 0.942 [0.937, 0.947].

| Mask | Leak | Discard | Balanced acc. |
|---|---|---|---|
| OmniCloudMask v4 | 6.1% | 4.6% | 0.942 |
| SEnSeI v2 | 6.2% | 6.2% | 0.935 |
| UnetMobV2 v2 (in-domain) | 7.1% | 5.3% | 0.933 |
| Cloud Score+ (cs >= 0.60) | 8.5% | 12.7% | 0.891 |
| s2cloudless (defaults) | 23.2% | 7.1% | 0.807 |
| Sen2Cor SCL | 28.5% | 11.0% | 0.747 |
| QA60 | 32.5% | 15.1% | 0.697 |

Paired against OmniCloudMask: SEnSeI v2 leaks the same (+0.1 pp
[-0.4, 0.6]) but discards +1.5 pp more; Sen2Cor SCL
leaks +22.4 pp more.

![Leak vs discard](results/figures/tradeoff.png)

**Operating points** (chosen on validation, reported on test): at a 2% leak target
OmniCloudMask discards 13.6% [12.4, 14.7] of clear pixels,
Cloud Score+ 58%; s2cloudless cannot reach 5% leak at any threshold
(floor 7.2%; it does not model shadow).

**Where it fails** (real data):

![Leak by land cover](results/figures/landcover_leak.png)
![By cloud regime](results/figures/regimes.png)
![Examples](results/figures/examples.png)

**Stress test (simulated, labeled as such):** change versus the clean run, paired over the
same scenes; n.s. = 95% CI includes zero.

| Degradation | Effect by severity |
|---|---|
| Sensor noise | σ 0.005: leak +2.7 pp, discard -0.7 pp · σ 0.01: leak +6.8 pp, discard -1.4 pp · σ 0.02: leak +13.3 pp, discard -2.3 pp |
| Optical blur | σ 0.5 px: leak -0.7 pp, discard +1.0 pp · σ 1 px: leak -3.0 pp, discard +10.9 pp · σ 2 px: leak -3.3 pp, discard +50.2 pp |
| Coarser resolution | 20 m: leak +0.4 pp, discard +1.2 pp · 30 m: leak +0.7 pp, discard +2.0 pp · 50 m: leak +2.2 pp, discard +3.7 pp |
| NIR misregistration | 1 px: leak +0.1 pp (n.s.), discard +0.1 pp · 2 px: leak +0.0 pp (n.s.), discard +0.4 pp · 4 px: leak +0.1 pp (n.s.), discard +0.9 pp |
| 8-bit JPEG delivery | q95: leak +3.2 pp, discard -0.1 pp (n.s.) · q75: leak +4.7 pp, discard +0.6 pp (n.s.) · q50: leak +6.4 pp, discard +1.7 pp |
| 8-bit lossless (clip level) | clip 1: leak +0.5 pp, discard -0.3 pp · clip 0.6: leak +2.4 pp, discard -0.1 pp (n.s.) · clip 0.3: leak +13.4 pp, discard -1.1 pp |
| Coarse data on a 10 m grid | 20 m: leak -2.7 pp, discard +9.3 pp · 30 m: leak -3.3 pp, discard +19.9 pp · 50 m: leak -3.5 pp, discard +59.7 pp |

Coarser data, leak / discard against the 10 m expert label (clean 10 m run:
6.1% / 4.6%). *Native grid*: OCM runs on the coarse pixels.
*On a 10 m grid*: the same coarse data resampled (bilinear) to 10 m first, as data cubes do.
*Oracle*: the expert label itself reduced to that grid, the best any mask at that resolution
could score.

| Grid | OCM, native grid | OCM, on a 10 m grid | Oracle |
|---|---|---|---|
| 20 m | 6.5% / 5.9% | 3.4% / 14.0% | 0.7% / 0.1% |
| 30 m | 6.8% / 6.7% | 2.8% / 24.6% | 0.6% / 0.6% |
| 50 m | 8.3% / 8.3% | 2.6% / 64.3% | 1.1% / 1.0% |

## Findings and recommendations

| Severity | Finding | Recommendation |
|---|---|---|
| High | Coarse imagery resampled to a finer grid is heavily over-masked | Run the mask on each sensor's native grid, then resample the mask, never the imagery first. Tested: the native run keeps the discard increase near 2 pp. |
| High | Scene filters built on the product's own cloud mask keep cloudy scenes | Gate scenes on OmniCloudMask-derived cover, not the SCL band. Tested: cloud reaching the pipeline inside kept scenes falls from 3.3% to 1.6% [0.7, 2.7] after masking. |
| Medium | At its default setting, about 6% of accepted pixels are cloud or shadow | For leak-sensitive uses (time series, change detection), accept a pixel only if P(clear) >= 0.85 (chosen on the validation split): test leak 2.0% [1.4, 2.6] at 13.6% discard. Do not buffer the mask instead: an 8-pixel buffer still leaks 3.6% at 13.1% discard. |
| Medium | Clipped 8-bit inputs let bright cloud through | Feed unclipped reflectance (12/16-bit), or 8-bit only with the range kept to at least 1.0 reflectance; do not run the mask on visual/stretched products. |
| Medium | Cloud over water leaks more often (10.4% vs 6.1% overall) | For coastal, inland-water and flood work, use the stricter threshold above and spot-check a local sample before relying on the mask. |
| Medium | Small amounts of cloud are missed in mostly clear scenes | Covered by the threshold recommendation; the tested buffer does not fix it as cheaply. |
| Medium | Usable ground is discarded in heavily clouded scenes | If gap-filling from cloudy scenes matters, validate a lower threshold for that regime (not tested here). |
| Low | Snow and ice: higher leak, but weak evidence | Validate locally before using the mask in snow-covered regions. |
| Low | Noisier sensors leak more | Re-validate before using the mask on lower-SNR small-satellite imagery. |
| Info | Verified non-issues | No action. |

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
Built 2026-10-26 by Shane Cairns. Generated by `scripts/build_docs.py`.
