---
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
960 expert-labeled Sentinel-2 scenes from 192 sites (CloudSEN12+):

- metrics phrased as the decision: **leak** (cloud in the pixels you accept) and **discard**
  (clear ground you throw away), plus a scene-filter test;
- a screen of the model's own published training records against the test scenes, removing
  anything that might overlap;
- side-by-side comparison with the masks that ship with Sentinel-2 products;
- failure analysis on real scenes, a simulated stress test, and settings chosen on one data
  split and checked on another.

## Results

- At its default setting, **6.1%** of accepted pixels are cloud or shadow and
  **4.6%** of clear pixels are discarded. A threshold chosen on held-out data
  brings leak to **2.0%** at 13.6% discard; buffering the mask,
  a common alternative, did not reach that leak at any tested size.
- The cloud classification shipped in Sentinel-2 L2A products is much less reliable for scene
  filtering: of the patches it keeps under a "10% cloud" rule, **43.4%** are
  cloudier than that, against 5.9% with OmniCloudMask.
- Leak is higher over water (11.0%) and over snow and ice (on thinner evidence), and
  in mostly clear scenes it misses 16.6% of the cloud and shadow present.
- In a simulation, 30 m data placed on a 10 m grid before masking lost 24.6% of clear
  ground, against 6.6% when masked on its own grid.

![Leak vs discard](/images/cloudmask/tradeoff.png)

## What this means for a client

If you are about to automate on a model you did not build, the useful question is not "how
accurate is it?" but "what will it let through, what will it cost me, and which setting should
I run?" This audit answers that on public data and a single GPU, with confidence intervals, a
ranked list of findings, and settings that were chosen on one split and verified on another.
The same approach can be adapted to your model and your imagery.
