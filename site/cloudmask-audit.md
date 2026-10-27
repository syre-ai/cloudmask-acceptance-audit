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

An audit harness that answers the buyer's question for
[OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask), a popular open model, on
965 expert-labeled Sentinel-2 scenes from 193 sites (CloudSEN12+):

- metrics phrased as the decision: **leak** (cloud in the pixels you accept) and **discard**
  (clear ground you throw away), plus a scene filter test;
- a check of the model's own published training data against the test set, so the scores
  are not inflated by overlap;
- side-by-side comparison with the masks that ship with Sentinel-2 products;
- failure analysis on real scenes, a simulated stress test, and mitigations that were
  chosen on one split and verified on another.

## Results

- At its default setting, **6.1%** of accepted pixels are cloud or shadow and
  **4.6%** of clear pixels are discarded. One setting change, chosen on held-out
  data, brings leak to **2.0%** at 13.6% discard; a common
  alternative, buffering the mask, cannot get below 3.6%.
- The cloud classification shipped inside Sentinel-2 L2A products is far less safe for
  scene filtering: **43.4%** of scenes it lets through a "10% cloud" filter are
  cloudier than that, against 5.9% with OmniCloudMask.
- It is weakest over water (leak 10.4%) and in mostly clear scenes, where it
  misses 16.5% of the cloud and shadow present.

![Leak vs discard](/images/cloudmask/tradeoff.png)

## What this means for a client

If you are about to automate on a model you did not build, the useful question is not "how
accurate is it?" but "what will it let through, what will it cost me, and which setting should
I run?" This audit answers that in a week on public data and a single GPU, with confidence
intervals, a ranked list of findings and recommendations that were tested, not just
suggested. The same harness runs on your model and your imagery.
