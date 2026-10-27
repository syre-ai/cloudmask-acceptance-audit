"""Run OmniCloudMask v4 on CloudSEN12+ samples.

OCM takes Red, Green, NIR (B04, B03, B8A) as uint16 reflectance x 10000 and z-scores each
band per image (nodata = 0 excluded) before inference, so it is invariant by construction to
any per-band linear gain/offset applied uniformly across an image. Its two v4 U-Nets are
ensembled by averaging softmax outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

OCM_VERSION = 4.0
MODEL_DIR = Path("models/ocm")
# Indexes into the stored 4-band array (B02, B03, B04, B8A).
OCM_BANDS = [2, 1, 3]  # -> B04 red, B03 green, B8A NIR


def load_models(device: str = "cuda", dtype: torch.dtype = torch.float16) -> list:
    from omnicloudmask.cloud_mask import collect_models

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return collect_models(
        custom_models=None,
        inference_device=torch.device(device),
        inference_dtype=dtype,
        source="hugging_face",
        destination_model_dir=MODEL_DIR,
        model_version=OCM_VERSION,
    )


def predict_probs(
    models: list,
    images: list[np.ndarray],
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    batch_size: int = 8,
) -> np.ndarray:
    """Softmax probabilities (N, 4, H, W) float32 for a list of (3, H, W) uint16 arrays."""
    from omnicloudmask import predict_from_batch

    out = predict_from_batch(
        images,
        batch_size=batch_size,
        inference_device=device,
        inference_dtype=dtype,
        export_confidence=True,
        softmax_output=True,
        no_data_value=0,
        custom_models=models,
    )
    return np.asarray(out, dtype=np.float32)


def ocm_input(bands4: np.ndarray) -> np.ndarray:
    """Select OCM's R, G, NIR from the stored (4, H, W) B02/B03/B04/B8A array."""
    return np.ascontiguousarray(bands4[OCM_BANDS])


def quantize(probs: np.ndarray) -> np.ndarray:
    """Store probabilities as uint8 (resolution 1/255 is ample for threshold sweeps)."""
    return np.clip(np.rint(probs * 255), 0, 255).astype(np.uint8)


def dequantize(q: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) / 255.0
