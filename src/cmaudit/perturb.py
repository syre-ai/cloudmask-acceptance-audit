"""Standard, simulated input degradations for the A4 stress test.

Each function takes OCM's input (3, H, W) uint16 reflectance x 10000 (R, G, NIR), with 0 as
nodata, and returns the same shape and dtype. Nodata stays 0 and valid pixels never become 0
(so a perturbation cannot silently change OCM's nodata mask). None of these moves a cloud, so
the expert label remains the reference. All are labeled SIMULATED in every output.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage

SCALE = 10000.0


def _finish(x: np.ndarray, nodata: np.ndarray) -> np.ndarray:
    out = np.clip(np.rint(x), 1, 65535).astype(np.uint16)
    out[nodata] = 0
    return out


def _nodata(img: np.ndarray) -> np.ndarray:
    return (img == 0).all(0)[None].repeat(img.shape[0], 0)


def noise(img: np.ndarray, sigma_reflectance: float, seed: int = 0) -> np.ndarray:
    """Additive Gaussian sensor noise, sigma in reflectance units (e.g. 0.01)."""
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32) + rng.normal(0, sigma_reflectance * SCALE, img.shape)
    return _finish(x, _nodata(img))


def blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Optical blur (Gaussian PSF), sigma in 10 m pixels."""
    x = np.stack([ndimage.gaussian_filter(b.astype(np.float32), sigma_px) for b in img])
    return _finish(x, _nodata(img))


REAL = 509  # CloudSEN12+ real patch size inside the 512 padded array
COARSE_CANVAS = 510  # 509 edge-padded by one pixel: divisible by 2, 3 and 5


def coarsen(img: np.ndarray, factor: int) -> np.ndarray:
    """Coarser ground sampling (2 -> 20 m, 3 -> 30 m Landsat-like, 5 -> 50 m): the real 509 px
    window is edge-padded to 510 and block-averaged by ``factor``. Returns the *small* image,
    which OCM runs on natively (it documents support for 10-50 m input); map the prediction
    back to 10 m with ``upsample_pred``. Blocks containing nodata are averaged over valid
    pixels only."""
    x = img[:, :REAL, :REAL].astype(np.float32)
    x = np.pad(x, ((0, 0), (0, COARSE_CANVAS - REAL), (0, COARSE_CANVAS - REAL)), mode="edge")
    valid = (x > 0).any(0).astype(np.float32)
    c, n = x.shape[0], COARSE_CANVAS // factor
    blocks = (x * valid).reshape(c, n, factor, n, factor).sum(axis=(2, 4))
    counts = valid.reshape(n, factor, n, factor).sum(axis=(1, 3))
    small = np.where(counts > 0, blocks / np.maximum(counts, 1), 0)
    out = np.clip(np.rint(small), 1, 65535).astype(np.uint16)
    out[:, counts == 0] = 0
    return out


def upsample_pred(pred_small: np.ndarray, factor: int, size: int = 512,
                  fill: int = 255) -> np.ndarray:
    """Nearest-neighbour a coarse prediction back onto the padded 10 m grid."""
    up = pred_small.repeat(factor, 0).repeat(factor, 1)[:REAL, :REAL]
    out = np.full((size, size), fill, dtype=pred_small.dtype)
    out[:REAL, :REAL] = up
    return out


def misregister(img: np.ndarray, shift_px: int, band: int = 2) -> np.ndarray:
    """Shift one band (default NIR) by ``shift_px`` pixels diagonally relative to the others,
    as in poor band-to-band registration."""
    x = img.astype(np.float32).copy()
    x[band] = ndimage.shift(x[band], (shift_px, shift_px), order=0, mode="nearest")
    return _finish(x, _nodata(img))


def jpeg(img: np.ndarray, quality: int, max_reflectance: float = 0.6) -> np.ndarray:
    """Delivery as an 8-bit JPEG: linear scale 0..max_reflectance -> 0..255, JPEG at
    ``quality``, then back to reflectance. Combines 8-bit quantization and compression."""
    x = np.clip(img.astype(np.float32) / (max_reflectance * SCALE) * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(x.transpose(1, 2, 0)).save(buf, format="JPEG", quality=quality)
    y = np.asarray(Image.open(io.BytesIO(buf.getvalue()))).transpose(2, 0, 1).astype(np.float32)
    return _finish(y / 255.0 * max_reflectance * SCALE, _nodata(img))


def gain_offset(img: np.ndarray, gains=(1.0, 1.0, 1.1), offsets=(0.0, 0.0, 0.01)) -> np.ndarray:
    """Per-band linear calibration drift (gain, offset in reflectance). OCM z-scores each band
    per image, so this is expected to be invariant up to clipping; used as a check."""
    x = img.astype(np.float32) * np.asarray(gains, np.float32)[:, None, None]
    x = x + np.asarray(offsets, np.float32)[:, None, None] * SCALE
    return _finish(x, _nodata(img))


# name -> (function, kwarg, severities, units label)
SUITE = {
    "Sensor noise": (noise, "sigma_reflectance", [0.005, 0.01, 0.02], "σ reflectance"),
    "Optical blur": (blur, "sigma_px", [0.5, 1.0, 2.0], "σ px (10 m)"),
    "Coarser resolution": (coarsen, "factor", [2, 3, 5], "× 10 m"),
    "NIR misregistration": (misregister, "shift_px", [1, 2, 4], "px (10 m)"),
    "8-bit JPEG delivery": (jpeg, "quality", [95, 75, 50], "JPEG quality"),
}
