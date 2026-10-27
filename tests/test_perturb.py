import numpy as np
import pytest

from cmaudit.perturb import SUITE, coarsen, gain_offset, misregister, noise


def _img(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(200, 5000, (3, 64, 64)).astype(np.uint16)
    img[:, -3:, :] = 0  # padding-like nodata strip
    return img


SMALL_OK = [n for n in SUITE if not n.startswith(("Coarser", "Coarse data"))]


@pytest.mark.parametrize("name", SMALL_OK)
def test_suite_preserves_shape_dtype_and_nodata(name):
    fn, kw, sevs, _ = SUITE[name]
    img = _img()
    for s in sevs:
        out = fn(img, **{kw: s})
        assert out.shape == img.shape and out.dtype == np.uint16
        nod = (img == 0).all(0)
        assert (out[:, nod] == 0).all()  # nodata stays nodata
        assert (out[:, ~nod] > 0).all()  # valid never becomes nodata


def test_noise_scale_and_seed():
    img = np.full((3, 128, 128), 3000, np.uint16)
    out = noise(img, 0.01).astype(np.float64)
    assert abs(out.std() - 100) < 5  # 0.01 reflectance = 100 DN
    assert np.array_equal(noise(img, 0.01, seed=1), noise(img, 0.01, seed=1))


def test_coarsen_block_means_and_upsample():
    from cmaudit.perturb import upsample_pred

    rng = np.random.default_rng(0)
    img = rng.integers(100, 4000, (3, 512, 512)).astype(np.uint16)
    for f, n in ((2, 255), (3, 170), (5, 102)):
        small = coarsen(img, f)
        assert small.shape == (3, n, n) and small.dtype == np.uint16
        assert small[0, 0, 0] == round(img[0, :f, :f].astype(float).mean())
        up = upsample_pred(np.arange(n * n, dtype=np.int64).reshape(n, n), f)
        assert up.shape == (512, 512) and up[0, 0] == up[f - 1, f - 1] == 0
        assert up[508, 508] == (508 // f) * n + 508 // f and up[510, 0] == 255


def test_misregister_shifts_only_nir():
    img = _img()
    out = misregister(img, 2)
    assert np.array_equal(out[:2], img[:2])
    assert np.array_equal(out[2, 10:40, 10:40], img[2, 8:38, 8:38])


def test_gain_offset_is_linear():
    img = np.full((3, 8, 8), 1000, np.uint16)
    out = gain_offset(img, gains=(1.0, 2.0, 1.0), offsets=(0.0, 0.0, 0.01))
    assert out[0, 0, 0] == 1000 and out[1, 0, 0] == 2000 and out[2, 0, 0] == 1100


def test_buffer_grows_cloud_not_abstention():
    import sys

    sys.path.insert(0, "scripts")
    from mitigations import buffered

    pred = np.zeros((9, 9), np.uint8)
    pred[4, 4] = 2
    pred[0, 0] = 255
    out = buffered(pred, 1)
    assert out[4, 4] == 2 and out[3, 4] == 1 and out[4, 5] == 1 and out[3, 3] == 0
    assert out[0, 1] == 0 and out[0, 0] == 255
    assert np.array_equal(buffered(pred, 0), pred)


def test_resampled_to_fine_smoothness():
    from cmaudit.perturb import resampled_to_fine

    rng = np.random.default_rng(0)
    img = rng.integers(100, 4000, (3, 512, 512)).astype(np.uint16)
    out = resampled_to_fine(img, 3)
    assert out.shape == img.shape and out.dtype == np.uint16
    assert (out[:, 509:, :] == img[:, 509:, :]).all()  # padding untouched
    # bilinear upsampling of block means is much smoother than the input
    assert np.abs(np.diff(out[0, :509, :509].astype(float), axis=1)).mean() < \
        0.5 * np.abs(np.diff(img[0, :509, :509].astype(float), axis=1)).mean()


def test_quantize8_clips_bright_values():
    from cmaudit.perturb import quantize8

    img = np.array([[[1000, 3000, 9000]]], np.uint16).repeat(3, 0)
    out = quantize8(img, 0.3)
    assert out[0, 0, 2] == 3000 and abs(int(out[0, 0, 0]) - 1000) <= 6


def test_per_band_missing_values_preserved():
    from cmaudit.perturb import blur

    img = _img()
    img[2, 10, 10] = 0  # one band missing at one pixel
    for name in SMALL_OK:
        fn, kw, sevs, _ = SUITE[name]
        out = fn(img, **{kw: sevs[-1]})
        assert out[2, 10, 10] == 0 and out[0, 10, 10] > 0, name
    assert blur(img, 1.0)[2, 10, 10] == 0


def test_blur_does_not_bleed_nodata():
    from cmaudit.perturb import blur

    img = np.full((3, 64, 64), 1000, np.uint16)
    img[:, -3:, :] = 0
    out = blur(img, 2.0)
    assert (out[:, :-3, :] == 1000).all()  # constant valid area stays constant at the edge


def test_coarse_on_fine_grid_controls():
    from cmaudit.perturb import resampled_to_fine, resampled_to_fine_nearest

    rng = np.random.default_rng(1)
    img = rng.integers(100, 4000, (3, 512, 512)).astype(np.uint16)
    img[:, 509:, :] = 0
    img[:, :, 509:] = 0
    for fn in (resampled_to_fine, resampled_to_fine_nearest):
        out = fn(img, 3)
        assert out.shape == img.shape and (out[:, 509:, :] == 0).all() and (out[:, :509, :509] > 0).all()
    near = resampled_to_fine_nearest(img, 3)
    assert near[0, 0, 0] == near[0, 2, 2]  # blocky


def test_jpeg_uses_same_rounding_as_lossless():
    from cmaudit.perturb import jpeg, quantize8

    img = np.full((3, 64, 64), 2345, np.uint16)  # flat image: JPEG at q100 is exact
    assert np.array_equal(jpeg(img, 100), quantize8(img, 0.6))
