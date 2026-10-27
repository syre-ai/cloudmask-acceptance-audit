import numpy as np
import pytest

from cmaudit.perturb import SUITE, coarsen, gain_offset, misregister, noise


def _img(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(200, 5000, (3, 64, 64)).astype(np.uint16)
    img[:, -3:, :] = 0  # padding-like nodata strip
    return img


@pytest.mark.parametrize("name", [n for n in SUITE if n != "Coarser resolution"])
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
