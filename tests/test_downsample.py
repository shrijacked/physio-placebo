"""Paradigm A downsample schemes keep the documented lengths."""

from __future__ import annotations

import numpy as np
import pytest

from physio_placebo.paradigms.downsample import apply_scheme, bin_mean, uniform_stride


def test_uniform_stride_keeps_n_and_endpoints():
    x = np.arange(1000, dtype=float)
    out = uniform_stride(x, n_keep=256)
    assert out.shape == (256,)
    assert out[0] == 0
    assert out[-1] == 999


def test_uniform_stride_short_series_is_copied():
    x = np.arange(10, dtype=float)
    out = uniform_stride(x, n_keep=256)
    assert np.array_equal(out, x)
    assert out is not x


def test_bin_mean_block_average():
    x = np.arange(8, dtype=float)
    out = bin_mean(x, n_bins=4)
    assert out.shape == (4,)
    assert np.allclose(out, [0.5, 2.5, 4.5, 6.5])


def test_apply_scheme_reads_locked_config():
    x = np.arange(1000, dtype=float)
    assert apply_scheme(x, "uniform_stride").shape == (256,)
    assert apply_scheme(x, "bin_mean").shape == (128,)


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError, match="unknown"):
        apply_scheme(np.ones(8), "wavelet")
