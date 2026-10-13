"""Defining properties of the surrogate ladder. Synthetic windows only."""

from __future__ import annotations

import numpy as np
import pytest

from physio_placebo.surrogates.ladder import (
    DESCRIPTIVE,
    NULL_RUNGS,
    apply_surrogates,
    deleted_text,
    iaaft,
    pooled_moments,
)


def _two_class_batch() -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Class 0 sits near 0. Class 1 sits near 10. Two subjects each."""
    rng = np.random.default_rng(0)
    low = rng.normal(0.0, 0.2, size=(4, 64))
    high = rng.normal(10.0, 0.2, size=(4, 64))
    x = np.vstack([low, high])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    subjects = ["a", "a", "b", "b", "a", "a", "b", "b"]
    return {"ecg": x, "eda": x + 1.0}, y, subjects


def test_null_rungs_do_not_include_the_label_preserving_permutation():
    assert "subject_perm_preserved" not in NULL_RUNGS
    assert DESCRIPTIVE == ("subject_perm_preserved",)


def test_gaussian_uses_the_global_pool_not_the_class():
    channels, y, subjects = _two_class_batch()
    out = apply_surrogates(channels, y=y, subjects=subjects, kind="gaussian", seed=1337)
    again = apply_surrogates(channels, y=y, subjects=subjects, kind="gaussian", seed=1337)
    assert out is not None and again is not None
    assert np.allclose(out["ecg"], again["ecg"])
    mean, variance = pooled_moments(channels["ecg"])
    class0 = float(np.mean(out["ecg"][y == 0]))
    # Class 0 of the recording sits near 0. The noise for those windows sits
    # nearer the pooled mean than that class mean.
    assert abs(class0 - mean) < abs(class0 - 0.0)
    assert float(np.var(out["ecg"])) == pytest.approx(variance, rel=0.25)
    assert abs(float(np.mean(channels["ecg"][y == 1])) - 10) < 1


def test_flatline_is_the_pooled_mean():
    channels, y, subjects = _two_class_batch()
    out = apply_surrogates(channels, y=y, subjects=subjects, kind="flatline", seed=1)
    assert out is not None
    mean, _ = pooled_moments(channels["ecg"])
    assert np.allclose(out["ecg"], mean)
    assert not np.allclose(out["eda"], mean)


def test_iaaft_keeps_the_amplitude_spectrum_and_changes_the_series():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 4 * np.pi, 128, endpoint=False)
    x = np.sin(t) + 0.1 * rng.normal(size=t.size)
    y = iaaft(x, np.random.default_rng(3))
    spec_x = np.abs(np.fft.rfft(x))
    spec_y = np.abs(np.fft.rfft(y))
    assert np.allclose(spec_x, spec_y, atol=1e-6)
    assert not np.allclose(x, y)


def test_broken_permutation_changes_subject_and_does_not_keep_the_label():
    channels, y, subjects = _two_class_batch()
    out = apply_surrogates(
        channels, y=y, subjects=subjects, kind="subject_perm_broken", seed=4
    )
    assert out is not None
    # A donor row is an exact copy of some other subject's row.
    original = channels["ecg"]
    same_subject = 0
    label_matches = 0
    for i in range(original.shape[0]):
        matches = np.where(np.all(np.isclose(original, out["ecg"][i]), axis=1))[0]
        assert matches.size == 1
        donor = int(matches[0])
        assert subjects[donor] != subjects[i]
        same_subject += subjects[donor] == subjects[i]
        label_matches += int(y[donor] == y[i])
        # Both channels of the window come from the same donor.
        assert np.allclose(out["eda"][i], channels["eda"][donor])
    assert same_subject == 0
    assert label_matches < len(y)


def test_preserved_permutation_keeps_the_label_and_changes_the_subject():
    channels, y, subjects = _two_class_batch()
    out = apply_surrogates(
        channels, y=y, subjects=subjects, kind="subject_perm_preserved", seed=5
    )
    assert out is not None
    original = channels["ecg"]
    for i in range(original.shape[0]):
        matches = np.where(np.all(np.isclose(original, out["ecg"][i]), axis=1))[0]
        donor = int(matches[0])
        assert subjects[donor] != subjects[i]
        assert y[donor] == y[i]


def test_apply_iaaft_preserves_each_row_spectrum():
    channels, y, subjects = _two_class_batch()
    out = apply_surrogates(channels, y=y, subjects=subjects, kind="iaaft", seed=6)
    assert out is not None
    for i in range(channels["ecg"].shape[0]):
        spec_x = np.abs(np.fft.rfft(channels["ecg"][i]))
        spec_y = np.abs(np.fft.rfft(out["ecg"][i]))
        assert np.allclose(spec_x, spec_y, atol=1e-5)
        assert not np.allclose(channels["ecg"][i], out["ecg"][i])


def test_deleted_text_has_no_digits():
    text = deleted_text("ecg")
    assert "ecg" in text
    assert not any(ch.isdigit() for ch in text)
    channels, y, subjects = _two_class_batch()
    assert apply_surrogates(channels, y=y, subjects=subjects, kind="deleted", seed=0) is None
