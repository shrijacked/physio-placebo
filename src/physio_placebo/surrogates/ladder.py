"""The pre-registered surrogate ladder (OSF, Weeks 7–8).

Null rungs replace the window's signal. The label-preserving subject permutation
is computed by the same code and is never a null: it keeps the condition and
changes the person.

Moments for Gaussian noise and the flatline are pooled over every window and
every sample of a channel. They are never computed inside one class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

NULL_RUNGS = (
    "gaussian",
    "iaaft",
    "subject_perm_broken",
    "flatline",
    "deleted",
)
DESCRIPTIVE = ("subject_perm_preserved",)
ALL_SURROGATES = NULL_RUNGS + DESCRIPTIVE
_IAAFT_ITERS = 20


def pooled_moments(x: np.ndarray) -> tuple[float, float]:
    """Global mean and population variance. ``x`` is (n_windows, n_samples)."""
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        raise ValueError("cannot pool moments of an empty channel")
    return float(np.mean(values)), float(np.var(values))


def deleted_text(channel: str) -> str:
    """Scaffolding with no signal digits."""
    return f"{channel}: signal deleted"


def iaaft_batch(x: np.ndarray, rng: np.random.Generator, n_iter: int = _IAAFT_ITERS) -> np.ndarray:
    """IAAFT on each row of ``x`` (n_windows, n_samples)."""
    series = np.asarray(x, dtype=float)
    if series.ndim != 2:
        raise ValueError("iaaft_batch expects (n_windows, n_samples)")
    n_win, n = series.shape
    if n < 4:
        raise ValueError("IAAFT needs at least 4 samples")
    sorted_x = np.sort(series, axis=1)
    spectrum = np.abs(np.fft.rfft(series, axis=1))
    current = np.empty_like(series)
    for i in range(n_win):
        current[i] = rng.permutation(series[i])
    for _ in range(n_iter):
        phase = np.angle(np.fft.rfft(current, axis=1))
        adjusted = np.fft.irfft(spectrum * np.exp(1j * phase), n=n, axis=1)
        order = np.argsort(np.argsort(adjusted, axis=1), axis=1)
        current = np.take_along_axis(sorted_x, order, axis=1)
    phase = np.angle(np.fft.rfft(current, axis=1))
    return np.fft.irfft(spectrum * np.exp(1j * phase), n=n, axis=1)


def permutation_donors(
    subjects: Sequence[str],
    y: np.ndarray,
    *,
    preserve_label: bool,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _donors(np.asarray(list(subjects)), np.asarray(y), preserve_label=preserve_label, rng=rng)


def iaaft(x: np.ndarray, rng: np.random.Generator, n_iter: int = _IAAFT_ITERS) -> np.ndarray:
    """One window. Amplitude spectrum matches ``x``; the series itself does not."""
    series = np.asarray(x, dtype=float)
    n = series.size
    if n < 4:
        raise ValueError("IAAFT needs at least 4 samples")
    sorted_x = np.sort(series)
    spectrum = np.abs(np.fft.rfft(series))
    current = rng.permutation(series)
    for _ in range(n_iter):
        phase = np.angle(np.fft.rfft(current))
        adjusted = np.fft.irfft(spectrum * np.exp(1j * phase), n=n)
        order = np.argsort(np.argsort(adjusted))
        current = sorted_x[order]
    phase = np.angle(np.fft.rfft(current))
    return np.fft.irfft(spectrum * np.exp(1j * phase), n=n)


def _donors(
    subjects: np.ndarray,
    y: np.ndarray,
    *,
    preserve_label: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    subjects = np.asarray(subjects)
    y = np.asarray(y)
    if subjects.shape[0] != y.shape[0]:
        raise ValueError("subjects and labels must have the same length")
    donors = np.empty(subjects.shape[0], dtype=int)
    for i, subject in enumerate(subjects):
        mask = subjects != subject
        if preserve_label:
            mask = mask & (y == y[i])
        choices = np.flatnonzero(mask)
        if choices.size == 0:
            raise ValueError(f"no donor window for index {i}")
        donors[i] = int(rng.choice(choices))
    return donors


def apply_surrogates(
    channels: Mapping[str, np.ndarray],
    *,
    y: np.ndarray,
    subjects: Sequence[str],
    kind: str,
    seed: int,
) -> dict[str, np.ndarray] | None:
    """Return surrogate channels, or None when the signal is deleted.

    Every channel of one window moves together under a subject permutation.
    """
    if kind not in ALL_SURROGATES:
        raise ValueError(f"unknown surrogate {kind!r}")
    names = list(channels)
    if not names:
        raise ValueError("no channels")
    arrays = {name: np.asarray(channels[name], dtype=float) for name in names}
    n = arrays[names[0]].shape[0]
    y_arr = np.asarray(y)
    subject_arr = np.asarray(list(subjects))
    if y_arr.shape[0] != n or subject_arr.shape[0] != n:
        raise ValueError("y and subjects must match the window count")
    for name in names[1:]:
        if arrays[name].shape[0] != n:
            raise ValueError(f"{name} has a different window count")
    if kind == "deleted":
        return None

    rng = np.random.default_rng(seed)
    if kind in ("subject_perm_broken", "subject_perm_preserved"):
        donors = _donors(
            subject_arr,
            y_arr,
            preserve_label=kind == "subject_perm_preserved",
            rng=rng,
        )
        return {name: arrays[name][donors] for name in names}

    out: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        mean, variance = pooled_moments(values)
        if kind == "flatline":
            out[name] = np.full_like(values, mean)
        elif kind == "gaussian":
            out[name] = rng.normal(mean, np.sqrt(variance), size=values.shape)
        elif kind == "iaaft":
            out[name] = iaaft_batch(values, rng)
        else:
            raise ValueError(kind)
    return out
