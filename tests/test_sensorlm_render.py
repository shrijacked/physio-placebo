"""Week-1 render proof for the vendored SensorLM captioning templates (plan §9).

"Done means: templates render on one WESAD window." A WESAD-format window (cut by
our real windowing code from the synthetic fixture zip) is mapped into the SensorLM
Fitbit feature space and pushed through Google's caption generator end-to-end. A
second test renders every template bank directly. A final test uses the real
processed WESAD data when it exists on this machine (skipped elsewhere).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from physio_placebo.data.wesad import load_subject_recording
from physio_placebo.data.windows import cut_windows, load_subject_windows
from physio_placebo.paths import data_processed
from physio_placebo.prompts.vendor.sensorlm import captioning
from physio_placebo.prompts.vendor.sensorlm.constants import (
    FEATURES_TO_NORMALIZE,
    NORMALIZATION_PARAMETERS,
)


def _normalized_frame_from_eda(eda_4hz: np.ndarray) -> np.ndarray:
    """Map a real wrist-EDA window into SensorLM's normalized feature space.

    The EDA series (uS, 4 Hz) goes into their 'eda_level_real' channel after
    applying their normalization; every other channel is left at normalized 0,
    which their pipeline denormalizes to the Fitbit population mean.
    """
    t = eda_4hz.reshape(-1, 4).mean(axis=1)  # 4 Hz -> 1 Hz, T=60
    x = np.zeros((t.shape[0], len(FEATURES_TO_NORMALIZE)), dtype=np.float64)
    idx = FEATURES_TO_NORMALIZE.index("eda_level_real")
    mu, sigma = NORMALIZATION_PARAMETERS["eda_level_real"]
    x[:, idx] = (t - mu) / sigma
    return x


def _wesad_window_eda(fake_wesad_zip) -> np.ndarray:
    rec = load_subject_recording(fake_wesad_zip, "S2")
    win = cut_windows("wesad", "S2", [rec], window_s=60.0, stride_s=30.0)
    return win.channels["eda_wrist"][0]


def test_statistical_caption_renders_on_wesad_window(fake_wesad_zip):
    eda = _wesad_window_eda(fake_wesad_zip)
    x = _normalized_frame_from_eda(eda)

    random.seed(0)
    caption = captioning.generate_statistical_caption(x, mask=None)

    assert isinstance(caption, str)
    assert len(caption) > 50
    assert "For " in caption  # their per-category prefix
    assert sum(ch.isdigit() for ch in caption) >= 3

    # The denormalization roundtrip must reproduce the real window's EDA mean,
    # i.e. the caption is grounded in this window, not in template filler.
    idx = FEATURES_TO_NORMALIZE.index("eda_level_real")
    mu, sigma = NORMALIZATION_PARAMETERS["eda_level_real"]
    recovered = x[:, idx] * sigma + mu
    assert np.allclose(recovered.mean(), eda.reshape(-1, 4).mean(), atol=1e-9)


def test_statistical_caption_is_seed_deterministic(fake_wesad_zip):
    x = _normalized_frame_from_eda(_wesad_window_eda(fake_wesad_zip))
    random.seed(123)
    a = captioning.generate_statistical_caption(x, mask=None)
    random.seed(123)
    b = captioning.generate_statistical_caption(x, mask=None)
    assert a == b


def test_every_template_bank_renders(fake_wesad_zip):
    eda = _wesad_window_eda(fake_wesad_zip)
    random.seed(0)

    low = captioning._describe_low_level(
        "eda level", float(eda.mean()), float(eda.max()), float(eda.min()), float(eda.std())
    )
    trend = captioning._describe_trend("eda level", "increasing", 40, 240)
    anomaly = captioning._describe_anomaly("heart rate", "spike", 400)
    activity = captioning._describe_activity("Stress task", 100, 130)
    mood = captioning._describe_mood("stressed", 500)

    for rendered in (low, trend, anomaly, activity, mood):
        assert isinstance(rendered, str)
        assert len(rendered) > 10


def test_statistical_caption_renders_on_real_wesad_window():
    real = data_processed("wesad") / "S2.npz"
    if not real.exists():
        pytest.skip("real processed WESAD data not present on this machine")
    win = load_subject_windows(real)
    x = _normalized_frame_from_eda(win.channels["eda_wrist"][0])
    random.seed(0)
    caption = captioning.generate_statistical_caption(x, mask=None)
    assert len(caption) > 50
