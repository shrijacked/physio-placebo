"""Prompt templates render 0-shot and 4-shot without leaking eval answers."""

from __future__ import annotations

import pytest

from physio_placebo.prompts.templates import TEMPLATE_IDS, render_prompt


def test_all_three_template_ids_end_with_answer_colon():
    for tid in TEMPLATE_IDS:
        text = render_prompt(tid, "wesad", body="hrv_sdnn is 40.")
        assert text.rstrip().endswith("Answer:")
        assert "A = non-stress" in text
        assert "B = stress" in text
        assert "hrv_sdnn is 40." in text


def test_four_shot_includes_exemplar_tokens():
    exemplars = [(f"body {i}", "A" if i % 2 == 0 else "B") for i in range(4)]
    text = render_prompt("instruction", "maus", body="query", exemplars=exemplars)
    assert text.count("Answer: A") == 2
    assert text.count("Answer: B") == 2
    assert text.rstrip().endswith("Answer:")
    assert "A = low-workload" in text


def test_unknown_template_rejected():
    with pytest.raises(ValueError, match="unknown"):
        render_prompt("poetic", "wesad", body="x")
