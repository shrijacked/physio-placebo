"""Week-3 fidelity-anchor scoring and zero-shot prompt construction."""

from __future__ import annotations

import json

import pytest

from physio_placebo.anchor import (
    DEFAULT_EVAL_JSON,
    PAPER_MEDALPACA,
    SEEDS,
    SPLIT_MANIFEST,
    ZERO_SHOT_INSTRUCTION,
    build_zero_shot_prompts,
    constant3_mae,
    label_distribution,
    load_eval_items,
    locked_eval_sha256,
    parse_first_number,
    parse_label,
    remaining_prompts,
    score_records,
    zero_shot_question,
)
from physio_placebo.provenance import sha256_bytes, sha256_file


def test_parse_first_number_takes_the_first_digit_run():
    assert parse_first_number("Answer: 3") == 3.0
    assert parse_first_number("The predicted stress is 4.") == 4.0
    assert parse_first_number("I think 2, or maybe 5") == 2.0
    assert parse_first_number("N/A") is None
    assert parse_first_number("") is None


def test_parse_label_matches_their_output_sentence():
    assert parse_label("The predicted stress level is 2.") == 2
    with pytest.raises(ValueError, match="unparsable"):
        parse_label("no number here")


def test_zero_shot_prompt_ignores_eval_instruction_field():
    item_input = "sensor blob; What would be the predicted stress?"
    prompt = zero_shot_question(item_input, 3)
    assert prompt.startswith(ZERO_SHOT_INSTRUCTION)
    assert item_input in prompt
    assert prompt.endswith("Answer: 3")
    assert "personalized healthcare agent" not in prompt


def test_format_rand_nums_are_seed_deterministic():
    items = [{"input": "q", "output": "The predicted stress level is 3."}] * 8
    a = [p["rand_num"] for p in build_zero_shot_prompts(items, seed=0)]
    b = [p["rand_num"] for p in build_zero_shot_prompts(items, seed=0)]
    c = [p["rand_num"] for p in build_zero_shot_prompts(items, seed=1)]
    assert a == b
    assert a != c
    assert set(a) <= set(range(6))


def test_score_records_drops_unparsable_and_reports_mae():
    records = [
        {"label": "The predicted stress level is 3.", "answer": "Answer: 3"},
        {"label": "The predicted stress level is 2.", "answer": "4"},
        {"label": "The predicted stress level is 1.", "answer": "N/A"},
    ]
    scored = score_records(records)
    assert scored["n_items"] == 3
    assert scored["n_parsed"] == 2
    assert scored["n_parse_failures"] == 1
    assert scored["mae"] == pytest.approx(1.0)


def test_score_records_rejects_all_failures():
    with pytest.raises(ValueError, match="no parsable"):
        score_records([{"label": "The predicted stress level is 3.", "answer": "none"}])


@pytest.mark.skipif(not DEFAULT_EVAL_JSON.is_file(), reason="Health-LLM eval split not on disk")
def test_locked_eval_split_hash_and_constant3():
    items = load_eval_items()
    assert len(items) == 299
    assert sha256_file(DEFAULT_EVAL_JSON) == locked_eval_sha256()
    assert constant3_mae(items) == pytest.approx(0.4013377926421405)
    dist = label_distribution(items)
    assert dist == {"1": 4, "2": 53, "3": 186, "4": 53, "5": 3}


@pytest.mark.skipif(not DEFAULT_EVAL_JSON.is_file(), reason="Health-LLM eval split not on disk")
def test_zero_shot_prompts_cover_all_eval_items_for_three_seeds():
    items = load_eval_items()
    expected = {
        0: "0abba6a2fd570735901fcff5d9f05be5a352d64c9aadfeb126b0be9cc2b872a1",
        1: "6789c80970c367980c7e6a2b5a112ca354d481cf003127d9d6ca7cc0332c5f44",
        2: "50bee7218e757530f932bfdab3442dd72801161dbc96aa441bd3a47536d969c9",
    }
    for seed in SEEDS:
        prompts = build_zero_shot_prompts(items, seed)
        assert len(prompts) == 299
        assert prompts[0]["no"] == 1
        assert prompts[-1]["no"] == 299
        assert all(p["question"].startswith(ZERO_SHOT_INSTRUCTION) for p in prompts)
        payload = json.dumps(prompts, indent=2) + "\n"
        assert sha256_bytes(payload.encode()) == expected[seed]


def test_remaining_prompts_resumes_from_contiguous_prefix():
    prompts = [{"no": i, "question": f"q{i}"} for i in range(1, 6)]
    assert remaining_prompts(prompts, []) == prompts
    assert [p["no"] for p in remaining_prompts(prompts, prompts[:2])] == [3, 4, 5]
    assert remaining_prompts(prompts, prompts) == []
    with pytest.raises(ValueError, match="contiguous"):
        remaining_prompts(prompts, [{"no": 2}])
    with pytest.raises(ValueError, match="partial has"):
        remaining_prompts(prompts[:2], prompts)


def test_paper_anchor_target_is_the_medalpaca_cell():
    assert PAPER_MEDALPACA["mae_mean"] == 0.76
    manifest = json.loads(SPLIT_MANIFEST.read_text())
    assert manifest["eval_items"] == 299
    assert manifest["constant3_mae_on_eval"] == 0.401
