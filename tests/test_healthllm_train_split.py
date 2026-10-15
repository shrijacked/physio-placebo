"""Train complement for the Health-LLM fine-tune cell. Eval text must not remain."""

from __future__ import annotations

import json

import pytest

from physio_placebo.anchor import (
    DEFAULT_EVAL_JSON,
    DEFAULT_TRAIN_POOL_JSON,
    assert_disjoint,
    medalpaca_train_text,
    record_body,
    train_complement,
)
from physio_placebo.comparison import (
    LOCK_PATH,
    build_locked_train_items,
    load_finetune_lock,
    train_split_payload,
)


def test_complement_drops_a_shared_sensor_string():
    pool = [
        {"input": "shared", "output": "a"},
        {"input": "only-train", "output": "b"},
    ]
    held = [{"question": "shared", "output": "a"}]
    kept = train_complement(pool, held)
    assert [record_body(item) for item in kept] == ["only-train"]
    assert_disjoint(kept, held)


def test_disjoint_rejects_a_leak():
    with pytest.raises(ValueError, match="leaked"):
        assert_disjoint(
            [{"input": "same"}],
            [{"input": "same"}],
        )


def test_medalpaca_template_wraps_instruction_input_and_output():
    text = medalpaca_train_text(
        {
            "instruction": "predict stress",
            "input": "steps 10",
            "output": "The predicted stress level is 3.",
        }
    )
    assert text.startswith("Below is an instruction")
    assert "### Instruction:\npredict stress" in text
    assert "### Input:\nsteps 10" in text
    assert text.endswith("### Response:\nThe predicted stress level is 3.")


def test_lock_is_the_7b_lora_cell():
    lock = load_finetune_lock()
    assert lock["model"] == "medalpaca/medalpaca-7b"
    assert lock["num_epochs"] == 3
    assert lock["learning_rate"] == 2.0e-5
    assert lock["global_batch_size"] == 128
    assert lock["eval_seed"] == 0
    assert lock["do_sample"] is False
    assert LOCK_PATH.is_file()


@pytest.mark.skipif(
    not (DEFAULT_EVAL_JSON.is_file() and DEFAULT_TRAIN_POOL_JSON.is_file()),
    reason="Health-LLM PMData stress files are not on disk",
)
def test_released_pool_leaks_one_row_and_the_complement_removes_it():
    pool = json.loads(DEFAULT_TRAIN_POOL_JSON.read_text())
    held = json.loads(DEFAULT_EVAL_JSON.read_text())
    pool_text = {record_body(item) for item in pool}
    held_text = {record_body(item) for item in held}
    assert len(pool_text & held_text) == 1
    kept = build_locked_train_items(pool, held)
    assert len(kept) == len(pool) - 1
    assert len(kept) == 653
    payload = train_split_payload(kept)
    assert payload["n_dropped"] == 1
    assert payload["n_train"] == 653
    assert payload["eval_sha256"].startswith("fea3b879")
