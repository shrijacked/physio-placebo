"""vLLM step parsing without importing vLLM."""

from __future__ import annotations

import pytest

from physio_placebo.scoring.logprob import ScoringFailure
from physio_placebo.scoring.vllm_extract import extract_label_logprobs


class _LP:
    def __init__(self, logprob: float):
        self.logprob = logprob


def test_extracts_named_label_logprobs_from_token_ids():
    step = {10: _LP(-0.2), 11: _LP(-1.8), 99: _LP(-0.01)}
    table = extract_label_logprobs(step, {"A": 10, "B": 11})
    assert table == {"A": -0.2, "B": -1.8}


def test_plain_float_entries_work():
    table = extract_label_logprobs({1: -0.5, 2: -3.0}, {"A": 1, "B": 2})
    assert table["A"] == -0.5


def test_missing_label_id_is_scoring_failure():
    with pytest.raises(ScoringFailure, match="missing"):
        extract_label_logprobs({10: _LP(-0.1)}, {"A": 10, "B": 11})


def test_prepare_vllm_runtime_disables_flashinfer_sampler(monkeypatch):
    import os

    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    from physio_placebo.scoring.vllm_client import prepare_vllm_runtime

    prepare_vllm_runtime()
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_llm_engine_kwargs_omit_null_quantization():
    from physio_placebo.scoring.vllm_client import llm_engine_kwargs, load_model_spec

    spec = {
        "quantization": None,
        "dtype": "float16",
        "logprobs_topk": 32,
        "max_model_len": 8192,
    }
    kw = llm_engine_kwargs(spec, {"hf_id": "Qwen/Qwen3-8B"})
    assert "quantization" not in kw
    assert kw["model"] == "Qwen/Qwen3-8B"
    assert kw["dtype"] == "float16"
    assert kw["max_model_len"] == 8192

    locked = load_model_spec()
    assert locked.get("quantization") not in {"bitsandbytes", "bnb"}


def test_build_sampling_params_logprobs_count_matches_label_ids():
    from physio_placebo.scoring.vllm_client import build_sampling_params

    class VLLMLike:
        def __init__(self, **kwargs):
            ids = kwargs.get("logprob_token_ids")
            lp = kwargs.get("logprobs")
            if ids is not None and lp is not None and lp != len(ids):
                raise ValueError(
                    "When both logprobs and logprob_token_ids are set, "
                    "logprobs must equal len(logprob_token_ids)"
                )
            self.kwargs = kwargs

    spec = {"temperature": 0.0, "max_tokens": 1, "logprobs_topk": 32}
    p = build_sampling_params(VLLMLike, [10, 11], spec)
    assert p.kwargs.get("logprob_token_ids") == [10, 11]
    assert p.kwargs.get("logprobs") == 2
    assert p.kwargs.get("allowed_token_ids") is None
