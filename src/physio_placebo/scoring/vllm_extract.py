"""Pure helpers for reading label logprobs out of a vLLM generate step.

Imported by the GPU client and by unit tests. Does not import vLLM.
"""

from __future__ import annotations

from collections.abc import Mapping

from physio_placebo.scoring.logprob import ScoringFailure


def _as_logprob(item: object) -> float:
    if hasattr(item, "logprob"):
        return float(item.logprob)
    return float(item)


def extract_label_logprobs(
    step: Mapping[int, object],
    token_ids: Mapping[str, int],
) -> dict[str, float]:
    """Read label-token logprobs from one generated-token step (full-vocab table)."""
    missing = [name for name, tid in token_ids.items() if int(tid) not in step]
    if missing:
        raise ScoringFailure(f"missing label logprobs for {missing}")
    return {name: _as_logprob(step[int(tid)]) for name, tid in token_ids.items()}


def single_token_id(tokenizer, text: str) -> int:
    """Encode a label string and require exactly one token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not a single token: {ids}")
    return int(ids[0])
