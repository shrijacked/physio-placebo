"""Argmax over label-token logprobs.

The GPU client (vLLM) is responsible for returning a mapping token -> logprob
for the first generated position. This module never samples text and never
parses a free-form answer. Missing label tokens are scoring failures and are
not imputed (OSF scoring rule).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import yaml

from physio_placebo.paths import configs_dir


class ScoringFailure(Exception):
    """A label token was missing from the returned logprob table."""


def load_label_spec() -> dict:
    return yaml.safe_load((configs_dir() / "label_tokens.yaml").read_text())


def token_for_y(y: int, tokens: Sequence[str] | None = None) -> str:
    toks = list(tokens if tokens is not None else load_label_spec()["tokens"])
    if y not in (0, 1):
        raise ValueError(f"binary y must be 0 or 1, got {y!r}")
    return toks[y]


def y_from_token(token: str, tokens: Sequence[str] | None = None) -> int:
    toks = list(tokens if tokens is not None else load_label_spec()["tokens"])
    try:
        return toks.index(token)
    except ValueError as exc:
        raise ValueError(f"token {token!r} not in {toks}") from exc


def argmax_label(
    logprobs: Mapping[str, float],
    labels: Sequence[str] | None = None,
) -> str:
    """Return the label token with the largest logprob.

    Equal logprobs break toward the earlier entry in ``labels`` (the locked
    ``tie_break: first_in_tokens`` rule). Any label absent from ``logprobs``
    is a ``ScoringFailure``.
    """
    toks = list(labels if labels is not None else load_label_spec()["tokens"])
    if len(toks) < 2:
        raise ValueError(f"need at least two label tokens, got {toks}")
    missing = [t for t in toks if t not in logprobs]
    if missing:
        raise ScoringFailure(f"missing label logprobs for {missing}")
    # key = (logprob, reverse index) so ties keep the earlier token
    return max(toks, key=lambda t: (float(logprobs[t]), -toks.index(t)))
