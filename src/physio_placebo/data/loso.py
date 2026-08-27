"""Strict leave-one-subject-out (LOSO) split harness.

Guarantees enforced (and tested in tests/test_loso.py):
- every subject appears exactly once as the held-out test subject;
- the test subject never appears in its fold's training set;
- prompt-development subjects (used in Weeks 4-5 for template selection and
  few-shot exemplars) are disjoint from all evaluation folds.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Fold = tuple[str, tuple[str, ...]]  # (test_subject, train_subjects)


def loso_folds(subjects: Sequence[str]) -> list[Fold]:
    subs = sorted(set(subjects))
    if len(subs) < 2:
        raise ValueError(f"LOSO needs >= 2 subjects, got {subs}")
    return [(s, tuple(x for x in subs if x != s)) for s in subs]


def prompt_dev_split(
    subjects: Sequence[str],
    n_dev: int = 2,
    seed: int = 1337,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministically reserve `n_dev` subjects for prompt selection.

    Dev subjects are excluded from evaluation entirely; evaluation LOSO folds are
    built from the remaining subjects only.
    """
    subs = sorted(set(subjects))
    if not 0 < n_dev < len(subs):
        raise ValueError(f"n_dev must be in (0, {len(subs)}), got {n_dev}")
    rng = np.random.default_rng(seed)
    dev = tuple(sorted(rng.choice(np.asarray(subs), size=n_dev, replace=False).tolist()))
    eval_subjects = tuple(s for s in subs if s not in dev)
    return dev, eval_subjects
