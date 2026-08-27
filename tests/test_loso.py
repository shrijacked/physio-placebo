"""Automated leakage tests for the strict LOSO harness (Week-1 exit criterion)."""

from __future__ import annotations

import pytest

from physio_placebo.data.loso import loso_folds, prompt_dev_split

SUBJECTS = ["S7", "S2", "S10", "S3", "S13", "S5"]


def test_every_subject_held_out_exactly_once():
    folds = loso_folds(SUBJECTS)
    held_out = [t for t, _ in folds]
    assert sorted(held_out) == sorted(set(SUBJECTS))


def test_no_leakage_test_subject_never_in_train():
    for test_subject, train_subjects in loso_folds(SUBJECTS):
        assert test_subject not in train_subjects
        assert len(train_subjects) == len(set(SUBJECTS)) - 1


def test_deterministic_regardless_of_input_order():
    assert loso_folds(SUBJECTS) == loso_folds(sorted(SUBJECTS)) == loso_folds(SUBJECTS[::-1])


def test_duplicates_collapsed():
    assert loso_folds(["S1", "S1", "S2"]) == loso_folds(["S1", "S2"])


def test_too_few_subjects_rejected():
    with pytest.raises(ValueError):
        loso_folds(["S1"])


def test_prompt_dev_split_disjoint_and_deterministic():
    dev, eval_ = prompt_dev_split(SUBJECTS, n_dev=2, seed=1337)
    assert set(dev).isdisjoint(eval_)
    assert sorted(dev + eval_) == sorted(set(SUBJECTS))
    assert (dev, eval_) == prompt_dev_split(SUBJECTS[::-1], n_dev=2, seed=1337)
    # dev subjects never appear anywhere in evaluation folds
    for test_subject, train_subjects in loso_folds(eval_):
        assert test_subject not in dev
        assert set(train_subjects).isdisjoint(dev)


def test_prompt_dev_split_seed_sensitivity():
    a, _ = prompt_dev_split(SUBJECTS, n_dev=2, seed=1)
    b, _ = prompt_dev_split(SUBJECTS, n_dev=2, seed=2)
    assert a != b  # different seeds pick different dev sets for this roster
