"""Repository path helpers. All scripts resolve locations through here."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_raw(dataset: str | None = None) -> Path:
    p = REPO_ROOT / "data" / "raw"
    return p / dataset if dataset else p


def data_processed(dataset: str | None = None) -> Path:
    p = REPO_ROOT / "data" / "processed"
    return p / dataset if dataset else p


def results_dir(*parts: str) -> Path:
    return REPO_ROOT / "results" / Path(*parts) if parts else REPO_ROOT / "results"


def configs_dir() -> Path:
    return REPO_ROOT / "configs"
