"""Provenance helpers: content hashes, config hashes, library versions, git SHA.

Rule 6 of the working agreement: every results file must carry enough metadata to be
regenerated exactly — library versions, git SHA, config hash. These helpers are used
by every script that writes an artifact.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path | str, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of a config dict (canonical JSON). First 8 hex chars are used in
    directory names; the full hash goes into manifests."""
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def library_versions() -> dict[str, str]:
    import cvxopt
    import neurokit2
    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "neurokit2": neurokit2.__version__,
        "cvxopt": cvxopt.__version__,
    }


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Standard provenance block embedded in every artifact manifest."""
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "config": config,
        "config_sha256": config_hash(config),
        "library_versions": library_versions(),
    }
