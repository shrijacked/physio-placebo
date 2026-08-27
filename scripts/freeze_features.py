"""Freeze the feature extraction code: record its hash + library versions.

Run once at the end of Week 1. After this, tests/test_feature_hash.py fails on any
edit to src/physio_placebo/features/frozen.py until the lock is deliberately
regenerated (which requires explicit approval per the working agreement).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from physio_placebo.features.frozen import frozen_code_hash
from physio_placebo.paths import configs_dir
from physio_placebo.provenance import git_sha


def main() -> None:
    lock_path = configs_dir() / "frozen_features.lock.json"
    payload = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_sha_at_freeze": git_sha(),
        **frozen_code_hash(),
    }
    lock_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {lock_path}")


if __name__ == "__main__":
    main()
