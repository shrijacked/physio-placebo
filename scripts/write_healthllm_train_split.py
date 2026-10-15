"""Write the PMData stress train complement. Refuses to write if eval text remains.

    python scripts/write_healthllm_train_split.py
"""

from __future__ import annotations

import json

from physio_placebo.comparison import build_locked_train_items, train_split_payload
from physio_placebo.paths import REPO_ROOT
from physio_placebo.provenance import config_hash, run_metadata, utc_stamp


def main() -> None:
    items = build_locked_train_items()
    payload = train_split_payload(items)
    cfg = {"what": "pmdata_stress_train_complement", **payload}
    dest = (
        REPO_ROOT
        / "results"
        / "comparison"
        / "pmdata_stress_train"
        / f"{utc_stamp()}_{config_hash(cfg)[:8]}"
    )
    dest.mkdir(parents=True, exist_ok=False)
    train_path = dest / "train.json"
    body = json.dumps(items, indent=2) + "\n"
    train_path.write_text(body)
    manifest = {
        **payload,
        "train_file": str(train_path.relative_to(REPO_ROOT)),
        **run_metadata(cfg),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"n_train={payload['n_train']} dropped={payload['n_dropped']} dir={dest}")


if __name__ == "__main__":
    main()
