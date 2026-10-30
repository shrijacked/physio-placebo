"""Build the Week 10 SDS card from saved results.

    python scripts/build_sds_card.py
    python scripts/build_sds_card.py --real-rows-dir results/eval_grid_rows/<stamp>

Writes ``results/sds_card/<stamp>/card.json`` and ``card.md``, and refreshes
``reports/figures/fig_sds_heatmap.pdf`` and ``fig_auroc_vs_f1.pdf``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physio_placebo.paths import REPO_ROOT
from physio_placebo.provenance import run_metadata, utc_stamp
from physio_placebo.report.card import build_card, render_markdown, write_figures


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and value != value:
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-rows-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    real_rows = Path(args.real_rows_dir) if args.real_rows_dir else None
    if real_rows is not None and not real_rows.is_dir():
        raise SystemExit(f"no real-rows directory {real_rows}")
    card = build_card(REPO_ROOT, real_rows_dir=real_rows)
    dest = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "sds_card" / utc_stamp()
    dest.mkdir(parents=True, exist_ok=False)
    payload = {
        **run_metadata({"real_rows_dir": str(real_rows) if real_rows else None}),
        "card": _jsonable(card),
    }
    (dest / "card.json").write_text(json.dumps(payload, indent=2) + "\n")
    (dest / "card.md").write_text(render_markdown(card))
    write_figures(card, REPO_ROOT / "reports" / "figures")
    print(
        json.dumps(
            {
                "wrote": str(dest),
                "n_primary": card["n_primary"],
                "n_sds_undefined": card["n_sds_undefined"],
            }
        )
    )


if __name__ == "__main__":
    main()
