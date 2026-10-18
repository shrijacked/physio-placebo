"""Turn the Week 9 cells into the margin-loss and flip-rate panel.

Reads results/verbalizer_ablation/{model}/{dataset}/{i,ii,iii}/cell.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physio_placebo.paths import REPO_ROOT
from physio_placebo.prompts.ablation_metrics import (
    chance_mean,
    flip_rate_report,
    neutralization_report,
    paired_flips,
    paired_wilcoxon,
)

DATASETS = ("wesad", "maus", "cogwear")


def _load(root: Path, model: str, dataset: str, variant: str) -> dict:
    path = root / model / dataset / variant / "cell.json"
    return json.loads(path.read_text())["cell"]


def _floor(dataset: str) -> dict:
    import yaml

    pointer = yaml.safe_load((REPO_ROOT / "configs" / "locked_floors.yaml").read_text())
    return json.loads((REPO_ROOT / pointer[dataset]).read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root) if args.root else REPO_ROOT / "results" / "verbalizer_ablation"
    models = sorted(p.name for p in root.iterdir() if p.is_dir())
    panel = []
    for model in models:
        for dataset in DATASETS:
            if not (root / model / dataset / "i" / "cell.json").exists():
                continue
            original = _load(root, model, dataset, "i")
            neutralized = _load(root, model, dataset, "ii")
            flipped = _load(root, model, dataset, "iii")
            chance = chance_mean(_floor(dataset))
            loss = neutralization_report(
                original["macro_f1"],
                original["ci_lo"],
                neutralized["macro_f1"],
                chance,
            )
            subjects, flags = paired_flips(original["rows"], flipped["rows"])
            flips = flip_rate_report(subjects, flags)
            panel.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "macro_f1": {
                        "i": original["macro_f1"],
                        "ii": neutralized["macro_f1"],
                        "iii": flipped["macro_f1"],
                    },
                    "neutralization": loss,
                    "flip": {k: v for k, v in flips.items() if k != "per_subject"},
                    "flip_per_subject": flips["per_subject"],
                    "wilcoxon_i_vs_ii": paired_wilcoxon(
                        original["per_subject"], neutralized["per_subject"]
                    ),
                    "wilcoxon_i_vs_iii": paired_wilcoxon(
                        original["per_subject"], flipped["per_subject"]
                    ),
                }
            )
    out = Path(args.out) if args.out else root / "panel.json"
    out.write_text(json.dumps({"panel": panel}, indent=2))
    print(f"wrote {out} rows={len(panel)}")


if __name__ == "__main__":
    main()
