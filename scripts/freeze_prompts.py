"""Write configs/frozen_prompts.lock.yaml from real vLLM sweep JSONs.

    python scripts/freeze_prompts.py \\
        --sweep results/prompt_sweep/sweep_vllm_qwen3_8b.json \\
        --sweep results/prompt_sweep/sweep_vllm_llama31_8b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physio_placebo.prompts.freeze import write_lockfile, write_lockfile_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", dest="sweeps", required=True)
    ap.add_argument("--paradigm", choices=("AB", "C"), default="AB")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    paths = [Path(p) for p in args.sweeps]
    dest = Path(args.out) if args.out else None
    if args.paradigm == "C":
        wrote = write_lockfile_c(paths, dest=dest)
    else:
        wrote = write_lockfile(paths, dest=dest)
    print(json.dumps({"wrote": str(wrote), "frozen": True, "paradigm": args.paradigm}, indent=2))


if __name__ == "__main__":
    main()
