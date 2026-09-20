"""Write configs/frozen_prompts.lock.yaml from real vLLM sweep JSONs.

    python scripts/freeze_prompts.py \\
        --sweep results/prompt_sweep/sweep_vllm_qwen3_8b.json \\
        --sweep results/prompt_sweep/sweep_vllm_llama31_8b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physio_placebo.prompts.freeze import write_lockfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", dest="sweeps", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dest = write_lockfile([Path(p) for p in args.sweeps], dest=Path(args.out) if args.out else None)
    print(json.dumps({"wrote": str(dest), "frozen": True}, indent=2))


if __name__ == "__main__":
    main()
