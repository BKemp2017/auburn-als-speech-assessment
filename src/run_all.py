"""Regenerate every result, figure and paper number from the raw VOC-ALS data.

Run from the repository root:
    python src/run_all.py                       # full pipeline
    python src/run_all.py --start nested_selection   # resume from a stage
    python src/run_all.py --extract-embeddings  # also recompute WavLM embeddings

WavLM embeddings are reused when results/generated/raw_audio_wavlm_embeddings.npz
exists, since extracting them needs a GPU to be quick. The full run takes
about an hour.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import EMBEDDINGS_PATH

STAGES = [
    "prepare_data",
    "audit_raw_audio",
    "baseline",
    "discovery",
    "nested_selection",
    "protocol_stability",
    "duplicate_sensitivity",
    "wavlm_embeddings",
    "wavlm_selection",
    "recording_burden",
    "figures",
    "paper_numbers",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", choices=STAGES, default=STAGES[0], help="first stage to run")
    parser.add_argument("--extract-embeddings", action="store_true", help="recompute WavLM embeddings even if they exist")
    args = parser.parse_args()

    source = Path(__file__).parent
    for stage in STAGES[STAGES.index(args.start):]:
        if stage == "wavlm_embeddings" and EMBEDDINGS_PATH.exists() and not args.extract_embeddings:
            print(f"== {stage}: reusing {EMBEDDINGS_PATH}", flush=True)
            continue
        print(f"== {stage}", flush=True)
        started = time.time()
        subprocess.run([sys.executable, str(source / f"{stage}.py")], check=True)
        print(f"   done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
