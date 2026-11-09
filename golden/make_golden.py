"""Računa golden izlaze B1-B10 u float64 i sprema ih u golden/ kao float32.

    python golden/make_golden.py            # svi benchmarkovi
    python golden/make_golden.py B7 B10     # samo odabrani

Pokreće se nakon data/generate_data.py.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.spec import BENCHMARKS  # noqa: E402
from reference import workloads  # noqa: E402

GOLDEN = ROOT / "golden"


def main():
    wanted = sys.argv[1:] or [b["id"] for b in BENCHMARKS]
    for bid in wanted:
        start = time.perf_counter()
        w = workloads.make(bid, ROOT / "data", np.float64)
        w.reset()
        outputs = w.finalize(w.run())
        for name, arr in outputs.items():
            np.save(GOLDEN / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float32))
        shapes = ", ".join(f"{k} {v.shape}" for k, v in outputs.items())
        print(f"{bid:<4} {time.perf_counter() - start:7.1f} s   {shapes}")
        if bid == "B10":
            print(f"     točnost reference: {outputs['B10_accuracy'][0, 0]:.2f} %")


if __name__ == "__main__":
    main()
