"""Provjerava izlaze testova ispravnosti.

    python conformance/check.py OUT_DIR        # vaši izlazi: OUT_DIR/<id>.npy za svaki test
    python conformance/check.py --reference    # provjera same NumPy reference u float32

Izlazni kod je 0 samo ako prođu svi testovi.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.metrics import norm_error  # noqa: E402
from reference.ops import API  # noqa: E402

HERE = ROOT / "conformance"


def reference_output(c):
    inputs = [np.load(HERE / p) for p in c["inputs"]]
    return np.asarray(API[c["op"]](*inputs, *c["args"].values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", nargs="?", type=Path)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true", help="ispiši i testove koji prolaze")
    args = ap.parse_args()
    if not args.reference and args.out_dir is None:
        ap.error("navedite OUT_DIR ili --reference")

    meta = json.loads((HERE / "cases.json").read_text())
    tol = meta["tolerance"]
    failed = 0
    for c in meta["cases"]:
        expected = np.load(HERE / c["expected"])
        if args.reference:
            got = reference_output(c)
        else:
            path = args.out_dir / f"{c['id']}.npy"
            if not path.exists():
                print(f"NEMA   {c['id']}")
                failed += 1
                continue
            got = np.load(path)
        if got.shape != expected.shape:
            print(f"OBLIK  {c['id']}: dobiveno {got.shape}, očekivano {expected.shape}")
            failed += 1
            continue
        err = norm_error(got, expected)
        if err > tol:
            print(f"PAD    {c['id']}: greška {err:.2e} > {tol:.0e}")
            failed += 1
        elif args.verbose:
            print(f"OK     {c['id']}: {err:.1e}")
    total = len(meta["cases"])
    print(f"\nprošlo {total - failed} / {total}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
