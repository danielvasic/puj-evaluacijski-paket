"""NumPy strana benchmarkova: mjeri B1-B10 u float32 i piše CSV u službenom formatu.

    python bench/bench_numpy.py --threads all
    python bench/bench_numpy.py --threads 1
    python bench/bench_numpy.py --only B1,B7 --quick     # brza proba

Izlazi se spremaju u out/numpy_precise/, a greška se računa prema golden/.
"""

import os
import sys


def _threads_from_argv(argv):
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return "all"


# BLAS čita broj dretvi pri učitavanju NumPyja, pa se varijable postavljaju prije importa.
if _threads_from_argv(sys.argv[1:]) == "1":
    for var in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import metrics  # noqa: E402
from bench.spec import BENCHMARKS, BY_ID, CSV_FIELDS, NUMPY_REPS  # noqa: E402
from reference import workloads  # noqa: E402


def percentile(values, q):
    return float(np.percentile(values, q))


def blas_name():
    try:
        return np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:
        return "nepoznat"


def measure(bid, warmup, runs):
    w = workloads.make(bid, ROOT / "data", np.float32)
    times = []
    for i in range(warmup + runs):
        w.reset()
        start = time.perf_counter_ns()
        result = w.run()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if i >= warmup:
            times.append(elapsed)
    return times, w.finalize(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", choices=["1", "all"], default="all")
    ap.add_argument("--only", help="popis benchmarkova odvojen zarezom, npr. B1,B7")
    ap.add_argument("--quick", action="store_true", help="1 zagrijavanje i 3 mjerenja (nije službeno)")
    ap.add_argument("--csv", type=Path, help="zadano: results/numpy_<threads>.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "numpy_precise")
    args = ap.parse_args()

    ids = args.only.split(",") if args.only else [b["id"] for b in BENCHMARKS]
    csv_path = args.csv or ROOT / "results" / f"numpy_{args.threads}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    blas = blas_name()
    print(f"NumPy {np.__version__}, BLAS: {blas}, dretve: {args.threads}, "
          f"{platform.machine()} {platform.system()}")
    if "openblas" not in blas.lower() or platform.system() != "Linux":
        print("napomena: službeno se mjeri na Linuxu s OpenBLAS-om; ovi rezultati nisu usporedivi sa službenima")
    print(f"{'ID':<4} {'medijan ms':>11} {'p10':>9} {'p90':>9}  {'greška':>9}  liga")
    rows = []
    for bid in ids:
        warmup, runs = (1, 3) if args.quick else NUMPY_REPS[BY_ID[bid]["reps"]]
        times, outputs = measure(bid, warmup, runs)
        for name, arr in outputs.items():
            np.save(args.out / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float32))
        err = metrics.benchmark_error(bid, outputs, ROOT / "golden")
        ok = "prolazi" if metrics.passes(bid, "precise", err) else "NE PROLAZI"
        row = {"benchmark": bid, "impl": "numpy", "threads": args.threads, "league": "precise",
               "median_ms": f"{statistics.median(times):.4f}",
               "p10_ms": f"{percentile(times, 10):.4f}", "p90_ms": f"{percentile(times, 90):.4f}",
               "error": f"{err:.3e}"}
        rows.append(row)
        print(f"{bid:<4} {row['median_ms']:>11} {percentile(times, 10):9.3f} {percentile(times, 90):9.3f}"
              f"  {err:9.2e}  {ok}")

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path}")


if __name__ == "__main__":
    main()
