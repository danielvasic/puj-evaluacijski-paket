"""Spaja CSV rezultate, računa greške iz izlaza i boduje automatski mjerljive dijelove.

    python score.py --results results/*.csv --outputs out/ [--memory memory.json] [--report izvjestaj/]

Izlazi Jave očekuju se u out/<impl>_<league>/<ime>.npy (npr. out/fast_precise/B3.npy).
Boduje se: ubrzanje naspram baselinea (15), usporedba s NumPyjem (10), memorija (5).
Brzu ligu (5) nastavnik boduje ručno; izvještaj navodi koji benchmarkovi u njoj prolaze.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bench import metrics  # noqa: E402
from bench.spec import BENCHMARKS  # noqa: E402

IDS = [b["id"] for b in BENCHMARKS]
SPEEDUP_STEPS = ((25, 15), (10, 10), (3, 5))


def load_rows(paths):
    rows = {}
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[(r["benchmark"], r["impl"], r["threads"], r["league"])] = r
    return rows


def time_of(rows, bid, impl, league="precise", prefer=("all", "1")):
    for threads in prefer:
        r = rows.get((bid, impl, threads, league))
        if r:
            return float(r["median_ms"])
    return None


def error_of(rows, outputs, bid, impl, league):
    if impl == "numpy":
        r = rows.get((bid, "numpy", "all", league)) or rows.get((bid, "numpy", "1", league))
        return float(r["error"]) if r and r["error"] else math.inf
    found = metrics.load_outputs(outputs / f"{impl}_{league}", bid)
    return metrics.benchmark_error(bid, found, ROOT / "golden") if found else math.inf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, default=ROOT / "out")
    ap.add_argument("--memory", type=Path)
    ap.add_argument("--report", type=Path, default=ROOT / "izvjestaj")
    args = ap.parse_args()
    rows = load_rows(args.results)
    args.report.mkdir(parents=True, exist_ok=True)

    lines = ["# Izvještaj bodovanja", "",
             "| ID | naive ms | fast ms | NumPy ms | greška (precizna) | prolazi | ubrzanje | R = NumPy/Java | bodovi NumPy |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    speedups, numpy_points, chart = [], 0.0, []
    for bid in IDS:
        t_naive = time_of(rows, bid, "naive", prefer=("1", "all"))
        t_fast = time_of(rows, bid, "fast")
        t_np = time_of(rows, bid, "numpy")
        err = error_of(rows, args.outputs, bid, "fast", "precise")
        ok = metrics.passes(bid, "precise", err)
        # Benchmark koji ne prolazi ili nije izmjeren ulazi u geometrijsku sredinu kao 1×.
        sp = t_naive / t_fast if ok and t_naive and t_fast else 1.0
        speedups.append(sp)
        ratio = t_np / t_fast if ok and t_np and t_fast else None
        pts = 1.0 if ratio and ratio >= 1 else 0.5 if ratio and ratio >= 0.5 else 0.0
        numpy_points += pts
        chart.append((bid, sp, ratio))
        fmt = lambda v: f"{v:.3f}" if v is not None else "–"  # noqa: E731
        lines.append(f"| {bid} | {fmt(t_naive)} | {fmt(t_fast)} | {fmt(t_np)} | {err:.2e} | "
                     f"{'da' if ok else 'NE'} | {sp:.1f}× | {fmt(ratio)} | {pts} |")

    geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    speed_points = next((p for limit, p in SPEEDUP_STEPS if geo >= limit), 0)

    mem_points = None
    if args.memory:
        m = json.loads(args.memory.read_text())
        mem_points = (2 if m["bytes_per_element"] <= 4 else 0) + \
                     (3 if m["challenge_seconds"] <= 180 and m["challenge_peak_rss_mb"] <= 1536 else 0)

    fast_ok = [bid for bid in IDS
               if time_of(rows, bid, "fast", league="fast") is not None
               and metrics.passes(bid, "fast", error_of(rows, args.outputs, bid, "fast", "fast"))]

    lines += ["", f"Geometrijska sredina ubrzanja: **{geo:.1f}×** → **{speed_points} / 15** bodova",
              f"Usporedba s NumPyjem: **{numpy_points:g} / 10** bodova",
              f"Memorija: **{mem_points if mem_points is not None else 'nije predano'} / 5** bodova",
              f"Brza liga prolazi na: {', '.join(fast_ok) or 'nijednom'} (bodovi ručno, iz dnevnika)"]
    (args.report / "izvjestaj.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar([c[0] for c in chart], [c[1] for c in chart])
        axes[0].set(yscale="log", title="Ubrzanje naspram baselinea", ylabel="× (log)")
        axes[1].bar([c[0] for c in chart], [c[2] or 0 for c in chart])
        axes[1].axhline(1, color="gray", linestyle="--")
        axes[1].set(yscale="log", title="R = t_NumPy / t_Java (iznad 1: Java brža)", ylabel="omjer (log)")
        fig.tight_layout()
        fig.savefig(args.report / "grafovi.png", dpi=120)
    except ImportError:
        print("matplotlib nije instaliran, grafovi preskočeni")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
