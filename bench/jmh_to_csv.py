"""Pretvara JMH JSON rezultate u službeni CSV format.

    java -jar target/benchmarks.jar -rf json -rff results/jmh_all.json
    java -jar target/benchmarks.jar -jvmArgsAppend "-Dtensor.threads=1" -rf json -rff results/jmh_1.json
    python bench/jmh_to_csv.py results/jmh_all.json --threads all
    python bench/jmh_to_csv.py results/jmh_1.json --threads 1

Pravila imenovanja u Javi:
  - ime metode počinje ID-jem benchmarka i podvlakom: B1_matmul64, B10_epoch, ...
  - @Param "impl" je naive ili fast, @Param "league" je precise ili fast
  - način mjerenja je Mode.AverageTime

Stupac error ostaje prazan; službenu grešku računa score.py iz izlaza u out/.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.spec import BY_ID, CSV_FIELDS, LEAGUES  # noqa: E402

TO_MS = {"ns/op": 1e-6, "us/op": 1e-3, "ms/op": 1.0, "s/op": 1e3}
METHOD = re.compile(r"^(B\d+)(?:_|$)")


def convert(entries, threads):
    rows = []
    for e in entries:
        method = e["benchmark"].rsplit(".", 1)[-1]
        m = METHOD.match(method)
        if not m or m.group(1) not in BY_ID:
            print(f"preskačem {e['benchmark']}: ime metode ne počinje s B1_ ... B10_", file=sys.stderr)
            continue
        params = e.get("params", {})
        impl = params.get("impl", "fast")
        league = params.get("league", "precise")
        if impl not in ("naive", "fast") or league not in LEAGUES:
            sys.exit(f"{method}: impl mora biti naive ili fast, league precise ili fast")
        metric = e["primaryMetric"]
        unit = metric["scoreUnit"]
        if unit not in TO_MS:
            sys.exit(f"{method}: jedinica {unit} nije vrijeme po operaciji; koristite Mode.AverageTime")
        samples = [v * TO_MS[unit] for fork in metric["rawData"] for v in fork]
        rows.append({"benchmark": m.group(1), "impl": impl, "threads": threads, "league": league,
                     "median_ms": f"{statistics.median(samples):.4f}",
                     "p10_ms": f"{np.percentile(samples, 10):.4f}",
                     "p90_ms": f"{np.percentile(samples, 90):.4f}",
                     "error": ""})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", type=Path)
    ap.add_argument("--threads", choices=["1", "all"], required=True)
    ap.add_argument("--csv", type=Path, help="zadano: results/java_<threads>.csv")
    args = ap.parse_args()
    rows = convert(json.loads(args.json.read_text()), args.threads)
    out = args.csv or ROOT / "results" / f"java_{args.threads}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} redaka -> {out}")


if __name__ == "__main__":
    main()
