"""Generira testove ispravnosti za obavezni API.

Svaki test ima ulaze (float32 .npy), argumente i očekivani izlaz izračunat
u float64 (spremljen kao float32). Popis testova zapisuje se dvaput:
cases.json i cases.tsv (lako se čita iz Jave bez ikakve biblioteke).

Oblici su namjerno "neugodni" (1×1, neparni, ostaci pri dijeljenju s 8),
kako bi se uhvatile greške u petljama i SIMD repovima.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference.ops import API  # noqa: E402

HERE = ROOT / "conformance"
CASES = HERE / "cases"
TOLERANCE = 1e-4

g = np.random.default_rng([2026, 100])
cases = []


def uniform(shape, lo=-1.0, hi=1.0):
    return g.uniform(lo, hi, shape).astype(np.float32)


def away_from_zero(shape):
    return (g.uniform(0.5, 2.0, shape) * g.choice([-1, 1], shape)).astype(np.float32)


def case(cid, op, inputs=(), **args):
    """args moraju biti navedeni redom kojim ih funkcija prima."""
    wide = [x.astype(np.float64) for x in inputs]
    expected = np.asarray(API[op](*wide, *args.values()))
    files = []
    for k, x in enumerate(inputs):
        name = f"{cid}__in{k}.npy"
        np.save(CASES / name, np.ascontiguousarray(x))
        files.append(f"cases/{name}")
    exp_name = f"{cid}__expected.npy"
    np.save(CASES / exp_name, np.ascontiguousarray(expected, dtype=np.float32))
    cases.append({"id": cid, "op": op, "inputs": files, "args": args,
                  "expected": f"cases/{exp_name}", "shape": list(expected.shape)})


def shape_tag(s):
    return f"{s[0]}x{s[1]}"


def build():
    big = (130, 257)

    for (m, n) in [(3, 4), (1, 7)]:
        case(f"zeros_{m}x{n}", "zeros", m=m, n=n)
        case(f"ones_{m}x{n}", "ones", m=m, n=n)
    case("full_2x5", "full", m=2, n=5, value=-1.5)

    case("oneHot_7_10", "oneHot", [g.integers(0, 10, (7, 1)).astype(np.float32)], classes=10)
    case("oneHot_1_5", "oneHot", [np.array([[3]], np.float32)], classes=5)

    case("reshape_6x4_3x8", "reshape", [uniform((6, 4))], m=3, n=8)
    case("reshape_12x2_1x24", "reshape", [uniform((12, 2))], m=1, n=24)
    case("reshape_130x257_257x130", "reshape", [uniform(big)], m=257, n=130)
    for s in [(7, 5), (1, 9), big]:
        case(f"transpose_{shape_tag(s)}", "transpose", [uniform(s)])
    case("sliceRows_7x5_2_5", "sliceRows", [uniform((7, 5))], start=2, end=5)
    case("sliceCols_7x5_1_4", "sliceCols", [uniform((7, 5))], start=1, end=4)
    case("sliceCols_130x257_3_4", "sliceCols", [uniform(big)], start=3, end=4)
    case("concatCols_7x5_7x1", "concatCols", [uniform((7, 5)), uniform((7, 1))])
    case("concatCols_130x257_130x3", "concatCols", [uniform(big), uniform((130, 3))])

    for op in ("add", "sub", "mul", "div", "minimum", "maximum"):
        second = away_from_zero if op == "div" else uniform
        for s in [(1, 1), (7, 5), (33, 17), big]:
            case(f"{op}_{shape_tag(s)}", op, [uniform(s), second(s)])
        for s in [(7, 5), big]:
            case(f"{op}_row_{shape_tag(s)}", op, [uniform(s), second((1, s[1]))])
            case(f"{op}_col_{shape_tag(s)}", op, [uniform(s), second((s[0], 1))])

    for op, scalar in (("add", 2.5), ("sub", -0.75), ("mul", -1.25), ("div", 0.4)):
        for s in [(7, 5), big]:
            case(f"{op}_scalar_{shape_tag(s)}", op, [uniform(s)], scalar=scalar)
    for s in [(7, 5), big]:
        case(f"clamp_{shape_tag(s)}", "clamp", [uniform(s)], lo=-0.3, hi=0.6)
        case(f"gt_{shape_tag(s)}", "gt", [uniform(s)], scalar=0.25)

    domains = {"neg": (-1, 1), "abs": (-1, 1), "exp": (-5, 5), "log": (0.01, 10),
               "sqrt": (0.01, 10), "rsqrt": (0.01, 10), "relu": (-1, 1),
               "sigmoid": (-8, 8), "tanh": (-4, 4)}
    for op, (lo, hi) in domains.items():
        for s in [(1, 1), (7, 5), big]:
            case(f"{op}_{shape_tag(s)}", op, [uniform(s, lo, hi)])

    for a, b in [((1, 1), (1, 1)), ((4, 4), (4, 4)), ((7, 5), (5, 3)), ((1, 8), (8, 1)),
                 ((8, 1), (1, 8)), ((33, 17), (17, 65)), ((64, 64), (64, 64)),
                 ((130, 257), (257, 129)), ((1000, 4), (4, 4))]:
        case(f"matmul_{shape_tag(a)}_{shape_tag(b)}", "matmul", [uniform(a), uniform(b)])
    for s in [(1, 9), (9, 1), (1, 1031)]:
        case(f"dot_{shape_tag(s)}", "dot", [uniform(s), uniform(s)])

    for op in ("sum", "mean", "max", "min", "argmax"):
        for s in [(7, 5), (1, 9), (9, 1), big]:
            x = uniform(s)
            case(f"{op}_all_{shape_tag(s)}", op, [x])
            case(f"{op}_axis0_{shape_tag(s)}", op, [x], axis=0)
            case(f"{op}_axis1_{shape_tag(s)}", op, [x], axis=1)

    for s in [(7, 3), (1000, 3), (5, 1), (33, 17)]:
        case(f"normalizeRows_{shape_tag(s)}", "normalizeRows", [g.standard_normal(s).astype(np.float32)])
    case("softmaxRows_7x10", "softmaxRows", [uniform((7, 10), -3, 3)])
    case("softmaxRows_64x10_large", "softmaxRows", [uniform((64, 10), -90, 90)])
    case("softmaxRows_1x1", "softmaxRows", [uniform((1, 1))])
    case("softmaxRows_5x33", "softmaxRows", [uniform((5, 33), -3, 3)])


def write_index():
    meta = {"version": 1, "tolerance": TOLERANCE,
            "error": "max|x - expected| / max|expected|, oblik mora biti jednak",
            "cases": cases}
    (HERE / "cases.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    lines = ["# id\top\tinputs\targs\texpected"]
    for c in cases:
        inputs = ",".join(c["inputs"]) or "-"
        args = ";".join(f"{k}={v}" for k, v in c["args"].items()) or "-"
        lines.append(f"{c['id']}\t{c['op']}\t{inputs}\t{args}\t{c['expected']}")
    (HERE / "cases.tsv").write_text("\n".join(lines) + "\n")


def main():
    shutil.rmtree(CASES, ignore_errors=True)
    CASES.mkdir(parents=True)
    build()
    write_index()
    print(f"{len(cases)} testova u {CASES.relative_to(ROOT)}/, popis u cases.json i cases.tsv")


if __name__ == "__main__":
    main()
