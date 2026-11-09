"""Benchmarkovi B1-B10 kao NumPy izračuni u zadanom dtypeu.

`make(bid, data_dir, dtype)` vraća Workload: `reset()` se poziva prije svakog
mjerenja (izvan mjerenog vremena), `run()` je mjereni dio, a `finalize()`
od rezultata složi izlaze pod imenima iz bench/spec.py.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.spec import B7_FRAME, B7_MESH, B8_MESH  # noqa: E402
from reference import mlp, ops, render  # noqa: E402


@dataclass
class Workload:
    run: Callable[[], object]
    finalize: Callable[[object], dict]
    reset: Callable[[], None] = lambda: None


def _named(*names):
    return lambda result: dict(zip(names, result if isinstance(result, tuple) else (result,)))


def make(bid, data_dir, dtype):
    d = Path(data_dir)

    def load(rel):
        return np.load(d / rel).astype(dtype)

    if bid in ("B1", "B2", "B3"):
        a, b = load(f"bench/{bid}_a.npy"), load(f"bench/{bid}_b.npy")
        return Workload(lambda: ops.matmul(a, b), _named(bid))

    if bid == "B4":
        a, b, c = (load(f"bench/B4_{k}.npy") for k in "abc")
        return Workload(lambda: ops.relu(ops.add(ops.mul(a, b), c)), _named("B4"))

    if bid == "B5":
        x = load("bench/B4_a.npy")
        return Workload(lambda: (ops.sum_(x, axis=0), ops.sum_(x, axis=1), ops.max_(x)),
                        _named("B5_sum0", "B5_sum1", "B5_max"))

    if bid == "B6":
        v = load("bench/B6_v.npy")
        return Workload(lambda: ops.normalize_rows(v), _named("B6"))

    if bid == "B7":
        scene = render.load_scene(d, dtype)
        pos, nrm = render.load_mesh(d, B7_MESH, dtype)
        return Workload(lambda: render.frame(scene, pos, nrm, B7_FRAME),
                        _named("B7_screen", "B7_depth", "B7_color"))

    if bid == "B8":
        scene = render.load_scene(d, dtype)
        pos, nrm = render.load_mesh(d, B8_MESH, dtype)
        return Workload(lambda: render.render_frames(scene, pos, nrm),
                        _named("B8_screen", "B8_depth", "B8_color"))

    if bid == "B9":
        params = mlp.load_params(d, dtype)
        test_x = load("mnist/test_x.npy")
        return Workload(lambda: mlp.infer(test_x, params), _named("B9_logits"))

    if bid == "B10":
        init = mlp.load_params(d, dtype)
        data = mlp.load_mnist(d, dtype)
        train_y1h = ops.one_hot(data["train_y"], 10)
        params = {}

        def reset():
            params.clear()
            params.update({k: v.copy() for k, v in init.items()})

        def finalize(losses):
            acc = mlp.accuracy(mlp.infer(data["test_x"], params), data["test_y"])
            return {"B10_loss": losses, "B10_accuracy": np.array([[acc]], dtype)}

        return Workload(lambda: mlp.train_epoch(data["train_x"], train_y1h, params), finalize, reset)

    raise ValueError(f"nepoznat benchmark {bid}")
