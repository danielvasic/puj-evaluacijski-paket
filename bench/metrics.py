"""Greška i prolaz kroz ligu, isti za NumPy i Javu."""

import math
from pathlib import Path

import numpy as np

from bench.spec import BY_ID


def norm_error(x, ref):
    """e = max|x - ref| / max|ref|; pogrešan oblik ili NaN daju beskonačno."""
    if x.shape != ref.shape:
        return math.inf
    x, ref = x.astype(np.float64), ref.astype(np.float64)
    if not np.all(np.isfinite(x)):
        return math.inf
    scale = np.max(np.abs(ref))
    diff = np.max(np.abs(x - ref))
    return float(diff / scale) if scale > 0 else float(diff)


def benchmark_error(bid, outputs, golden_dir):
    """Greška benchmarka iz rječnika izlaza {ime: niz}.

    Za B10 to je razlika točnosti u postotnim bodovima (s predznakom),
    a za ostale najveća normalizirana greška po svim izlazima.
    """
    golden_dir = Path(golden_dir)
    spec = BY_ID[bid]
    if spec.get("metric") == "accuracy":
        ref = np.load(golden_dir / "B10_accuracy.npy")[0, 0]
        acc = outputs.get("B10_accuracy")
        if acc is None or acc.shape != (1, 1) or not np.isfinite(acc[0, 0]):
            return math.inf
        return float(acc[0, 0] - ref)
    worst = 0.0
    for name in spec["outputs"]:
        if name not in outputs:
            return math.inf
        worst = max(worst, norm_error(outputs[name], np.load(golden_dir / f"{name}.npy")))
    return worst


def passes(bid, league, error):
    spec = BY_ID[bid]
    if error is None or math.isnan(error) or math.isinf(error):
        return False
    if spec.get("metric") == "accuracy":
        return abs(error) <= spec["precise"] if league == "precise" else error >= -spec["fast"]
    return error <= spec[league]


def load_outputs(directory, bid):
    """Učita izlaze benchmarka iz mape (npr. out/fast_precise/); nedostajući se preskaču."""
    directory = Path(directory)
    found = {}
    for name in BY_ID[bid]["outputs"]:
        path = directory / f"{name}.npy"
        if path.exists():
            found[name] = np.load(path)
    return found
