"""Referentna NumPy implementacija obaveznog API-ja.

Svaka funkcija radi u dtypeu svojih ulaza: float32 za mjerenje brzine,
float64 za golden izlaze. Tenzori su uvijek 2D, a skalarni rezultat
(npr. ukupna suma) ima oblik (1, 1).

Imena u Pythonu su snake_case; rječnik API na dnu povezuje ih s imenima
iz specifikacije (camelCase), koja koriste i testovi ispravnosti.
"""

import numpy as np


def _check_2d(*xs):
    for x in xs:
        if x.ndim != 2:
            raise ValueError(f"očekivan 2D tenzor, dobiven oblik {x.shape}")


def _check_broadcast(a, b):
    """Dopušteno: isti oblik, (M×N) s (1×N) i (M×N) s (M×1)."""
    _check_2d(a, b)
    m, n = a.shape
    if b.shape in ((m, n), (1, n), (m, 1)):
        return
    raise ValueError(f"nekompatibilni oblici {a.shape} i {b.shape}")


# --- stvaranje ---------------------------------------------------------------

def zeros(m, n, dtype=np.float32):
    return np.zeros((m, n), dtype)


def ones(m, n, dtype=np.float32):
    return np.ones((m, n), dtype)


def full(m, n, value, dtype=np.float32):
    return np.full((m, n), value, dtype)


def from_array(data, m, n, dtype=np.float32):
    return np.asarray(data, dtype).reshape(m, n)


def one_hot(labels, classes):
    """labels: (M×1) s cijelim brojevima zapisanima kao float; rezultat (M×classes)."""
    _check_2d(labels)
    if labels.shape[1] != 1:
        raise ValueError(f"oneHot očekuje oblik (M, 1), dobiven {labels.shape}")
    idx = labels[:, 0].astype(np.int64)
    out = np.zeros((labels.shape[0], classes), labels.dtype)
    out[np.arange(labels.shape[0]), idx] = 1
    return out


# --- oblik -------------------------------------------------------------------

def reshape(x, m, n):
    if x.size != m * n:
        raise ValueError(f"ne može se preoblikovati {x.shape} u ({m}, {n})")
    return x.reshape(m, n)


def transpose(x):
    _check_2d(x)
    return x.T


def slice_rows(x, start, end):
    return x[start:end, :]


def slice_cols(x, start, end):
    return x[:, start:end]


def concat_cols(a, b):
    _check_2d(a, b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"concatCols: različit broj redova {a.shape} i {b.shape}")
    return np.concatenate([a, b], axis=1)


# --- po elementima -----------------------------------------------------------

def _binary(fn):
    def op(a, b):
        if np.isscalar(b):
            return fn(a, b)
        _check_broadcast(a, b)
        return fn(a, b)
    return op


add = _binary(np.add)
sub = _binary(np.subtract)
mul = _binary(np.multiply)
div = _binary(np.divide)
minimum = _binary(np.minimum)
maximum = _binary(np.maximum)


def clamp(x, lo, hi):
    return np.clip(x, lo, hi)


def neg(x):
    return -x


def abs_(x):
    return np.abs(x)


def exp(x):
    return np.exp(x)


def log(x):
    return np.log(x)


def sqrt(x):
    return np.sqrt(x)


def rsqrt(x):
    return 1 / np.sqrt(x)


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def gt(x, scalar):
    return (x > scalar).astype(x.dtype)


# --- linearna algebra --------------------------------------------------------

def matmul(a, b):
    _check_2d(a, b)
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul: nekompatibilni oblici {a.shape} i {b.shape}")
    return a @ b


def dot(a, b):
    """Skalarni produkt dvaju vektora istog oblika (1×N ili N×1); rezultat (1×1)."""
    _check_2d(a, b)
    if a.shape != b.shape or 1 not in a.shape:
        raise ValueError(f"dot: očekivana dva vektora istog oblika, dobiveni {a.shape} i {b.shape}")
    return np.array([[np.dot(a.ravel(), b.ravel())]], dtype=np.result_type(a, b))


# --- redukcije ---------------------------------------------------------------

def _reduce(fn):
    def op(x, axis=None):
        _check_2d(x)
        if axis is None:
            return np.array([[fn(x)]], dtype=x.dtype)
        return fn(x, axis=axis, keepdims=True)
    return op


sum_ = _reduce(np.sum)
mean = _reduce(np.mean)
max_ = _reduce(np.max)
min_ = _reduce(np.min)


def argmax(x, axis=None):
    """Indeks najveće vrijednosti, zapisan kao float. Bez osi: indeks u redoslijedu po redovima."""
    _check_2d(x)
    if axis is None:
        return np.array([[np.argmax(x)]], dtype=x.dtype)
    return np.expand_dims(np.argmax(x, axis=axis), axis).astype(x.dtype)


# --- operacije po redovima ---------------------------------------------------

def normalize_rows(x):
    return x * rsqrt(np.sum(x * x, axis=1, keepdims=True))


def softmax_rows(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


API = {
    "zeros": zeros, "ones": ones, "full": full, "fromArray": from_array, "oneHot": one_hot,
    "reshape": reshape, "transpose": transpose, "sliceRows": slice_rows,
    "sliceCols": slice_cols, "concatCols": concat_cols,
    "add": add, "sub": sub, "mul": mul, "div": div, "minimum": minimum, "maximum": maximum,
    "clamp": clamp, "neg": neg, "abs": abs_, "exp": exp, "log": log, "sqrt": sqrt,
    "rsqrt": rsqrt, "relu": relu, "sigmoid": sigmoid, "tanh": tanh, "gt": gt,
    "matmul": matmul, "dot": dot,
    "sum": sum_, "mean": mean, "max": max_, "min": min_, "argmax": argmax,
    "normalizeRows": normalize_rows, "softmaxRows": softmax_rows,
}
