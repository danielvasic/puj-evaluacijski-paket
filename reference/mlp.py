"""Referentna neuronska mreža za Demo B: 784 → 256 (ReLU) → 10 (softmax).

Treniranje točno prati specifikaciju: SGD bez momentuma, batch 64, stopa
učenja 0.1, jedna epoha, batchevi redom, zadnji nepotpuni batch se preskače.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference import ops  # noqa: E402

BATCH = 64
LR = 0.1
LOG_EVERY = 100
P_MIN = 1e-7


def load_params(data_dir, dtype=np.float32):
    d = Path(data_dir) / "mlp"
    return {k: np.load(d / f"{k}.npy").astype(dtype) for k in ("W1", "b1", "W2", "b2")}


def load_mnist(data_dir, dtype=np.float32):
    d = Path(data_dir) / "mnist"
    return {k: np.load(d / f"{k}.npy").astype(dtype)
            for k in ("train_x", "train_y", "test_x", "test_y")}


def forward(x, p):
    h = ops.relu(ops.add(ops.matmul(x, p["W1"]), p["b1"]))
    z = ops.add(ops.matmul(h, p["W2"]), p["b2"])
    return h, z


def infer(x, p):
    """B9: logiti za cijeli skup u jednom prolazu."""
    return forward(x, p)[1]


def loss(prob, y):
    picked = ops.sum_(ops.mul(y, ops.log(ops.clamp(prob, P_MIN, 1))), axis=1)
    return -float(ops.mean(picked)[0, 0])


def train_epoch(train_x, train_y1h, p):
    """B10: jedna epoha; mijenja `p` na mjestu i vraća gubitke (svakih 100 koraka) kao (K×1)."""
    steps = train_x.shape[0] // BATCH
    losses = []
    for s in range(steps):
        x = ops.slice_rows(train_x, s * BATCH, (s + 1) * BATCH)
        y = ops.slice_rows(train_y1h, s * BATCH, (s + 1) * BATCH)
        h, z = forward(x, p)
        prob = ops.softmax_rows(z)
        if s % LOG_EVERY == 0:
            losses.append(loss(prob, y))
        dz = ops.div(ops.sub(prob, y), BATCH)
        dw2 = ops.matmul(ops.transpose(h), dz)
        db2 = ops.sum_(dz, axis=0)
        dh = ops.mul(ops.matmul(dz, ops.transpose(p["W2"])), ops.gt(h, 0))
        dw1 = ops.matmul(ops.transpose(x), dh)
        db1 = ops.sum_(dh, axis=0)
        p["W1"] -= LR * dw1
        p["b1"] -= LR * db1
        p["W2"] -= LR * dw2
        p["b2"] -= LR * db2
    return np.array(losses, dtype=train_x.dtype).reshape(-1, 1)


def accuracy(logits, labels):
    """Točnost u postocima; labels je (M×1)."""
    pred = ops.argmax(logits, axis=1)
    return float(np.mean(pred[:, 0] == labels[:, 0]) * 100)
