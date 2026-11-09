"""Generira sve ulazne podatke paketa u data/ (fiksni seedovi, float32 .npy).

    python data/generate_data.py --mnist-dir ~/Downloads/mnist   # MNIST već skinut
    python data/generate_data.py --download                      # skini MNIST (~11 MB)
    python data/generate_data.py --synthetic-mnist               # samo za probu paketa

Na kraju zapisuje data/MANIFEST.txt sa SHA-256 svake datoteke, pa studenti
mogu provjeriti da imaju iste podatke kao nastavnik.
"""

import argparse
import gzip
import hashlib
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference.render import H, W, look_at, perspective  # noqa: E402

DATA = ROOT / "data"
SEED = 2026

MESHES = {"10k": (200, 50), "100k": (1000, 100), "1m": (4000, 250)}

MNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte",
    "train_y": "train-labels-idx1-ubyte",
    "test_x": "t10k-images-idx3-ubyte",
    "test_y": "t10k-labels-idx1-ubyte",
}


def rng(stream):
    return np.random.default_rng([SEED, stream])


def save(rel, arr):
    path = DATA / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{rel}: svi tenzori moraju biti 2D, dobiven {arr.shape}")
    np.save(path, arr)
    print(f"  {rel:<38} {str(arr.shape):>16}")


# --- benchmark ulazi ---------------------------------------------------------

def bench_inputs():
    print("benchmark ulazi")
    for bid, n, stream in (("B1", 64, 1), ("B2", 512, 2), ("B3", 2048, 3)):
        g = rng(stream)
        save(f"bench/{bid}_a.npy", g.uniform(-1, 1, (n, n)))
        save(f"bench/{bid}_b.npy", g.uniform(-1, 1, (n, n)))
    g = rng(4)
    for name in ("a", "b", "c"):
        save(f"bench/B4_{name}.npy", g.uniform(-1, 1, (4096, 4096)).astype(np.float32))
    save("bench/B6_v.npy", rng(6).standard_normal((1_000_000, 3)))


# --- scena i modeli ----------------------------------------------------------

def scene():
    print("scena")
    save("scene/V.npy", look_at(eye=(0, 0.8, 10.5), target=(0, 0, 0), up=(0, 1, 0)))
    save("scene/P.npy", perspective(45, W / H, 1, 30))
    light = np.array([[0.4, 0.8, 0.6]])
    save("scene/light.npy", light / np.linalg.norm(light))
    save("scene/albedo.npy", [[0.90, 0.55, 0.20]])
    save("scene/ambient.npy", [[0.12]])


def torus_knot(u, v, p=2, q=3, tube=0.45):
    """Cijev oko (p, q) torusnog čvora: u točaka uzduž čvora, v oko cijevi."""
    t = np.arange(u) * (2 * np.pi / u)
    r = 2 + np.cos(q * t)
    dr = -q * np.sin(q * t)
    ddr = -q * q * np.cos(q * t)
    cp, sp = np.cos(p * t), np.sin(p * t)
    c = np.stack([r * cp, r * sp, -np.sin(q * t)], axis=1)
    d1 = np.stack([dr * cp - p * r * sp, dr * sp + p * r * cp, -q * np.cos(q * t)], axis=1)
    d2 = np.stack([ddr * cp - 2 * p * dr * sp - p * p * r * cp,
                   ddr * sp + 2 * p * dr * cp - p * p * r * sp,
                   q * q * np.sin(q * t)], axis=1)
    tang = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    binorm = np.cross(d1, d2)
    bn = np.linalg.norm(binorm, axis=1, keepdims=True)
    assert bn.min() > 1e-6, "čvor ima točku bez zakrivljenosti"
    binorm /= bn
    norm = np.cross(binorm, tang)

    phi = np.arange(v) * (2 * np.pi / v)
    radial = (np.cos(phi)[None, :, None] * norm[:, None, :]
              + np.sin(phi)[None, :, None] * binorm[:, None, :])
    positions = (c[:, None, :] + tube * radial).reshape(-1, 3)
    normals = radial.reshape(-1, 3)

    i, j = np.meshgrid(np.arange(u), np.arange(v), indexing="ij")
    a = i * v + j
    b = ((i + 1) % u) * v + j
    cc = ((i + 1) % u) * v + (j + 1) % v
    d = i * v + (j + 1) % v
    tri = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3),
                          np.stack([b, cc, d], -1).reshape(-1, 3)])

    # Trokuti su CCW gledano izvana: normala trokuta gleda kao normala vrha.
    e1 = positions[tri[:, 1]] - positions[tri[:, 0]]
    e2 = positions[tri[:, 2]] - positions[tri[:, 0]]
    facing = np.sum(np.cross(e1, e2) * normals[tri[:, 0]], axis=1)
    if np.mean(facing > 0) < 0.5:
        tri = tri[:, [0, 2, 1]]
        facing = -facing
    assert np.mean(facing > 0) > 0.999, "trokuti nemaju dosljednu orijentaciju"
    return positions, normals, tri


def meshes():
    print("modeli (torusni čvor)")
    for name, (u, v) in MESHES.items():
        positions, normals, tri = torus_knot(u, v)
        assert tri.max() < 2 ** 24, "indeksi moraju biti točno zapisivi u float32"
        save(f"mesh/mesh_{name}_positions.npy", positions)
        save(f"mesh/mesh_{name}_normals.npy", normals)
        save(f"mesh/mesh_{name}_triangles.npy", tri)


# --- MNIST i težine ----------------------------------------------------------

def read_idx(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        raw = fh.read()
    zero, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    if zero != 0 or dtype_code != 0x08:
        raise ValueError(f"{path} nije IDX datoteka s uint8 podacima")
    dims = struct.unpack(f">{ndim}I", raw[4:4 + 4 * ndim])
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def find_idx(directory, stem):
    for name in (stem, stem + ".gz"):
        if (directory / name).exists():
            return directory / name
    raise FileNotFoundError(f"nema {stem}[.gz] u {directory}")


def download_mnist(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for stem in MNIST_FILES.values():
        target = raw_dir / f"{stem}.gz"
        if not target.exists():
            print(f"  skidam {MNIST_MIRROR}{stem}.gz")
            urllib.request.urlretrieve(MNIST_MIRROR + stem + ".gz", target)


def real_mnist(directory):
    out = {k: read_idx(find_idx(directory, stem)) for k, stem in MNIST_FILES.items()}
    expected = {"train_x": (60000, 28, 28), "train_y": (60000,),
                "test_x": (10000, 28, 28), "test_y": (10000,)}
    for k, shape in expected.items():
        if out[k].shape != shape:
            raise ValueError(f"MNIST {k}: očekivan oblik {shape}, dobiven {out[k].shape}")
    return out


def synthetic_mnist():
    """Naučiv skup istih oblika kao MNIST. Samo za probu paketa, nikad za ocjenjivanje."""
    g = rng(7)
    protos = (g.random((10, 28, 28)) < 0.25).astype(np.float64)
    out = {}
    for split, n in (("train", 60000), ("test", 10000)):
        y = g.integers(0, 10, n)
        x = protos[y] * g.uniform(0.6, 1.0, (n, 1, 1)) + g.normal(0, 0.2, (n, 28, 28))
        out[f"{split}_x"] = np.clip(x * 255, 0, 255).astype(np.uint8)
        out[f"{split}_y"] = y.astype(np.uint8)
    return out


def mnist(args):
    print("MNIST")
    if args.synthetic_mnist:
        source, raw = "synthetic", synthetic_mnist()
    else:
        directory = args.mnist_dir or DATA / "raw"
        if args.download:
            download_mnist(directory)
        source, raw = "idx", real_mnist(directory)
    for split in ("train", "test"):
        x = raw[f"{split}_x"].reshape(-1, 784).astype(np.float64) / 255
        save(f"mnist/{split}_x.npy", x)
        save(f"mnist/{split}_y.npy", raw[f"{split}_y"].reshape(-1, 1))
    return source


def mlp_weights():
    print("početne težine (He)")
    g = rng(8)
    save("mlp/W1.npy", g.standard_normal((784, 256)) * np.sqrt(2 / 784))
    save("mlp/b1.npy", np.zeros((1, 256)))
    save("mlp/W2.npy", g.standard_normal((256, 10)) * np.sqrt(2 / 256))
    save("mlp/b2.npy", np.zeros((1, 10)))


def manifest(mnist_source):
    lines = [f"# seed {SEED}", f"# mnist {mnist_source}"]
    for path in sorted(DATA.rglob("*.npy")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(DATA).as_posix()}")
    (DATA / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    print(f"MANIFEST.txt: {len(lines) - 2} datoteka")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--mnist-dir", type=Path, help="mapa s četiri MNIST IDX datoteke (.gz ili raspakirane)")
    src.add_argument("--synthetic-mnist", action="store_true", help="umjetni skup, samo za probu paketa")
    ap.add_argument("--download", action="store_true", help="skini MNIST u data/raw/ (ili --mnist-dir)")
    args = ap.parse_args()
    bench_inputs()
    scene()
    meshes()
    source = mnist(args)
    mlp_weights()
    manifest(source)


if __name__ == "__main__":
    main()
