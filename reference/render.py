"""Referentni geometrijski pipeline za Demo A (koraci 1-7 iz specifikacije).

Pokretanje kao skripta sprema pregled jednog framea kao PNG, tako da se vidi
radi li scena (točke se crtaju sa z-bufferom, bez rasterizacije trokuta):

    python reference/render.py --mesh 1m --frame 13 --png pregled.png
"""

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference import ops  # noqa: E402

W, H = 1280, 720
FRAMES = 100
B8_SAVED_FRAMES = (0, 25, 50, 75, 99)


def model_matrix(f, frames=FRAMES):
    """Rotacija oko osi y za kut 2π·f/frames (desni koordinatni sustav)."""
    t = 2 * np.pi * f / frames
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]], np.float64)


def look_at(eye, target, up):
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array([[*s, -s @ eye],
                     [*u, -u @ eye],
                     [*-f, f @ eye],
                     [0, 0, 0, 1]], np.float64)


def perspective(fovy_deg, aspect, near, far):
    f = 1 / np.tan(np.radians(fovy_deg) / 2)
    return np.array([[f / aspect, 0, 0, 0],
                     [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]], np.float64)


def load_scene(data_dir, dtype=np.float32):
    d = Path(data_dir) / "scene"
    return {
        "V": np.load(d / "V.npy").astype(dtype),
        "P": np.load(d / "P.npy").astype(dtype),
        "light": np.load(d / "light.npy").astype(dtype),
        "albedo": np.load(d / "albedo.npy").astype(dtype),
        "ambient": float(np.load(d / "ambient.npy")[0, 0]),
    }


def load_mesh(data_dir, name, dtype=np.float32):
    d = Path(data_dir) / "mesh"
    return (np.load(d / f"mesh_{name}_positions.npy").astype(dtype),
            np.load(d / f"mesh_{name}_normals.npy").astype(dtype))


def homogeneous(positions):
    return ops.concat_cols(positions, ops.ones(positions.shape[0], 1, positions.dtype))


def frame(scene, positions, normals, f, hom=None):
    """Jedan frame. Vraća (screen N×2, depth N×1, color N×3) u dtypeu ulaza.

    `hom` su unaprijed izračunate homogene koordinate; izvlačenje te
    invarijante izvan petlje frameova je dopuštena optimizacija.
    """
    dt = positions.dtype
    m = model_matrix(f).astype(dt)
    mvp = ops.matmul(ops.matmul(scene["P"], scene["V"]), m)                 # 1
    if hom is None:
        hom = homogeneous(positions)
    clip = ops.matmul(hom, ops.transpose(mvp))                              # 2
    ndc = ops.div(ops.slice_cols(clip, 0, 3), ops.slice_cols(clip, 3, 4))  # 3
    scale = np.array([[W / 2, -H / 2]], dt)
    offset = np.array([[W / 2, H / 2]], dt)
    screen = ops.add(ops.mul(ops.slice_cols(ndc, 0, 2), scale), offset)    # 4
    depth = ops.slice_cols(ndc, 2, 3)
    r = m[0:3, 0:3]
    n = ops.normalize_rows(ops.matmul(normals, ops.transpose(r)))           # 5
    diffuse = ops.relu(ops.matmul(n, ops.transpose(scene["light"])))        # 6
    color = ops.clamp(ops.add(ops.matmul(diffuse, scene["albedo"]),         # 7
                              scene["ambient"]), 0, 1)
    return screen, depth, color


def render_frames(scene, positions, normals, frames=FRAMES, keep=B8_SAVED_FRAMES):
    """B8: svih `frames` frameova; vraća izlaze odabranih frameova složene po redovima."""
    hom = homogeneous(positions)
    kept = []
    for f in range(frames):
        out = frame(scene, positions, normals, f, hom)
        if f in keep:
            kept.append(out)
    return tuple(np.concatenate(parts, axis=0) for parts in zip(*kept))


def write_png(path, img):
    """Minimalni PNG zapis (RGB, 8 bita) bez vanjskih biblioteka."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def preview(screen, depth, color):
    """Crta vrhove kao točke sa z-bufferom (bliže točke prekrivaju dalje)."""
    x = np.floor(screen[:, 0]).astype(np.int64)
    y = np.floor(screen[:, 1]).astype(np.int64)
    z = depth[:, 0]
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (z > -1) & (z < 1)
    order = np.argsort(-z[ok], kind="stable")
    img = np.full((H, W, 3), 24, np.uint8)
    img[y[ok][order], x[ok][order]] = np.round(color[ok][order] * 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser(description="Pregled jednog framea Demoa A kao PNG.")
    ap.add_argument("--data", default=ROOT / "data", type=Path)
    ap.add_argument("--mesh", default="1m", choices=["10k", "100k", "1m"])
    ap.add_argument("--frame", default=13, type=int)
    ap.add_argument("--png", default="pregled.png")
    args = ap.parse_args()
    scene = load_scene(args.data)
    positions, normals = load_mesh(args.data, args.mesh)
    screen, depth, color = frame(scene, positions, normals, args.frame)
    write_png(args.png, preview(screen, depth, color))
    print(f"spremljeno: {args.png}")


if __name__ == "__main__":
    main()
