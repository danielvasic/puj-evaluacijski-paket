"""Službena definicija benchmarkova B1-B10: ulazi, izlazi, pragovi i broj ponavljanja.

Isti opis koriste make_golden.py, bench_numpy.py i score.py, pa se nigdje ne
ponavlja. README.md ga opisuje riječima za studente.
"""

# Ponavljanja u JMH: (forkovi, zagrijavanje, mjerenja). "long" vrijedi za spore benchmarkove.
REPS = {"normal": (2, 5, 10), "long": (1, 2, 5)}
# Ponavljanja u NumPyju: (zagrijavanje, mjerenja); jedan proces, bez forkova.
NUMPY_REPS = {"normal": (3, 10), "long": (1, 5)}

B7_MESH, B7_FRAME = "1m", 13
B8_MESH = "100k"

BENCHMARKS = [
    {"id": "B1", "level": "mikro", "what": "matmul 64×64", "reps": "normal",
     "outputs": ["B1"], "precise": 1e-4, "fast": 1e-2},
    {"id": "B2", "level": "mikro", "what": "matmul 512×512", "reps": "normal",
     "outputs": ["B2"], "precise": 1e-4, "fast": 1e-2},
    {"id": "B3", "level": "mikro", "what": "matmul 2048×2048", "reps": "long",
     "outputs": ["B3"], "precise": 1e-4, "fast": 1e-2},
    {"id": "B4", "level": "mikro", "what": "relu(a * b + c), 4096×4096", "reps": "normal",
     "outputs": ["B4"], "precise": 1e-4, "fast": 1e-2},
    {"id": "B5", "level": "mikro", "what": "sum po osi 0 i 1 te max, 4096×4096", "reps": "normal",
     "outputs": ["B5_sum0", "B5_sum1", "B5_max"], "precise": 1e-4, "fast": 1e-2},
    {"id": "B6", "level": "mikro", "what": "normalizeRows, 1 000 000×3", "reps": "normal",
     "outputs": ["B6"], "precise": 1e-4, "fast": 1e-2},
    {"id": "B7", "level": "kernel", "what": "jedan frame Demoa A, 1M vrhova", "reps": "normal",
     "outputs": ["B7_screen", "B7_depth", "B7_color"], "precise": 1e-4, "fast": 1e-3},
    {"id": "B8", "level": "end-to-end", "what": "100 frameova Demoa A, 100k vrhova", "reps": "long",
     "outputs": ["B8_screen", "B8_depth", "B8_color"], "precise": 1e-4, "fast": 1e-3},
    {"id": "B9", "level": "kernel", "what": "inferencija Demoa B, 10 000 slika", "reps": "normal",
     "outputs": ["B9_logits"], "precise": 1e-4, "fast": 1e-2},
    # B10 se ne mjeri greškom nego točnošću na testnom skupu (postotni bodovi od reference).
    {"id": "B10", "level": "end-to-end", "what": "epoha treniranja Demoa B, 60 000 slika", "reps": "long",
     "outputs": ["B10_loss", "B10_accuracy"], "precise": 0.5, "fast": 1.0, "metric": "accuracy"},
]

BY_ID = {b["id"]: b for b in BENCHMARKS}
LEAGUES = ("precise", "fast")
CSV_FIELDS = ["benchmark", "impl", "threads", "league", "median_ms", "p10_ms", "p90_ms", "error"]
