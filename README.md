# Evaluacijski paket: brza numerička biblioteka u Javi

Paket daje svim parovima iste ulazne podatke, očekivane izlaze, testove ispravnosti i bodovanje. Uz njega ide specifikacija zadatka; ovdje su točne definicije koje specifikacija ne navodi.

## Sadržaj

```
data/generate_data.py      generira sve ulaze (fiksni seed) i data/MANIFEST.txt
golden/make_golden.py      očekivani izlazi B1–B10, računani u float64
conformance/               185 testova ispravnosti API-ja (cases.tsv, cases.json, cases/)
conformance/check.py       provjera vaših izlaza testova
reference/                 NumPy implementacija API-ja, Demoa A i Demoa B
bench/bench_numpy.py       NumPy strana benchmarkova -> results/numpy_<dretve>.csv
bench/jmh/                 primjer JMH benchmarka i dijelovi pom.xml-a
bench/jmh_to_csv.py        JMH JSON -> results/java_<dretve>.csv
score.py                   greške, bodovi i grafovi -> izvjestaj/
```

Potrebni su Python 3.10+ i `pip install -r requirements.txt`.

## Priprema podataka

```bash
python data/generate_data.py --download
python golden/make_golden.py
python conformance/check.py --reference
```

Prva naredba skida MNIST (oko 11 MB) u `data/raw/` i generira oko 510 MB podataka. Ako MNIST već imate, umjesto `--download` koristite `--mnist-dir <mapa>`.

Provjera da imate iste podatke kao nastavnik:

```bash
cd data && grep -v '^#' MANIFEST.txt | shasum -a 256 -c
```

Manifest je generiran na referentnom računalu. Na drugom procesoru ili s drugom verzijom NumPyja neke datoteke mogu se razlikovati u zadnjem bitu (npr. zbog `sin` i `cos` kod modela), što za vaš rad nije važno. Službeno se boduje s podacima s referentnog računala.

## Konvencije

- Sve datoteke su `.npy` verzije 1.0: `float32`, little-endian, redoslijed po redovima, uvijek 2D.
- Cijeli brojevi (oznake MNIST-a, indeksi trokuta, rezultat `argmax`) zapisani su kao `float32` i točni su do 2^24.
- Skalarni rezultat (ukupna suma, `max`, `dot`) ima oblik (1, 1). Redukcija po osi 0 daje (1, N), a po osi 1 daje (M, 1).
- `rand(seed)` se ne uspoređuje s NumPyjem, jer generatori slučajnih brojeva nisu isti.
- Greška je `e = max|x − ref| / max|ref|`, a pogrešan oblik ili NaN znače pad.

## Testovi ispravnosti

`conformance/cases.tsv` ima jedan test po retku, s pet stupaca odvojenih tabulatorom:

```
id    op    ulazi (zarezom, ili -)    argumenti (k=v;k=v, ili -)    očekivani izlaz
```

Primjer retka: `clamp_7x5	clamp	cases/clamp_7x5__in0.npy	lo=-0.3;hi=0.6	cases/clamp_7x5__expected.npy`. Putanje su relativne na `conformance/`.

Argumenti po operaciji: `scalar` (tenzor i skalar, `gt`), `lo` i `hi` (`clamp`), `axis` (redukcije; bez njega je redukcija ukupna), `m` i `n` (`reshape`, `zeros`, `ones`, `full`), `value` (`full`), `start` i `end` (`sliceRows`, `sliceCols`; `end` nije uključen), `classes` (`oneHot`).

Testove možete pretvoriti u JUnit testove ili za svaki test spremiti izlaz kao `<id>.npy` u jednu mapu i pokrenuti:

```bash
python conformance/check.py moji_izlazi/
```

Prag je 1e-4, isti kao za preciznu ligu.

## Ulazni podaci

| Datoteka | Oblik | Sadržaj |
| --- | --- | --- |
| `data/bench/B1_a.npy`, `B1_b.npy` | 64×64 | uniformno [−1, 1] |
| `data/bench/B2_a.npy`, `B2_b.npy` | 512×512 | uniformno [−1, 1] |
| `data/bench/B3_a.npy`, `B3_b.npy` | 2048×2048 | uniformno [−1, 1] |
| `data/bench/B4_a.npy`, `B4_b.npy`, `B4_c.npy` | 4096×4096 | uniformno [−1, 1] |
| `data/bench/B6_v.npy` | 1 000 000×3 | normalna razdioba |
| `data/scene/V.npy`, `P.npy` | 4×4 | kamera i projekcija |
| `data/scene/light.npy`, `albedo.npy` | 1×3 | smjer prema svjetlu (normaliziran), boja |
| `data/scene/ambient.npy` | 1×1 | 0.12 |
| `data/mesh/mesh_<v>_positions.npy`, `_normals.npy` | N×3 | v je `10k`, `100k` ili `1m` |
| `data/mesh/mesh_<v>_triangles.npy` | T×3 | indeksi vrhova, CCW gledano izvana |
| `data/mnist/train_x.npy`, `test_x.npy` | 60 000×784, 10 000×784 | pikseli skalirani na [0, 1] |
| `data/mnist/train_y.npy`, `test_y.npy` | 60 000×1, 10 000×1 | oznake 0–9 |
| `data/mlp/W1.npy`, `b1.npy`, `W2.npy`, `b2.npy` | 784×256, 1×256, 256×10, 1×10 | početne težine (He), biasi nula |

## Demo A: detalji

Kamera i projekcija slijede OpenGL konvencije: kamera gleda prema −z, a NDC je kocka [−1, 1]. Matrica modela za frame f je rotacija oko osi y za kut t = 2π·f/100, izračunata u `double` i pretvorena u `float`:

```
M = |  cos t   0   sin t   0 |
    |    0     1     0     0 |
    | -sin t   0   cos t   0 |
    |    0     0     0     1 |
```

`R` je gornji lijevi 3×3 blok od `M`. Ekran je 1280×720, pa je `screen = ndc[:, 0:2] * [640, -360] + [640, 360]`. Model je u svih 100 frameova cijeli na ekranu, a `depth` je između 0.78 i 0.92.

Homogene koordinate (`concatCols(positions, ones)`) smijete izračunati jednom, izvan petlje frameova. NumPy referenca u B8 radi isto.

Pregled jednog framea kao PNG: `python reference/render.py --mesh 1m --frame 13 --png pregled.png`.

## Demo B: detalji

- `softmaxRows` mora biti numerički stabilan: od svakog reda oduzmite njegov maksimum prije `exp`.
- Gubitak je `-mean(sum(Y * log(clamp(p, 1e-7, 1)), os 1))`. Bez `clamp` bi `0 · log(0)` dao NaN.
- Gubitak se računa na koracima 0, 100, …, 900, prije ažuriranja težina u tom koraku.
- 60 000 / 64 daje 937 punih batcheva; zadnjih 32 primjera se preskače.
- Referenca nakon jedne epohe postiže točnost 93.15 % na testnom skupu (float64), odnosno 93.14 % u float32.

## Benchmarkovi B1–B10

Mjeri se samo izračun: učitavanje datoteka i priprema (npr. `oneHot` oznaka u B10) su izvan mjerenja.

| ID | Ulaz | Mjereni izračun | Izlazne datoteke |
| --- | --- | --- | --- |
| B1 | `B1_a`, `B1_b` | `matmul(a, b)` | `B1` |
| B2 | `B2_a`, `B2_b` | `matmul(a, b)` | `B2` |
| B3 | `B3_a`, `B3_b` | `matmul(a, b)` | `B3` |
| B4 | `B4_a`, `B4_b`, `B4_c` | `relu(a * b + c)` | `B4` |
| B5 | `B4_a` | `sum(a, 0)`, `sum(a, 1)`, `max(a)` | `B5_sum0`, `B5_sum1`, `B5_max` |
| B6 | `B6_v` | `normalizeRows(v)` | `B6` |
| B7 | model `1m`, frame 13 | koraci 1–7 Demoa A | `B7_screen`, `B7_depth`, `B7_color` |
| B8 | model `100k`, frameovi 0–99 | koraci 1–7 za svih 100 frameova | `B8_screen`, `B8_depth`, `B8_color` |
| B9 | `test_x`, početne težine | logiti cijelog testnog skupa u jednom prolazu | `B9_logits` |
| B10 | `train_x`, `train_y`, početne težine | 937 koraka treniranja | `B10_loss` (10×1), `B10_accuracy` (1×1, postotak) |

B8 sprema samo frameove 0, 25, 50, 75 i 99, složene po redovima tim redom (5N redova). Točnost u B10 računa se nakon epohe, izvan mjerenja.

Pragovi grešaka i broj ponavljanja nalaze se u `bench/spec.py`. B3, B8 i B10 mjere se s manje ponavljanja (JMH: 1 fork, 2 zagrijavanja, 5 mjerenja), ostali s 2 forka, 5 zagrijavanja i 10 mjerenja.

## Mjerenje u Javi

Primjer je u `bench/jmh/` (`MatmulBenchmark.java`, `EpohaBenchmark.java`, `pom-dodatak.xml`). `jmh_to_csv.py` traži tri stvari:

- ime metode počinje ID-jem i podvlakom: `B1_matmul64`, `B10_epoha`
- `@Param impl` je `naive` ili `fast`, a `@Param league` je `precise` ili `fast`
- `Mode.AverageTime`

```bash
java -jar bench/target/benchmarks.jar -rf json -rff results/jmh_all.json
java -jar bench/target/benchmarks.jar -jvmArgsAppend "-Dtensor.threads=1" -rf json -rff results/jmh_1.json
python bench/jmh_to_csv.py results/jmh_all.json --threads all
python bench/jmh_to_csv.py results/jmh_1.json --threads 1
```

Izlaze za provjeru greške spremite iz zasebnog programa (ne iz JMH-a) u `out/<impl>_<league>/<ime>.npy`, npr. `out/fast_precise/B3.npy` ili `out/naive_precise/B10_accuracy.npy`.

NumPy strana: `python bench/bench_numpy.py --threads all` i `--threads 1`.

## Bodovanje

```bash
python score.py --results results/*.csv --outputs out/ --memory memory.json
```

`score.py` piše `izvjestaj/izvjestaj.md` i `izvjestaj/grafovi.png` te računa:

- **Ubrzanje naspram baselinea (15):** geometrijska sredina t_naive / t_fast u preciznoj ligi. Benchmark koji ne prolazi ili nije izmjeren ulazi kao 1×.
- **Usporedba s NumPyjem (10):** R = t_NumPy / t_fast po benchmarku; R ≥ 1 donosi 1 bod, a 0.5 ≤ R < 1 donosi 0.5 boda.
- **Memorija (5):** iz `memory.json` (primjer je `memory.example.json`).

Brzu ligu nastavnik boduje ručno iz dnevnika; izvještaj samo navodi na kojim benchmarkovima brza liga prolazi.

## Memorijski izazov

Program množi dvije matrice 8192×8192 u `float32`. Mora završiti unutar 180 s, s najviše 1536 MB memorije procesa:

```bash
/usr/bin/time -l java -cp ... MemorijskiIzazov     # macOS: "maximum resident set size" u bajtovima
/usr/bin/time -v java -cp ... MemorijskiIzazov     # Linux: "Maximum resident set size" u kB
```

Vrijeme i memoriju upišite u `memory.json`.

## Referentno računalo

Službeno se mjeri na jednom Linux računalu, s NumPyjem koji koristi OpenBLAS (standardni `pip install numpy` na Linuxu). `bench_numpy.py` ispisuje BLAS backend i upozorava kad mjerenje nije na takvom sustavu.

Na Macu s Apple Silicon NumPy koristi Accelerate, koji matmul računa na posebnom matričnom hardveru. Tamo je NumPy na B2, B3 i B9 mnogo brži nego na referentnom računalu, pa vlastita mjerenja na Macu nisu mjerilo bodova. Orijentacijski, NumPy 2.4 s Accelerate na Apple M2 (8 jezgri), sve dretve, medijan u ms:

| B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.002 | 0.20 | 14.2 | 12.5 | 5.6 | 10.9 | 53 | 483 | 5.7 | 172 |
