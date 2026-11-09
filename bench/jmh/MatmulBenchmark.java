package primjer.bench; // PRILAGODITE: vaš paket

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Primjer JMH benchmarka za B1 i B3.
 *
 * Pravila koja čita bench/jmh_to_csv.py:
 *   - ime metode počinje ID-jem benchmarka i podvlakom (B1_..., B3_...)
 *   - @Param impl je naive ili fast, @Param league je precise ili fast
 *   - Mode.AverageTime
 *
 * Tensor, loadNpy i matmul stoje na mjestu vašeg API-ja (PRILAGODITE).
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 2, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "-Xmx4g"})
@Warmup(iterations = 5)
@Measurement(iterations = 10)
public class MatmulBenchmark {

    /** Mapa evaluacijskog paketa, zadaje se s -jvmArgsAppend "-Dpaket=/putanja/do/paketa". */
    static final Path PAKET = Path.of(System.getProperty("paket", "../evaluacijski-paket"));

    @Param({"naive", "fast"})
    public String impl;

    @Param({"precise"})
    public String league;

    private Tensor b1a, b1b, b3a, b3b;

    @Setup(Level.Trial)
    public void ucitaj() throws IOException {
        System.setProperty("tensor.impl", impl); // PRILAGODITE: kako vaša biblioteka bira implementaciju
        b1a = Tensor.loadNpy(PAKET.resolve("data/bench/B1_a.npy"));
        b1b = Tensor.loadNpy(PAKET.resolve("data/bench/B1_b.npy"));
        b3a = Tensor.loadNpy(PAKET.resolve("data/bench/B3_a.npy"));
        b3b = Tensor.loadNpy(PAKET.resolve("data/bench/B3_b.npy"));
    }

    // Rezultat se vraća iz metode, inače bi ga JIT mogao izbaciti kao mrtav kod.
    @Benchmark
    public Tensor B1_matmul64() {
        return b1a.matmul(b1b);
    }

    // Spori benchmarkovi (B3, B8, B10) koriste manje ponavljanja.
    @Benchmark
    @Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "-Xmx4g"})
    @Warmup(iterations = 2)
    @Measurement(iterations = 5)
    public Tensor B3_matmul2048() {
        return b3a.matmul(b3b);
    }
}
