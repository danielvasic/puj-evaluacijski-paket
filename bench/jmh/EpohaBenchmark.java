package primjer.bench; // PRILAGODITE: vaš paket

import java.io.IOException;
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
 * Primjer za B10: svako mjerenje mora krenuti od istih početnih težina.
 *
 * Težine se kopiraju u @Setup(Level.Invocation), koji se ne mjeri.
 * Level.Invocation je prikladan samo za benchmarkove koji traju dugo,
 * zato je B10 u zasebnoj klasi. Tensor i Mlp stoje na mjestu vašeg koda.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "-Xmx4g"})
@Warmup(iterations = 2)
@Measurement(iterations = 5)
public class EpohaBenchmark {

    @Param({"naive", "fast"})
    public String impl;

    @Param({"precise"})
    public String league;

    private Tensor trainX, trainY1h;
    private Mlp pocetna, mreza;

    @Setup(Level.Trial)
    public void ucitaj() throws IOException {
        System.setProperty("tensor.impl", impl);
        trainX = Tensor.loadNpy(MatmulBenchmark.PAKET.resolve("data/mnist/train_x.npy"));
        trainY1h = Tensor.loadNpy(MatmulBenchmark.PAKET.resolve("data/mnist/train_y.npy")).oneHot(10);
        pocetna = Mlp.load(MatmulBenchmark.PAKET.resolve("data/mlp"));
    }

    @Setup(Level.Invocation)
    public void reset() {
        mreza = pocetna.copy();
    }

    @Benchmark
    public Tensor B10_epoha() {
        return mreza.trainEpoch(trainX, trainY1h); // vraća gubitke svakih 100 koraka
    }
}
