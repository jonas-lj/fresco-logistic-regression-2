package com.philips.research.regression.app;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import com.google.gson.Gson;
import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.Party;
import dk.alexandra.fresco.framework.ProtocolEvaluator;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.builder.numeric.field.BigIntegerFieldDefinition;
import dk.alexandra.fresco.framework.builder.numeric.field.FieldElement;
import dk.alexandra.fresco.framework.configuration.NetworkConfiguration;
import dk.alexandra.fresco.framework.configuration.NetworkConfigurationImpl;
import dk.alexandra.fresco.framework.network.CloseableNetwork;
import dk.alexandra.fresco.framework.network.Network;
import dk.alexandra.fresco.framework.network.socket.SocketNetwork;
import dk.alexandra.fresco.framework.sce.SecureComputationEngine;
import dk.alexandra.fresco.framework.sce.SecureComputationEngineImpl;
import dk.alexandra.fresco.framework.sce.evaluator.BatchEvaluationStrategy;
import dk.alexandra.fresco.framework.sce.evaluator.BatchedProtocolEvaluator;
import dk.alexandra.fresco.framework.sce.evaluator.BatchedStrategy;
import dk.alexandra.fresco.framework.sce.evaluator.EvaluationStrategy;
import dk.alexandra.fresco.framework.util.*;
import dk.alexandra.fresco.framework.value.SInt;
import dk.alexandra.fresco.lib.collections.Matrix;
import dk.alexandra.fresco.logging.BatchEvaluationLoggingDecorator;
import dk.alexandra.fresco.logging.EvaluatorLoggingDecorator;
import dk.alexandra.fresco.logging.NetworkLoggingDecorator;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticProtocolSuite;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticResourcePool;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticResourcePoolImpl;
import dk.alexandra.fresco.suite.spdz.SpdzExponentiationPipeProtocol;
import dk.alexandra.fresco.suite.spdz.SpdzProtocolSuite;
import dk.alexandra.fresco.suite.spdz.SpdzResourcePool;
import dk.alexandra.fresco.suite.spdz.SpdzResourcePoolImpl;
import dk.alexandra.fresco.suite.spdz.datatypes.SpdzSInt;
import dk.alexandra.fresco.suite.spdz.storage.SpdzDataSupplier;
import dk.alexandra.fresco.suite.spdz.storage.SpdzDummyDataSupplier;
import dk.alexandra.fresco.suite.spdz.storage.SpdzMascotDataSupplier;
import dk.alexandra.fresco.suite.spdz.storage.SpdzOpenedValueStoreImpl;
import dk.alexandra.fresco.tools.ot.base.DummyOt;
import dk.alexandra.fresco.tools.ot.base.Ot;
import dk.alexandra.fresco.tools.ot.otextension.RotList;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Option;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.philips.research.regression.util.MatrixConstruction.matrix;
import static com.philips.research.regression.util.MatrixConversions.map;
import static com.philips.research.regression.util.VectorUtils.vectorOf;

@CommandLine.Command(
    description = "Secure Multi-Party Logistic Regression",
    name="LogisticRegression",
    mixinStandardHelpOptions = true,
    version = "Logistic Regression 0.1.0")
public class LogisticRegressionApp implements Callable<Void> {
    @Option(
        names = {"-i", "--myId"},
        required = true,
        description = "Id of this party.")
    private int myId;
    @Option(
        names = {"-p", "--party"},
        required = true,
        split=":",
        description = "Specification of a party. One of these needs to be present for each party. For example: '-p1:localhost:8871 -p2:localhost:8872'.")
    private String[] parties;
    @Option(
        names = {"--lambda"},
        defaultValue = "1.0",
        description = "Lambda for fitting the logistic model. If omitted, the default value is ${DEFAULT-VALUE}.")
    private double lambda;
    @Option(
        names = {"--iterations"},
        defaultValue = "5",
        description = "The number of iterations performed by the fitter. If omitted, the default value is ${DEFAULT-VALUE}.")
    private int iterations;
    @Option(
        names = {"--privacy-budget", "-b"},
        defaultValue = "0",
        description = "Enables using differential privacy for updating the weights given this budget. If omitted, differential privacy is not used."
    )
    private double privacyBudget;
    @Option(
        names = {"--trace"},
        defaultValue = "false",
        description = "Enables trace logging. ⚠️ Warning: exposes secret values in order to log them! ⚠️"
    )
    private boolean traceLogging;
    @Option(
        names = {"--debug"},
        defaultValue = "false",
        description = "Enables debug logging. Ignored when --trace is also passed. ⚠️ Warning: exposes secret values in order to log them! ⚠️"
    )
    private boolean debugLogging;
    @Option(
        names = {"--dummy"},
        defaultValue = "false",
        description = "Evaluates using dummy arithmetic instead of real MPC with SPDZ"
    )
    private boolean dummyArithmetic;
    @Option(
        names = {"--dummy-data-supplier"},
        defaultValue = "false",
        description = "When using SPDZ, use this to use dummy data supplier instead of Mascot data supplier"
    )
    private boolean dummyDataSupplier;
    @Option(
        names = {"--mod-bit-length"},
        defaultValue = "512",
        description = "For experimenting; default is 512"
    )
    private int modBitLength;
    @Option(
        names = {"--max-bit-length"},
        defaultValue = "200",
        description = "For experimenting; default is 200"
    )
    private int maxBitLength;

    public static void main(String[] args) {
        CommandLine.call(new LogisticRegressionApp(), args);
    }

    @Override
    public Void call() throws IOException {
        setLogLevel();

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        Input input = new Gson().fromJson(reader, Input.class);
        Matrix<BigDecimal> m = map(matrix(input.predictors), BigDecimal::valueOf);
        Vector<BigDecimal> v = vectorOf(input.outcomes);

        LogisticRegression frescoApp = new LogisticRegression(myId, m, v, lambda, iterations, privacyBudget);
        ApplicationRunner<List<BigDecimal>> runner = createRunner(myId, createPartyMap());

        List<BigDecimal> result = runner.run(frescoApp);

        System.out.println(result);
        runner.close();
        return null;
    }

    private ApplicationRunner<List<BigDecimal>> createRunner(int myId, HashMap<Integer, Party> partyMap) {
        if (dummyArithmetic) {
            return new DummyRunner<>(myId, partyMap, modBitLength, maxBitLength);
        } else {
            return new SpdzRunner<>(myId, partyMap, dummyDataSupplier, modBitLength, maxBitLength);
        }
    }

    private void setLogLevel() {
        if (traceLogging) {
            Logger logger = (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
            logger.setLevel(Level.ALL);
        } else if (debugLogging){
            Logger logger = (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
            logger.setLevel(Level.DEBUG);
        }
    }

    private class Input {
        Double[][] predictors;
        double[] outcomes;
    }

    private HashMap<Integer, Party> createPartyMap() {
        HashMap<Integer, Party> partyMap = new HashMap<>();
        for (int p = 0; p < parties.length/3; ++p) {
            int partyId = Integer.parseInt(parties[p*3]);
            String host = parties[p*3 + 1];
            int port = Integer.parseInt(parties[p*3 + 2]);
            partyMap.put(partyId, new Party(partyId, host, port));
        }
        return partyMap;
    }
}

abstract class ApplicationRunner <Output> {
    NetworkFactory networkFactory;
    Network network;
    BigInteger modulus;

    ApplicationRunner(int myId, Map<Integer, Party> partyMap, int modBitLength) {
        network = new SocketNetwork(new NetworkConfigurationImpl(myId, partyMap));
        network = new NetworkLoggingDecorator(network);
        networkFactory = new NetworkFactory(partyMap);
        modulus = ModulusFinder.findSuitableModulus(modBitLength);
    }

    abstract Output run(Application<Output, ProtocolBuilderNumeric> application);

    void close() throws IOException {
        networkFactory.close();
        ((Closeable)network).close();
    }
}

class SpdzRunner <Output> extends ApplicationRunner<Output> {

    static final int PRG_SEED_LENGTH = 256;
    private static final int FIXED_POINT_PRECISION = 16;

    private SecureComputationEngineImpl<SpdzResourcePool, ProtocolBuilderNumeric> sce;
    private SpdzResourcePoolImpl resourcePool;
    private SpdzProtocolSuite protocolSuite;

    SpdzRunner(int myId, Map<Integer, Party> partyMap, Boolean dummyDataSupplier, int modBitLength, int maxBitLength) {
        super(myId, partyMap, modBitLength);
        int numberOfPlayers = partyMap.size();

        this.protocolSuite = new SpdzProtocolSuite(maxBitLength, FIXED_POINT_PRECISION);
        BatchEvaluationStrategy<SpdzResourcePool> strategy = EvaluationStrategy.SEQUENTIAL.getStrategy();
        strategy = new BatchEvaluationLoggingDecorator<>(strategy);
        ProtocolEvaluator<SpdzResourcePool> evaluator = new BatchedProtocolEvaluator<>(strategy, protocolSuite);
        evaluator = new EvaluatorLoggingDecorator<>(evaluator);
        sce = new SecureComputationEngineImpl<>(protocolSuite, evaluator);

        SpdzOpenedValueStoreImpl store = new SpdzOpenedValueStoreImpl();
        final BigIntegerFieldDefinition definition = new BigIntegerFieldDefinition(modulus);
        if (!dummyDataSupplier) {
            Drbg drbg = Random.getDrbg(myId);
            List<Integer> partyIds = new ArrayList<>(partyMap.keySet());
            Map<Integer, RotList> seedOts = getSeedOts(myId, partyIds, PRG_SEED_LENGTH, drbg, network);
            FieldElement ssk = SpdzMascotDataSupplier.createRandomSsk(definition, PRG_SEED_LENGTH);
            PreprocessedValuesSupplier preprocessedValuesSupplier
                = new PreprocessedValuesSupplier(myId, numberOfPlayers, networkFactory, protocolSuite, modBitLength, definition, seedOts, ssk, maxBitLength);
            SpdzDataSupplier supplier = SpdzMascotDataSupplier.createSimpleSupplier(
                myId, numberOfPlayers,
                () -> networkFactory.createExtraNetwork(myId),
                modBitLength, definition,
                preprocessedValuesSupplier::provide,
                seedOts, drbg, ssk);
            resourcePool = new SpdzResourcePoolImpl(myId, numberOfPlayers, store, supplier, AesCtrDrbg::new);
        } else {
            SpdzDataSupplier supplier = new SpdzDummyDataSupplier(myId, partyMap.size(), definition,
                new BigInteger(modulus.bitLength(), new java.util.Random(0)).mod(modulus));
            resourcePool = new SpdzResourcePoolImpl(myId, partyMap.size(), store, supplier, AesCtrDrbg::new);
        }
    }

    private Map<Integer, RotList> getSeedOts(int myId, List<Integer> partyIds, int prgSeedLength,
                                             Drbg drbg, Network network) {
        // This method was copied from Fresco AbstractSpdzTest
        Map<Integer, RotList> seedOts = new HashMap<>();
        for (Integer otherId : partyIds) {
            if (myId != otherId) {
                Ot ot = new DummyOt(otherId, network);
                RotList currentSeedOts = new RotList(drbg, prgSeedLength);
                if (myId < otherId) {
                    currentSeedOts.send(ot);
                    currentSeedOts.receive(ot);
                } else {
                    currentSeedOts.receive(ot);
                    currentSeedOts.send(ot);
                }
                seedOts.put(otherId, currentSeedOts);
            }
        }
        return seedOts;
    }


    @Override
    public Output run(Application<Output, ProtocolBuilderNumeric> application) {
        Duration timeout = Duration.ofDays(365);
        return sce.runApplication(application, resourcePool, network, timeout);
    }

    @Override
    public void close() throws IOException {
        super.close();
        sce.shutdownSCE();
    }
}

class DummyRunner <Output> extends ApplicationRunner<Output> {

    private DummyArithmeticResourcePoolImpl resourcePool;
    private SecureComputationEngine<DummyArithmeticResourcePool, ProtocolBuilderNumeric> sce;

    DummyRunner(int myId, Map<Integer, Party> partyMap, int modBitLength, int maxBitLength) {
        super(myId, partyMap, modBitLength);

        final BigIntegerFieldDefinition definition = new BigIntegerFieldDefinition(modulus);
        DummyArithmeticProtocolSuite protocolSuite = new DummyArithmeticProtocolSuite(definition, maxBitLength,16);
        BatchEvaluationStrategy<DummyArithmeticResourcePool> strategy = EvaluationStrategy.SEQUENTIAL.getStrategy();
        ProtocolEvaluator<DummyArithmeticResourcePool> evaluator = new BatchedProtocolEvaluator<>(strategy, protocolSuite);
        sce = new SecureComputationEngineImpl<>(protocolSuite, evaluator);

        resourcePool = new DummyArithmeticResourcePoolImpl(myId, partyMap.size(), definition);
    }

    @Override
    public Output run(Application<Output, ProtocolBuilderNumeric> application) {
        return sce.runApplication(application, resourcePool, network);
    }

    @Override
    public void close() throws IOException {
        super.close();
        sce.shutdownSCE();
    }
}
 class NetworkFactory implements Closeable {

    private static final AtomicInteger PORT_OFFSET_COUNTER = new AtomicInteger(50);
    private static final int PORT_INCREMENT = 10;
    private final List<Closeable> openedNetworks;
     private final Map<Integer, Party> parties;

     public NetworkFactory(Map<Integer, Party> parties) {
        this.parties = parties;
        this.openedNetworks = new ArrayList<>();
    }

    public CloseableNetwork createExtraNetwork(int myId) {
        int portOffset = PORT_OFFSET_COUNTER.addAndGet(PORT_INCREMENT);
        Map<Integer, Party> partiesWithPortOffset = parties.entrySet()
            .stream()
            .peek(e -> e.setValue(applyOffset(portOffset, e.getValue())))
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        NetworkConfiguration config = new NetworkConfigurationImpl(myId, parties);
        CloseableNetwork net = new SocketNetwork(config);
        openedNetworks.add(net);
        return net;
    }

     private Party applyOffset(int portOffset, Party party) {
         return new Party(party.getPartyId(), party.getHostname(), party.getPort() + portOffset);
     }

     @Override
    public void close() {
        openedNetworks.forEach(this::close);
    }

    private void close(Closeable closeable) {
        ExceptionConverter.safe(() -> {
            closeable.close();
            return null;
        }, "IO Exception");
    }
}
