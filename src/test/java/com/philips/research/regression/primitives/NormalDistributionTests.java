package com.philips.research.regression.primitives;

import static com.philips.research.regression.Runner.run;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.real.SReal;
import java.math.BigDecimal;
import java.util.DoubleSummaryStatistics;
import java.util.Vector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Normal Distribution")
class NormalDistributionTests {
    private static DoubleSummaryStatistics stats;

    @BeforeAll
    static void setUp() {
        stats = run(new RandomNormalApplication())
            .stream()
            .map(DRes::out)
            .mapToDouble(BigDecimal::doubleValue)
            .summaryStatistics();
    }

    @Test
    @DisplayName("generates negative numbers")
    void negativeNumbers() {
        assertTrue(stats.getMin() < 0);
    }

    @Test
    @DisplayName("generates positive numbers")
    void positiveNumbers() {
        assertTrue(stats.getMax() > 0);
    }

    @Test
    @DisplayName("has a mean of 0")
    void mean() {
        assertEquals(0, stats.getAverage(), 0.1);
    }
}

class RandomNormalApplication implements Application<Vector<DRes<BigDecimal>>, ProtocolBuilderNumeric> {

    @Override
    public DRes<Vector<DRes<BigDecimal>>> buildComputation(ProtocolBuilderNumeric builder) {
        Vector<DRes<SReal>> randomNumbers = new Vector<>();
        for (int i=0; i<1000; i++) {
            DRes<SReal> random = builder.seq(NormalDistribution.random());
            randomNumbers.add(random);
        }
        return builder.realLinAlg().openVector(() -> randomNumbers);
    }
}
