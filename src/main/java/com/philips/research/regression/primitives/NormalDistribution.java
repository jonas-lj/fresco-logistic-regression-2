package com.philips.research.regression.primitives;

import static java.math.BigDecimal.valueOf;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.real.RealNumeric;
import dk.alexandra.fresco.lib.real.SReal;

class NormalDistribution {
    static Computation<SReal, ProtocolBuilderNumeric> random() {
        return NormalDistributionIwrinHall.random();
    }
}

class NormalDistributionIwrinHall {
    static Computation<SReal, ProtocolBuilderNumeric> random() {
        return builder -> builder
            .seq(seq -> {
                RealNumeric r = seq.realNumeric();
                DRes<SReal> sum = seq.seq(UniformDistribution.random(valueOf(0), valueOf(1)));
                for (int i = 1; i < 12; ++i) {
                    DRes<SReal> next = seq.seq(UniformDistribution.random(valueOf(0), valueOf(1)));
                    sum = r.add(sum, next);
                }
                DRes<SReal> result = r.sub(sum, valueOf(6));
                return result;
            });
    }
}
