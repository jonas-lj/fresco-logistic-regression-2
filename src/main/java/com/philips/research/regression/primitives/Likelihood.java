package com.philips.research.regression.primitives;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.real.SReal;
import java.util.List;
import java.util.Vector;

class Likelihood implements Computation<SReal, ProtocolBuilderNumeric> {
    private final DRes<Vector<DRes<SReal>>> xi;
    private final DRes<Vector<DRes<SReal>>> beta;

    Likelihood(DRes<Vector<DRes<SReal>>> xi, DRes<Vector<DRes<SReal>>> beta) {
        this.xi = xi;
        this.beta = beta;
    }

    @Override
    public DRes<SReal> buildComputation(ProtocolBuilderNumeric builder) {
        return builder.seq(seq -> {
            List<DRes<SReal>> xiList = xi.out();
            List<DRes<SReal>> betaList = beta.out();
            if (xiList.size() != betaList.size()) {
                throw new IllegalArgumentException("Likelihood expects vectors of equal size");
            }
            DRes<SReal> prod = seq.realAdvanced().innerProduct(xiList, betaList);
            DRes<SReal> negProd = seq.realNumeric().mult(-1, prod);
            DRes<SReal> exponential = seq.realAdvanced().exp(negProd);
            return seq.realAdvanced().reciprocal(seq.realNumeric().add(1, exponential));
        });
    }

}
