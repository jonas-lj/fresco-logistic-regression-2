package com.philips.research.regression.app;

import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.collections.Matrix;
import dk.alexandra.fresco.lib.real.SReal;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import static com.philips.research.regression.Runner.run;
import static com.philips.research.regression.app.CarDataSet.*;
import static com.philips.research.regression.util.ListAssert.assertEquals;
import static com.philips.research.regression.util.ListConversions.unwrap;
import static com.philips.research.regression.util.MatrixConstruction.matrix;
import static com.philips.research.regression.util.MatrixConversions.transpose;
import static java.math.BigDecimal.valueOf;
import static java.util.Arrays.asList;
import static java.util.Arrays.fill;

@DisplayName("Logistic Regression")
class FitLogisticModelTest {

    private static BigDecimal intercept = valueOf(1.65707);
    private static BigDecimal beta_hp = valueOf(0.00968555 / hp_scale);
    private static BigDecimal beta_wt = valueOf(-1.17481 / wt_scale);

    @Test
    @DisplayName("performs logistic regression")
    void fitsLogisticModel() {
        List<BigDecimal> beta = run(new FitLogisticModelApplication(Xs, Ys, 1.0, 5, null), 2);
        assertEquals(asList(beta_hp, beta_wt, intercept), beta, 0.01);
    }

    @Test
    @DisplayName("performs logistic regression with differential privacy")
    void fitsLogisticModelWithDifferentialPrivacy() {
        BigDecimal privacyBudget = valueOf(1000);
        List<BigDecimal> beta = run(new FitLogisticModelApplication(Xs, Ys, 1.0, 5, privacyBudget), 2);
        assertEquals(asList(beta_hp, beta_wt, intercept), beta, 0.1);
    }

    private static BigDecimal[] ones;
    static {
        ones = new BigDecimal[hp1.length];
        fill(ones, BigDecimal.ONE);
    }

    private static Matrix<BigDecimal> X1 = transpose(matrix(hp1, wt1, ones));
    private static Matrix<BigDecimal> X2 = transpose(matrix(hp2, wt2, ones));

    private static List<Matrix<BigDecimal>> Xs = asList(X1, X2);
    private static List<Vector<BigDecimal>> Ys = asList(am1, am2);
}

class FitLogisticModelApplication implements Application<List<BigDecimal>, ProtocolBuilderNumeric> {

    private List<Matrix<BigDecimal>> Xs;
    private List<Vector<BigDecimal>> Ys;
    private double lambda;
    private int numberOfIterations;
    private BigDecimal privacyBudget;

    FitLogisticModelApplication(List<Matrix<BigDecimal>> Xs, List<Vector<BigDecimal>> Ys, double lambda, int numberOfIterations, BigDecimal privacyBudget) {
        this.Xs = Xs;
        this.Ys = Ys;
        this.lambda = lambda;
        this.numberOfIterations = numberOfIterations;
        this.privacyBudget = privacyBudget;
    }

    @Override
    public DRes<List<BigDecimal>> buildComputation(ProtocolBuilderNumeric builder) {
        return builder.seq(seq -> {
            List<DRes<Matrix<DRes<SReal>>>> closedXs = new ArrayList<>();
            for (int party = 1; party <= Xs.size(); party++) {
                Matrix<BigDecimal> X = Xs.get(party - 1);
                DRes<Matrix<DRes<SReal>>> closedX = seq.realLinAlg().input(X, party);
                closedXs.add(closedX);
            }

            Matrix<BigDecimal> myX = Xs.get(seq.getBasicNumericContext().getMyId() - 1);
            Vector<BigDecimal> myY = Ys.get(seq.getBasicNumericContext().getMyId() - 1);

            DRes<Vector<DRes<SReal>>> result = seq.seq(new FitLogisticModel(closedXs, lambda, numberOfIterations, myX, myY, privacyBudget));
            DRes<Vector<DRes<BigDecimal>>> opened = seq.realLinAlg().openVector(result);

            return () -> unwrap(opened);
        });
    }
}
