package com.philips.research.regression.primitives;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.collections.Matrix;
import dk.alexandra.fresco.lib.real.SReal;
import java.util.ArrayList;
import java.util.Collections;

public class Hessian implements Computation<Matrix<DRes<SReal>>, ProtocolBuilderNumeric> {

    private DRes<Matrix<DRes<SReal>>> input;

    public Hessian(DRes<Matrix<DRes<SReal>>> input) {
        this.input = input;
    }
//
//    @Override
//    public DRes<Matrix<DRes<SReal>>> buildComputation(ProtocolBuilderNumeric builder) {
//        RealLinearAlgebra real = builder.realLinAlg();
//        return real.scale(valueOf(-0.25), new MultiplyWithTransposed(input).buildComputation(builder));//real.mult(real.transpose(input), input));
//    }
//    
//  private class MultiplyWithTransposed
//      implements Computation<Matrix<DRes<SReal>>, ProtocolBuilderNumeric> {
//
//    private DRes<Matrix<DRes<SReal>>> input;
//
//    public MultiplyWithTransposed(DRes<Matrix<DRes<SReal>>> input) {
//      this.input = input;
//    }

    @Override
    public DRes<Matrix<DRes<SReal>>> buildComputation(ProtocolBuilderNumeric builder) {
      return builder.par(par -> {

        ArrayList<ArrayList<DRes<SReal>>> rows = new ArrayList<>();
        for (int i = 0; i < input.out().getWidth(); i++) {
          ArrayList<DRes<SReal>> row =
              new ArrayList<>(Collections.nCopies(input.out().getWidth(), null));
          rows.add(row);
        }

        for (int i = 0; i < input.out().getWidth(); i++) {
          for (int j = 0; j <= i; j++) {

            final int I = i;
            final int J = j;

            DRes<SReal> c = par.seq(sub -> {
              DRes<SReal> x = sub.realAdvanced().innerProduct(input.out().getColumn(I),
                  input.out().getColumn(J));
              return sub.realNumeric().mult(-.25, x);
            });

            rows.get(i).set(j, c);

            if (i != j) {
              rows.get(j).set(i, c);
            }
          }
        }

        return () -> new Matrix<>(input.out().getWidth(), input.out().getWidth(), rows);
      });
    }
      
//    }
}
