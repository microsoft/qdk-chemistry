namespace QDKChemistry.TestUtils.AliasSamplingTests {
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepare;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepareWithFreeRider;

    function TestMakeConditionalAliasSamplingPrepOp(
        coefficients : Double[][],
        bitsPrecision : Int,
        conditionValue : Int,
        numSwapBits : Int,
    ) : Qubit[] => Unit {
        (qs) => {
            let nCond = Length(coefficients);
            let nCoeffs = Length(coefficients[0]);
            let nIndexBits = Ceiling(Lg(IntAsDouble(nCoeffs)));
            let nCondBits = Ceiling(Lg(IntAsDouble(nCond)));
            let nQromOutput = bitsPrecision + nIndexBits + 2;

            let conditionalReg = qs[0..nCondBits - 1];
            let indexReg = qs[nCondBits..nCondBits + nIndexBits - 1];
            let uniformReg = qs[nCondBits + nIndexBits..nCondBits + nIndexBits + bitsPrecision - 1];
            let flagQubit = qs[nCondBits + nIndexBits + bitsPrecision];
            let qromOut = qs[nCondBits + nIndexBits + bitsPrecision + 1..nCondBits + nIndexBits + bitsPrecision + nQromOutput];

            ApplyXorInPlace(conditionValue, conditionalReg);

            ConditionalAliasSamplingPrepare(
                coefficients,
                bitsPrecision,
                conditionalReg,
                indexReg,
                uniformReg,
                flagQubit,
                qromOut,
                numSwapBits
            );
        }
    }

    function TestMakeConditionalAliasSamplingPhaseOp(
        coefficients : Double[][],
        bitsPrecision : Int,
        conditionValue : Int,
        numSwapBits : Int,
    ) : Qubit[] => Unit {
        (qs) => {
            let nCond = Length(coefficients);
            let nCoeffs = Length(coefficients[0]);
            let nIndexBits = Ceiling(Lg(IntAsDouble(nCoeffs)));
            let nCondBits = Ceiling(Lg(IntAsDouble(nCond)));
            let nQromOutput = bitsPrecision + nIndexBits + 2;

            let conditionalReg = qs[0..nCondBits - 1];
            let indexReg = qs[nCondBits..nCondBits + nIndexBits - 1];
            let uniformReg = qs[nCondBits + nIndexBits..nCondBits + nIndexBits + bitsPrecision - 1];
            let flagQubit = qs[nCondBits + nIndexBits + bitsPrecision];
            let qromOut = qs[nCondBits + nIndexBits + bitsPrecision + 1..nCondBits + nIndexBits + bitsPrecision + nQromOutput];

            ApplyXorInPlace(conditionValue, conditionalReg);

            within {
                ConditionalAliasSamplingPrepare(
                    coefficients,
                    bitsPrecision,
                    conditionalReg,
                    indexReg,
                    uniformReg,
                    flagQubit,
                    qromOut,
                    numSwapBits
                );
            } apply {
                Z(indexReg[0]);
            }
        }
    }

    function TestMakeConditionalAliasSamplingPrepWithFreeRiderOp(
        coefficients : Double[][],
        freeRiderData : Bool[][],
        bitsPrecision : Int,
        conditionValue : Int,
    ) : Qubit[] => Unit {
        (qs) => {
            let nCond = Length(coefficients);
            let nCoeffs = Length(coefficients[0]);
            let nIndexBits = Ceiling(Lg(IntAsDouble(nCoeffs)));
            let nCondBits = Ceiling(Lg(IntAsDouble(nCond)));
            let nFreeRiderBits = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
            let nQromOutput = bitsPrecision + nIndexBits + 2;

            let conditionalReg = qs[0..nCondBits - 1];
            let indexReg = qs[nCondBits..nCondBits + nIndexBits - 1];
            let uniformReg = qs[nCondBits + nIndexBits..nCondBits + nIndexBits + bitsPrecision - 1];
            let flagQubit = qs[nCondBits + nIndexBits + bitsPrecision];
            let qromOut = qs[nCondBits + nIndexBits + bitsPrecision + 1..nCondBits + nIndexBits + bitsPrecision + nQromOutput];
            let freeRiderReg = qs[nCondBits + nIndexBits + bitsPrecision + 1 + nQromOutput..nCondBits + nIndexBits + bitsPrecision + nQromOutput + nFreeRiderBits];

            ApplyXorInPlace(conditionValue, conditionalReg);

            ConditionalAliasSamplingPrepareWithFreeRider(
                coefficients,
                freeRiderData,
                bitsPrecision,
                conditionalReg,
                indexReg,
                uniformReg,
                flagQubit,
                qromOut,
                freeRiderReg,
                0
            );
        }
    }
}
