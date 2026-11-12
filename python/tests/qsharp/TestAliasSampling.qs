// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.AliasSamplingTests {

    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepare;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepareWithFreeRider;

    /// Test wrapper: conditional alias sampling on `[condition | index | uniform | flag | qrom]`.
    function MakeConditionalAliasSamplingPrepOp(
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

    /// Test wrapper: prepare conditional alias sampling, apply an index phase,
    /// and unprepare to exercise the adjoint on a non-power-of-two table.
    function MakeConditionalAliasSamplingPhaseTestOp(
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

    /// Test wrapper: conditional alias sampling with free-rider data.
    function MakeConditionalAliasSamplingPrepWithFreeRiderOp(
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
