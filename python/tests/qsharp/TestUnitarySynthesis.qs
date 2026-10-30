// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.UnitarySynthesisTests {
    import Std.Arrays.IndexRange;
    import Std.Math.MaxI;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyPhasePolynomial;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyRealUnitaryViaGivens;
    import QDKChemistry.Utils.UnitarySynthesis.GivensDecomposition;
    import QDKChemistry.Utils.UnitarySynthesis.QuantizeGivensDecomposition;

    /// Returns an operation on [controls | target] that sets each control from `controlBits`,
    /// applies Ry(prepAngles[k]) to target qubit k, and then applies `givens` to the target
    /// controlled on every control.
    function TestMakeGivensOp(
        givens : GivensDecomposition,
        prepAngles : Double[],
        controlBits : Bool[],
        rotationBits : Int,
    ) : Qubit[] => Unit {
        (qs) => {
            let numControls = Length(controlBits);
            let controls = qs[...numControls - 1];
            let target = qs[numControls...];
            for k in IndexRange(controlBits) {
                if controlBits[k] {
                    X(controls[k]);
                }
            }
            for k in IndexRange(prepAngles) {
                Ry(prepAngles[k], target[k]);
            }
            let numAddresses = MaxI(1, 1 <<< (Length(target) - 1));
            use phaseGradient = Qubit[rotationBits];
            use angleReg = Qubit[rotationBits];
            within {
                PreparePhaseGradientState(phaseGradient);
            } apply {
                ApplyRealUnitaryViaGivens(
                    QuantizeGivensDecomposition(givens, numAddresses, rotationBits),
                    controls,
                    target,
                    phaseGradient,
                    angleReg
                );
            }
        }
    }

    /// Applies `givens` to a fresh register twice in one program, first with `firstBits` and
    /// then with `secondBits` rotation bits, each with its own phase gradient.
    operation TestGivensAtTwoPrecisions(
        givens : GivensDecomposition,
        numTarget : Int,
        firstBits : Int,
        secondBits : Int,
    ) : Unit {
        use target = Qubit[numTarget];
        let numAddresses = MaxI(1, 1 <<< (numTarget - 1));
        for bits in [firstBits, secondBits] {
            use phaseGradient = Qubit[bits];
            use angleReg = Qubit[bits];
            within {
                PreparePhaseGradientState(phaseGradient);
            } apply {
                ApplyRealUnitaryViaGivens(
                    QuantizeGivensDecomposition(givens, numAddresses, bits),
                    [],
                    target,
                    phaseGradient,
                    angleReg
                );
            }
        }
        ResetAll(target);
    }

    /// Applies `phases` to `register` controlled on every qubit of `controls`, either through
    /// `Controlled ApplyPhasePolynomial` or, when `padded`, as one uncontrolled polynomial on
    /// `register + controls` whose table is zero unless every control is set.
    operation TestApplyControlledPhasePolynomial(
        phases : Bool[],
        controls : Qubit[],
        register : Qubit[],
        padded : Bool,
    ) : Unit {
        if padded {
            let inactive = Repeated(false, ((1 <<< Length(controls)) - 1) * Length(phases));
            ApplyPhasePolynomial(inactive + phases, register + controls);
        } else {
            Controlled ApplyPhasePolynomial(controls, (phases, register));
        }
    }

    /// Returns `TestApplyControlledPhasePolynomial` as an operation on [register | controls].
    function TestMakeControlledPhasePolynomialOp(phases : Bool[], numRegister : Int, padded : Bool) : Qubit[] => Unit {
        qs => TestApplyControlledPhasePolynomial(phases, qs[numRegister...], qs[...numRegister - 1], padded)
    }

    /// `TestApplyControlledPhasePolynomial` on freshly allocated qubits.
    operation TestControlledPhasePolynomialCircuit(
        phases : Bool[],
        numRegister : Int,
        numControls : Int,
        padded : Bool,
    ) : Unit {
        use register = Qubit[numRegister];
        use controls = Qubit[numControls];
        TestApplyControlledPhasePolynomial(phases, controls, register, padded);
    }
}
