// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.UnitarySynthesisTests {
    import Std.Arrays.IndexRange;
    import Std.Math.MaxI;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyControlledMultiplexedRy;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyPhasePolynomial;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyRealUnitaryViaGivens;
    import QDKChemistry.Utils.UnitarySynthesis.GivensDecomposition;
    import QDKChemistry.Utils.UnitarySynthesis.QuantizeGivensDecomposition;

    /// Returns an operation on [controls | target] that sets the controls, prepares Ry(prepAngles) and applies `givens` or its adjoint.
    function TestMakeGivensOp(
        givens : GivensDecomposition,
        prepAngles : Double[],
        controlBits : Bool[],
        rotationBits : Int,
        inverse : Bool,
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
            let quantized = QuantizeGivensDecomposition(givens, numAddresses, rotationBits, -1);
            within {
                PreparePhaseGradientState(phaseGradient);
            } apply {
                if inverse {
                    Adjoint ApplyRealUnitaryViaGivens(quantized, controls, target, phaseGradient, angleReg);
                } else {
                    ApplyRealUnitaryViaGivens(quantized, controls, target, phaseGradient, angleReg);
                }
            }
        }
    }

    /// Returns `ApplyControlledMultiplexedRy`, or its adjoint, of `words` on [controls | address | target] with its own
    /// phase gradient, passing the first `numOuter` controls through the `Controlled` functor unless `numOuter` < 0.
    function TestMakeMultiplexedRyOp(
        words : Bool[][],
        maxSwapBits : Int,
        numControls : Int,
        numOuter : Int,
        numAddress : Int,
        inverse : Bool
    ) : Qubit[] => Unit {
        (qs) => {
            let bits = Length(words[0]);
            let split = MaxI(0, numOuter);
            let (outer, inner) = (qs[...split - 1], qs[split..numControls - 1]);
            let address = qs[numControls..numControls + numAddress - 1];
            let target = qs[numControls + numAddress];
            use phaseGradient = Qubit[bits];
            use angleReg = Qubit[bits];
            within {
                PreparePhaseGradientState(phaseGradient);
            } apply {
                let args = (words, maxSwapBits, address, inner, target, phaseGradient, angleReg);
                if numOuter < 0 {
                    if inverse {
                        Adjoint ApplyControlledMultiplexedRy(args);
                    } else {
                        ApplyControlledMultiplexedRy(args);
                    }
                } elif inverse {
                    Controlled Adjoint ApplyControlledMultiplexedRy(outer, args);
                } else {
                    Controlled ApplyControlledMultiplexedRy(outer, args);
                }
            }
        }
    }

    /// Returns an operation on [register | controls] that flips the signs of `phases` on the register when every
    /// control is set, via `Controlled ApplyPhasePolynomial` or, when `padded`, one zero-padded polynomial.
    function TestMakeControlledPhasePolynomialOp(phases : Bool[], numRegister : Int, padded : Bool) : Qubit[] => Unit {
        (qs) => {
            let (register, controls) = (qs[...numRegister - 1], qs[numRegister...]);
            if padded {
                let inactive = Repeated(false, ((1 <<< Length(controls)) - 1) * Length(phases));
                ApplyPhasePolynomial(inactive + phases, register + controls);
            } else {
                Controlled ApplyPhasePolynomial(controls, (phases, register));
            }
        }
    }
}
