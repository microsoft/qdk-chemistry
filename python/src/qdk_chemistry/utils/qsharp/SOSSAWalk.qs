// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SOSSA (Sum of Squares with Ancilla) walk operator for DFTHC block encoding.
///
/// Implements the walk operator from arXiv:2502.15882v1 (Low et al. 2025):
///   W = Ref_{a,B} · U† · Ref_B · U
/// where U = OuterPREP · within{InnerPREP} apply{SELECT}.
///
/// For QPE, only the reflections are controlled:
///   c-W = c-Ref_{a,B} · U† · c-Ref_B · U
namespace QDKChemistry.Utils.SOSSAWalk {

    import Std.Arrays.Subarray;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyToEachCA;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Math.PI;
    import Std.StatePreparation.PreparePureStateD;

    /// Parameters for the SOSSA walk step.
    struct SOSSAWalkParams {
        numOrbitals : Int,
        numRanks : Int,
        numBases : Int,
        numCopies : Int,
        numD1 : Int,
        outerStatevector : Double[],
        innerCoefficients : Double[][],
        dqRotationAngles : Double[][],
        sfRotationAngles : Double[][],
        rotationBitPrecision : Int,
        coefficientBitPrecision : Int,
        power : Int,
    }

    /// Reflection about the zero state: 2|0⟩⟨0| - I.
    operation ReflectAboutZero(qs : Qubit[]) : Unit is Adj + Ctl {
        let n = Length(qs);
        if n == 0 {
            // No qubits: global phase (no-op).
        } elif n == 1 {
            Z(qs[0]);
        } else {
            within {
                ApplyToEachCA(X, qs);
            } apply {
                Controlled Z(qs[1...], qs[0]);
            }
            R(PauliI, 2.0 * PI(), qs[0]);
        }
    }

    /// OuterPREP: prepare the outer superposition over x_o ∈ [0, X_o).
    operation OuterPrepare(statevector : Double[], qs : Qubit[]) : Unit is Adj + Ctl {
        PreparePureStateD(statevector, qs);
    }

    /// InnerPREP: conditional preparation over b ∈ [0, B] given x_o.
    /// Simplified simulation-mode implementation using state preparation.
    operation InnerPrepare(
        innerCoefficients : Double[][],
        outerReg : Qubit[],
        innerReg : Qubit[],
    ) : Unit is Adj + Ctl {
        let xo = Length(innerCoefficients);
        for i in 0..xo - 1 {
            ApplyControlledOnInt(
                i,
                PreparePureStateD(innerCoefficients[i], _),
                outerReg,
                innerReg,
            );
        }
    }

    /// SELECT: applies Givens rotations + Majorana controlled on (x_o, b).
    /// Simulation-mode implementation for correctness validation.
    operation Select(
        params : SOSSAWalkParams,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
    ) : Unit is Adj + Ctl {
        let numOrbitals = params.numOrbitals;
        let numD1 = params.numD1;
        let numQ1 = numOrbitals - numD1;
        let numSF = params.numRanks * params.numCopies;

        // Apply Givens rotations conditioned on x_o value.
        // D1 entries: x_o in [0, numD1)
        for xo in 0..numD1 - 1 {
            ApplyControlledOnInt(
                xo,
                ApplyGivensSequence(params.dqRotationAngles[xo], _),
                outerReg,
                systemReg,
            );
        }
        // Q1 entries: x_o in [numD1, numOrbitals)
        for xo in numD1..numOrbitals - 1 {
            ApplyControlledOnInt(
                xo,
                ApplyGivensSequence(params.dqRotationAngles[xo], _),
                outerReg,
                systemReg,
            );
        }
        // SF entries: x_o in [numOrbitals, numOrbitals + numSF)
        // SF angles indexed by r*(B+1)+b
        let numBp1 = params.numBases + 1;
        for xo in numOrbitals..numOrbitals + numSF - 1 {
            let r = (xo - numOrbitals) / params.numCopies;
            for b in 0..numBp1 - 1 {
                let angleIdx = r * numBp1 + b;
                if angleIdx < Length(params.sfRotationAngles) {
                    ApplyControlledOnInt(
                        b,
                        ApplyControlledOnInt(
                            xo,
                            ApplyGivensSequence(params.sfRotationAngles[angleIdx], _),
                            outerReg,
                            _,
                        ),
                        innerReg,
                        systemReg,
                    );
                }
            }
        }

        // Majorana operator on system[0] conditioned on generator type.
        // SF (b < B): Z on system[0] (two-body term)
        // D1: X on system[0], then CZ(spin, system[0])
        // Q1: X on system[0], then CZ(spin, system[0]), then Z(spin)
        for xo in 0..numD1 - 1 {
            ApplyControlledOnInt(xo, MajoranaD1(spinReg, _), outerReg, systemReg[0..0]);
        }
        for xo in numD1..numOrbitals - 1 {
            ApplyControlledOnInt(xo, MajoranaQ1(spinReg, _), outerReg, systemReg[0..0]);
        }
        for xo in numOrbitals..numOrbitals + numSF - 1 {
            ApplyControlledOnInt(xo, MajoranaSF(_), outerReg, systemReg[0..0]);
        }
    }

    /// D1 Majorana: X(sys[0]) · CZ(spin, sys[0]) — annihilation (γ₀).
    operation MajoranaD1(spinReg : Qubit[], target : Qubit[]) : Unit is Adj + Ctl {
        X(target[0]);
        if Length(spinReg) > 0 {
            CZ(spinReg[0], target[0]);
        }
    }

    /// Q1 Majorana: X(sys[0]) · CZ(spin, sys[0]) · Z(spin) — creation (iγ₁).
    operation MajoranaQ1(spinReg : Qubit[], target : Qubit[]) : Unit is Adj + Ctl {
        X(target[0]);
        if Length(spinReg) > 0 {
            CZ(spinReg[0], target[0]);
            Z(spinReg[0]);
        }
    }

    /// SF Majorana: Z(sys[0]) — two-body (b < B identity term).
    operation MajoranaSF(target : Qubit[]) : Unit is Adj + Ctl {
        Z(target[0]);
    }

    /// Apply a sequence of Givens rotations Ry(angle) to target qubits.
    operation ApplyGivensSequence(angles : Double[], target : Qubit[]) : Unit is Adj + Ctl {
        let numAngles = Length(angles);
        let numQubits = Length(target);
        for i in 0..numAngles - 1 {
            if i < numQubits {
                Ry(2.0 * angles[i], target[i]);
            }
        }
    }

    /// Full SOSSA walk step: W = Ref_{a,B} · U† · Ref_B · U.
    /// Controlled: c-Ref_{a,B} · U† · c-Ref_B · U (only reflections controlled).
    operation SOSSAWalkStep(
        params : SOSSAWalkParams,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
    ) : Unit is Adj + Ctl {
        body ... {
            // U: OuterPREP · within{InnerPREP} apply{SELECT}
            OuterPrepare(params.outerStatevector, outerReg);
            within {
                InnerPrepare(params.innerCoefficients, outerReg, innerReg);
            } apply {
                Select(params, outerReg, innerReg, spinReg, systemReg);
            }

            // Ref_B: inner reflection
            ReflectAboutZero(innerReg + spinReg);

            // U†
            Adjoint OuterPrepare(params.outerStatevector, outerReg);
            within {
                InnerPrepare(params.innerCoefficients, outerReg, innerReg);
            } apply {
                Adjoint Select(params, outerReg, innerReg, spinReg, systemReg);
            }

            // Ref_{a,B}: outer reflection
            ReflectAboutZero(outerReg + innerReg + spinReg);
        }
        adjoint auto;
        controlled (ctls, ...) {
            // Only reflections are controlled for QPE.
            // U runs unconditionally.
            OuterPrepare(params.outerStatevector, outerReg);
            within {
                InnerPrepare(params.innerCoefficients, outerReg, innerReg);
            } apply {
                Select(params, outerReg, innerReg, spinReg, systemReg);
            }

            // c-Ref_B
            Controlled ReflectAboutZero(ctls, innerReg + spinReg);

            // U†
            Adjoint OuterPrepare(params.outerStatevector, outerReg);
            within {
                InnerPrepare(params.innerCoefficients, outerReg, innerReg);
            } apply {
                Adjoint Select(params, outerReg, innerReg, spinReg, systemReg);
            }

            // c-Ref_{a,B}
            Controlled ReflectAboutZero(ctls, outerReg + innerReg + spinReg);
        }
        controlled adjoint auto;
    }

    /// Creates a controlled SOSSA walk callable from parameters.
    function MakeControlledSOSSAWalkOp(
        params : SOSSAWalkParams,
    ) : (Qubit, Qubit[]) => Unit {
        let numOrbitals = params.numOrbitals;
        let xo = numOrbitals + params.numRanks * params.numCopies;
        let numOuterQubits = Ceiling(Lg(IntAsDouble(xo)));
        let numInnerQubits = Ceiling(Lg(IntAsDouble(params.numBases + 1)));
        let numSpinQubits = 2;
        let numSystemQubits = 2 * numOrbitals;

        (control, allQubits) => {
            let outerReg = allQubits[0..numOuterQubits - 1];
            let innerReg = allQubits[numOuterQubits..numOuterQubits + numInnerQubits - 1];
            let spinReg = allQubits[numOuterQubits + numInnerQubits..numOuterQubits + numInnerQubits + numSpinQubits - 1];
            let systemReg = allQubits[numOuterQubits + numInnerQubits + numSpinQubits..numOuterQubits + numInnerQubits + numSpinQubits + numSystemQubits - 1];
            for _ in 0..params.power - 1 {
                Controlled SOSSAWalkStep([control], (params, outerReg, innerReg, spinReg, systemReg));
            }
        }
    }

    /// Circuit entry point for controlled SOSSA walk (allocates qubits).
    operation MakeControlledSOSSAWalkCircuit(
        numOrbitals : Int,
        numRanks : Int,
        numBases : Int,
        numCopies : Int,
        numD1 : Int,
        outerStatevector : Double[],
        innerCoefficients : Double[][],
        dqRotationAngles : Double[][],
        sfRotationAngles : Double[][],
        rotationBitPrecision : Int,
        coefficientBitPrecision : Int,
        power : Int,
    ) : Unit {
        let params = new SOSSAWalkParams {
            numOrbitals = numOrbitals,
            numRanks = numRanks,
            numBases = numBases,
            numCopies = numCopies,
            numD1 = numD1,
            outerStatevector = outerStatevector,
            innerCoefficients = innerCoefficients,
            dqRotationAngles = dqRotationAngles,
            sfRotationAngles = sfRotationAngles,
            rotationBitPrecision = rotationBitPrecision,
            coefficientBitPrecision = coefficientBitPrecision,
            power = power,
        };
        let xo = numOrbitals + numRanks * numCopies;
        let numOuterQubits = Ceiling(Lg(IntAsDouble(xo)));
        let numInnerQubits = Ceiling(Lg(IntAsDouble(numBases + 1)));
        let numSpinQubits = 2;
        let numSystemQubits = 2 * numOrbitals;
        let totalAncilla = numOuterQubits + numInnerQubits + numSpinQubits;

        use control = Qubit();
        use allQubits = Qubit[totalAncilla + numSystemQubits];
        let op = MakeControlledSOSSAWalkOp(params);
        op(control, allQubits);
    }
}
