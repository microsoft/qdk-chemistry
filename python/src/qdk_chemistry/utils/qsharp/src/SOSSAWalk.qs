// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SOSSA (Sum of Squares with Ancilla) walk operator for DFTHC block encoding.
///
/// Composable design: each sub-operation (OuterPrepare, InnerPrepare, Select)
/// is built independently as a Q# callable. The walk step composes them with
/// reflections, following the same pattern as PrepSelPrep.
///
/// Walk operator (arXiv:2502.15882v1, Eq. 77):
///   W = Ref_{a,B} · U† · Ref_B · U
/// where U = OuterPREP · within{InnerPREP} apply{SELECT}.
///
/// For QPE, only reflections are controlled:
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

    // ═══════════════════════════════════════════════════════════════════════════
    // Type aliases for composable sub-operations
    // ═══════════════════════════════════════════════════════════════════════════

    /// Outer PREPARE: (outerReg: Qubit[]) => Unit is Adj + Ctl
    /// Inner PREPARE: (outerReg: Qubit[], innerReg: Qubit[]) => Unit is Adj + Ctl
    /// SELECT: (outerReg: Qubit[], innerReg: Qubit[], spinReg: Qubit[], systemReg: Qubit[]) => Unit is Adj + Ctl

    // ═══════════════════════════════════════════════════════════════════════════
    // Outer PREPARE factories
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build an outer PREPARE using alias sampling.
    function MakeOuterPrepareAliasSampling(
        statevector : Double[],
        coefficientBitPrecision : Int,
    ) : (Qubit[]) => Unit is Adj + Ctl {
        // TODO: Replace with actual alias sampling implementation.
        // For now, uses PreparePureStateD as a placeholder.
        qs => PreparePureStateD(statevector, qs)
    }

    /// Build an outer PREPARE using coherent pure-state preparation.
    function MakeOuterPreparePureState(
        statevector : Double[]
    ) : (Qubit[]) => Unit is Adj + Ctl {
        qs => PreparePureStateD(statevector, qs)
    }

    /// Build an outer PREPARE using QROM amplitude loading.
    function MakeOuterPrepareQROM(
        statevector : Double[],
        coefficientBitPrecision : Int,
    ) : (Qubit[]) => Unit is Adj + Ctl {
        // TODO: Replace with QROM-based amplitude loading.
        qs => PreparePureStateD(statevector, qs)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Inner PREPARE factories
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build an inner PREPARE using controlled alias sampling.
    function MakeInnerPrepareAliasSampling(
        innerCoefficients : Double[][],
        coefficientBitPrecision : Int,
    ) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        // TODO: Replace with actual controlled alias sampling.
        (outerReg, innerReg) => {
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
    }

    /// Build an inner PREPARE using direct controlled preparation.
    function MakeInnerPrepareDirect(
        innerCoefficients : Double[][]
    ) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg) => {
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
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SELECT factories
    // ═══════════════════════════════════════════════════════════════════════════

    /// Parameters for SELECT factory functions.
    struct SelectParams {
        numOrbitals : Int,
        numRanks : Int,
        numBases : Int,
        numCopies : Int,
        numD1 : Int,
        dqRotationAngles : Double[][],
        sfRotationAngles : Double[][],
        rotationBitPrecision : Int,
    }

    /// Build a SELECT using QROM + phase gradient rotation.
    function MakeSelectPhaseGradient(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        // TODO: Replace with QROM angle load + phase gradient adder.
        // For now, uses direct Ry rotations as simulation placeholder.
        (outerReg, innerReg, spinReg, systemReg) => {
            SelectImpl(params, outerReg, innerReg, spinReg, systemReg);
        }
    }

    /// Build a SELECT using direct rotation synthesis.
    function MakeSelectDirectRotation(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg, spinReg, systemReg) => {
            SelectImpl(params, outerReg, innerReg, spinReg, systemReg);
        }
    }

    /// Shared SELECT implementation (Givens rotations + Majorana).
    operation SelectImpl(
        params : SelectParams,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
    ) : Unit is Adj + Ctl {
        let numOrbitals = params.numOrbitals;
        let numD1 = params.numD1;
        let numSF = params.numRanks * params.numCopies;

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

        // Majorana operators conditioned on generator type.
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

    // ═══════════════════════════════════════════════════════════════════════════
    // Walk step composer
    // ═══════════════════════════════════════════════════════════════════════════

    /// Compose the SOSSA walk step from pre-built sub-operation callables.
    ///
    /// W = Ref_{a,B} · U† · Ref_B · U
    /// c-W = c-Ref_{a,B} · U† · c-Ref_B · U (only reflections controlled)
    operation SOSSAWalkStep(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
    ) : Unit is Adj + Ctl {
        body ... {
            // U: OuterPREP · within{InnerPREP} apply{SELECT}
            outerPrepareOp(outerReg);
            within {
                innerPrepareOp(outerReg, innerReg);
            } apply {
                selectOp(outerReg, innerReg, spinReg, systemReg);
            }

            // Ref_B: inner reflection
            ReflectAboutZero(innerReg + spinReg);

            // U†
            Adjoint outerPrepareOp(outerReg);
            within {
                innerPrepareOp(outerReg, innerReg);
            } apply {
                Adjoint selectOp(outerReg, innerReg, spinReg, systemReg);
            }

            // Ref_{a,B}: outer reflection
            ReflectAboutZero(outerReg + innerReg + spinReg);
        }
        adjoint auto;
        controlled (ctls, ...) {
            // Only reflections are controlled for QPE.
            outerPrepareOp(outerReg);
            within {
                innerPrepareOp(outerReg, innerReg);
            } apply {
                selectOp(outerReg, innerReg, spinReg, systemReg);
            }

            // c-Ref_B
            Controlled ReflectAboutZero(ctls, innerReg + spinReg);

            // U†
            Adjoint outerPrepareOp(outerReg);
            within {
                innerPrepareOp(outerReg, innerReg);
            } apply {
                Adjoint selectOp(outerReg, innerReg, spinReg, systemReg);
            }

            // c-Ref_{a,B}
            Controlled ReflectAboutZero(ctls, outerReg + innerReg + spinReg);
        }
        controlled adjoint auto;
    }

    /// Creates a controlled SOSSA walk callable from pre-built sub-ops.
    function MakeControlledSOSSAWalkOp(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numOuterQubits : Int,
        numInnerQubits : Int,
        power : Int,
    ) : (Qubit, Qubit[]) => Unit {
        let numSpinQubits = 2;
        (control, allQubits) => {
            let outerReg = allQubits[0..numOuterQubits - 1];
            let innerReg = allQubits[numOuterQubits..numOuterQubits + numInnerQubits - 1];
            let spinReg = allQubits[numOuterQubits + numInnerQubits..numOuterQubits + numInnerQubits + numSpinQubits - 1];
            let systemReg = allQubits[numOuterQubits + numInnerQubits + numSpinQubits..numOuterQubits + numInnerQubits + numSpinQubits + numSystemQubits - 1];
            for _ in 0..power - 1 {
                Controlled SOSSAWalkStep(
                    [control],
                    (outerPrepareOp, innerPrepareOp, selectOp, outerReg, innerReg, spinReg, systemReg),
                );
            }
        }
    }

    /// Circuit entry point: allocates qubits and runs controlled walk.
    operation MakeControlledSOSSAWalkCircuit(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numOuterQubits : Int,
        numInnerQubits : Int,
        power : Int,
    ) : Unit {
        let numSpinQubits = 2;
        let totalAncilla = numOuterQubits + numInnerQubits + numSpinQubits;

        use control = Qubit();
        use allQubits = Qubit[totalAncilla + numSystemQubits];
        let op = MakeControlledSOSSAWalkOp(
            outerPrepareOp,
            innerPrepareOp,
            selectOp,
            numSystemQubits,
            numOuterQubits,
            numInnerQubits,
            power,
        );
        op(control, allQubits);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════════

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

    /// D1 Majorana: X(sys[0]) · CZ(spin, sys[0]).
    operation MajoranaD1(spinReg : Qubit[], target : Qubit[]) : Unit is Adj + Ctl {
        X(target[0]);
        if Length(spinReg) > 0 {
            CZ(spinReg[0], target[0]);
        }
    }

    /// Q1 Majorana: X(sys[0]) · CZ(spin, sys[0]) · Z(spin).
    operation MajoranaQ1(spinReg : Qubit[], target : Qubit[]) : Unit is Adj + Ctl {
        X(target[0]);
        if Length(spinReg) > 0 {
            CZ(spinReg[0], target[0]);
            Z(spinReg[0]);
        }
    }

    /// SF Majorana: Z(sys[0]).
    operation MajoranaSF(target : Qubit[]) : Unit is Adj + Ctl {
        Z(target[0]);
    }
}
