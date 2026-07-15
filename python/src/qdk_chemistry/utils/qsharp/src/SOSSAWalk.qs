// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Sum of Squares Spectral Amplification (SOSSA) walk operator.
///
/// Composable design: each sub-operation (OuterPrepare, InnerPrepare, Select)
/// is built independently as a Q# callable. The walk step composes them with
/// reflections.
///
/// Walk operator (arXiv:2502.15882v1, Eq. 77):
///   W = Ref_{a,B} · U† · Ref_B · U
/// where U = OuterPREP · within{InnerPREP} apply{SELECT}.
///
/// For QPE, we need controlled walk operators, where reflections are
/// controlled (arXiv:1805.03662, fig 1)
///   c-W = c-Ref_{a,B} · U† · c-Ref_B · U
namespace QDKChemistry.Utils.SOSSAWalk {

    import Std.Arrays.Padded;
    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyToEachCA;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Math.PI;
    import Std.Math.Round;
    import Std.StatePreparation.PreparePureStateD;
    import Std.TableLookup.Select;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import Std.ResourceEstimation.RepeatEstimates;
    import Std.ResourceEstimation.SingleVariant;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepareWithFreeRider;
    import Std.Arithmetic.RippleCarryCGIncByLE;
    import QDKChemistry.Utils.PhaseGradient.MakePhaseGradientAncillaPrep;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState, QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;


    // ═══════════════════════════════════════════════════════════════════════════
    // Inner PREPARE factories
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build an inner PREPARE using conditional alias sampling (2D QROM).
    ///
    /// Uses ConditionalAliasSamplingPrepareWithFreeRider to prepare:
    ///   |x_o⟩|0⟩ → |x_o⟩ Σ_b √(p̃_{x_o,b}) e^{iπ·sign} |b⟩|freeRider⟩|garbage⟩
    ///
    /// The returned callable expects:
    ///   outerReg — conditional address register (x_o)
    ///   innerReg — target register layout: indexReg[nIdx] + uniformReg[μ]
    ///              + flagQubit[1] + qromOutput[μ + nIdx + 2] + freeRiderReg[nFR]
    function MakeInnerPrepareAliasSampling(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
    ) : (Qubit[], Qubit[]) => Unit is Adj {
        let nCoeffs = Length(innerCoefficients[0]);
        let nIndexBits = Ceiling(Lg(IntAsDouble(nCoeffs)));
        let mu = coefficientBitPrecision;
        let nFreeRider = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        let qromEnd = 2 * nIndexBits + 2 * mu + 2;
        // innerReg layout: indexReg[nIdx] + uniformReg[mu] + flag[1]
        //   + qromOut[mu + nIdx + 2] + freeRiderReg[nFR]
        (outerReg, innerReg) => {
            let indexReg = innerReg[0..nIndexBits - 1];
            let uniformReg = innerReg[nIndexBits..nIndexBits + mu - 1];
            let flagQubit = innerReg[nIndexBits + mu];
            let qromOut = innerReg[nIndexBits + mu + 1..qromEnd];
            let freeRiderReg = if nFreeRider > 0 {
                innerReg[qromEnd + 1..qromEnd + nFreeRider]
            } else {
                []
            };
            ConditionalAliasSamplingPrepareWithFreeRider(
                innerCoefficients,
                freeRiderData,
                mu,
                outerReg,
                indexReg,
                uniformReg,
                flagQubit,
                qromOut,
                freeRiderReg, -1
            );
        }
    }

    /// Build an inner PREPARE using direct controlled preparation.
    /// Loads free-rider data (G, r) via Select QROM and prepares b superposition
    /// via controlled PreparePureStateD.
    function MakeInnerPrepareDirect(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][]
    ) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        let nCoeffs = Length(innerCoefficients[0]);
        let nIndexBits = Ceiling(Lg(IntAsDouble(if nCoeffs > 1 { nCoeffs } else { 2 })));
        let nFreeRider = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        // innerReg layout: bReg[nIndexBits] + freeRiderReg[nFR]
        (outerReg, innerReg) => {
            let bReg = innerReg[0..nIndexBits - 1];
            let freeRiderReg = if nFreeRider > 0 {
                innerReg[nIndexBits..nIndexBits + nFreeRider - 1]
            } else {
                []
            };

            // Step 1: Load free-rider data (G, r) via QROM indexed by x_o.
            if nFreeRider > 0 {
                Select(freeRiderData, outerReg, freeRiderReg);
            }

            // Step 2: Controlled PreparePureStateD on b register, indexed by x_o.
            // Only needed for SF terms where there's a nontrivial b-superposition.
            let xo = Length(innerCoefficients);
            for i in 0..xo - 1 {
                let nPadded = 1 <<< nIndexBits;
                let paddedAmps = Padded(-nPadded, 0.0, innerCoefficients[i]);
                ApplyControlledOnInt(
                    i,
                    PreparePureStateD(paddedAmps, _),
                    outerReg,
                    Reversed(bReg),
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
        numPositiveOneBody : Int,
        OneBodyRotationAngles : Double[][],
        TwoBodyRotationAngles : Double[][],
        rotationBitPrecision : Int,
        /// Number of free-rider bits at the end of innerReg loaded by inner PREPARE QROM.
        /// Layout: [sf_vs_dq(1), d_vs_q(1), r_bits(⌈log₂ R⌉)].
        /// When > 0, SelectImpl reads isSF and dvsq from innerReg instead of computing them.
        numFreeRiderBits : Int,
    }

    /// Build a SELECT using QROM + phase gradient rotation.
    ///
    /// Givens rotations are applied via split OneBody/TwoBody Select QROM + RyViaPhaseGradient.
    /// This is the production implementation for fault-tolerant resource estimation.
    /// The phase gradient register is allocated and prepared externally by QPE.
    function MakeSelectPhaseGradient(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg, spinReg, systemReg, phaseGradientReg) => {
            SelectImpl(params, true, outerReg, innerReg, spinReg, systemReg, phaseGradientReg);
        }
    }

    /// Build a SELECT using direct rotation synthesis.
    ///
    /// Givens rotations are applied via multi-controlled Ry gates.
    /// Useful for simulation and testing (no ancilla overhead).
    /// The phaseGradientReg argument is accepted but ignored.
    function MakeSelectDirectRotation(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg, spinReg, systemReg, phaseGradientReg) => {
            SelectImpl(params, false, outerReg, innerReg, spinReg, systemReg, phaseGradientReg);
        }
    }

    /// SELECT implementation (arXiv:2502.15882v1, Appendix B.3, B.5-B.6).
    ///
    /// Implements: within{SelectSpins} apply{ within{GivensRotations} apply{MajoranaOp} }
    ///
    /// isSF and dvsq are read from the free-rider register at the end of innerReg,
    /// loaded by inner PREPARE's conditional alias sampling QROM.
    /// spinDQ and spinSF are initialized during outer and inner preparation steps.
    ///
    /// # Parameters
    /// ## usePhaseGradient
    /// When true, uses QROM + phase gradient for Givens rotations (production).
    /// When false, uses direct controlled-Ry gates (simulation/testing).
    ///
    /// Register layout:
    ///   outerReg:  [xoReg (xoBits)]
    ///   innerReg:  [bReg (bBits)] [alias garbage...] [freeRider: isSF(1) + dvsq(1) + rBits(...)]
    ///   spinReg:   [spinDQ (1)] [spinSF (1)]
    ///   systemReg: [sysDown (N)] [sysUp (N)]
    ///   phaseGradientReg: [bRot qubits] (prepared externally; empty for direct mode)
    operation SelectImpl(
        params : SelectParams,
        usePhaseGradient : Bool,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj + Ctl {
        let N = params.numOrbitals;
        let numSF = params.numRanks * params.numCopies;
        let Xo = N + numSF;
        let xoBits = Ceiling(Lg(IntAsDouble(if Xo > 1 { Xo } else { 2 })));
        let numBp1 = params.numBases + 1;
        let bBits = Ceiling(Lg(IntAsDouble(if numBp1 > 1 { numBp1 } else { 2 })));
        let numRotAngles = N - 1;

        // Register slicing
        let xoReg = outerReg[0..xoBits - 1];
        let spinDQ = spinReg[0];
        let spinSF = spinReg[1];
        let bReg = innerReg[0..bBits - 1];
        let sysRegDown = systemReg[0..N - 1];
        let sysRegUp = systemReg[N..2 * N - 1];

        // Free-rider data from inner PREPARE QROM: [sf_vs_dq(1), d_vs_q(1), r_bits...]
        // Located at the end of innerReg.
        let nInner = Length(innerReg);
        let nFR = params.numFreeRiderBits;
        let isSF = innerReg[nInner - nFR];       // sf_vs_dq
        let dvsq = innerReg[nInner - nFR + 1];   // d_vs_q

        // Allocate spin ancilla and bEqB flag
        use spin = Qubit();
        use bEqBQubit = Qubit();

        within {
            SelectSpins(isSF, spinDQ, spinSF, spin, sysRegDown, sysRegUp);
        } apply {
            within {
                // Givens rotations: basis change to localize amplitude on qubit 0
                if usePhaseGradient {
                    let rBits = if nFR > 2 { innerReg[nInner - nFR + 2..nInner - 1] } else { [] };
                    ApplyGivensRotationsQROM(
                        params,
                        N,
                        numSF,
                        numBp1,
                        numRotAngles,
                        xoBits,
                        isSF,
                        xoReg,
                        bReg,
                        rBits,
                        sysRegDown,
                        phaseGradientReg,
                        bEqBQubit
                    );
                } else {
                    ApplyMultiControlledRotations(
                        params,
                        N,
                        numSF,
                        numBp1,
                        numRotAngles,
                        xoBits,
                        isSF,
                        xoReg,
                        bReg,
                        sysRegDown,
                        bEqBQubit
                    );
                }
            } apply {
                // Majorana operator (Fig. 4 / Appendix B.6)
                MajoranaOp(isSF, dvsq, bEqBQubit, spin, sysRegDown[0]);
                within { X(isSF); } apply {
                    Controlled Z([isSF], spinSF);
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Walk step composer
    // ═══════════════════════════════════════════════════════════════════════════

    /// Compose the SOSSA walk step from pre-built sub-operation callables.
    ///
    /// W = Ref_{a,B} · U† · Ref_B · U
    /// c-W = c-Ref_{a,B} · U† · c-Ref_B · U (only reflections controlled)
    ///
    /// ## Parameters
    /// ## numReflectInner
    /// Number of qubits from innerReg to include in reflections (typically bBits,
    /// excluding freeRider). Qubits beyond this are always |0⟩ at reflection time
    /// and are excluded to save Toffoli cost.
    operation SOSSAWalkStep(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numReflectInner : Int,
        numOuterIndexQubits : Int,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj + Ctl {
        body ... {
            // outerIndexReg: logical index qubits used by innerPrep and SELECT addressing.
            // For alias sampling, outerReg has extra garbage qubits beyond the index.
            let outerIndexReg = outerReg[0..numOuterIndexQubits - 1];

            // U: OuterPREP · H(spinDQ) · within{InnerPREP + H(spinSF)} apply{SELECT}
            outerPrepareOp(outerReg);
            H(spinReg[0]);
            within {
                innerPrepareOp(outerIndexReg, innerReg);
                H(spinReg[1]); // H(spinSF): part of inner prep for reflection
            } apply {
                selectOp(outerIndexReg, innerReg, spinReg, systemReg, phaseGradientReg);
            }

            // Ref_B: inner reflection on bReg + spinSF (excludes freeRider)
            Reflect(innerReg[0..numReflectInner - 1] + [spinReg[1]]);

            // U†: undo inner BE first, then outer
            within {
                innerPrepareOp(outerIndexReg, innerReg);
                H(spinReg[1]); // H(spinSF)
            } apply {
                Adjoint selectOp(outerIndexReg, innerReg, spinReg, systemReg, phaseGradientReg);
            }
            H(spinReg[0]);
            Adjoint outerPrepareOp(outerReg);

            // Ref_{a,B}: outer reflection on outerReg + bReg + spinReg
            Reflect(outerReg + innerReg[0..numReflectInner - 1] + spinReg);
        }
        adjoint auto;
        controlled (ctls, ...) {
            if BeginEstimateCaching("Controlled_SOSSAWalkStep", SingleVariant()) {
                let outerIndexReg = outerReg[0..numOuterIndexQubits - 1];

                // Only reflections are controlled for QPE.
                // U (uncontrolled)
                outerPrepareOp(outerReg);
                H(spinReg[0]);
                within {
                    innerPrepareOp(outerIndexReg, innerReg);
                    H(spinReg[1]); // H(spinSF)
                } apply {
                    selectOp(outerIndexReg, innerReg, spinReg, systemReg, phaseGradientReg);
                }

                // c-Ref_B
                Controlled Reflect(ctls, innerReg[0..numReflectInner - 1] + [spinReg[1]]);

                // U† (uncontrolled): inner BE adjoint first, then outer
                within {
                    innerPrepareOp(outerIndexReg, innerReg);
                    H(spinReg[1]); // H(spinSF)
                } apply {
                    Adjoint selectOp(outerIndexReg, innerReg, spinReg, systemReg, phaseGradientReg);
                }
                H(spinReg[0]);
                Adjoint outerPrepareOp(outerReg);

                // c-Ref_{a,B}
                Controlled Reflect(ctls, outerReg + innerReg[0..numReflectInner - 1] + spinReg);
                EndEstimateCaching();
            }
        }
        controlled adjoint auto;
    }

    /// Creates a controlled SOSSA walk callable from pre-built sub-ops.
    ///
    /// Register layout expected by QPE: allQubits = [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    /// QPE passes (control, systems + beAncillas) where systems come first.
    function MakeControlledSOSSAWalkOp(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numOuterQubits : Int,
        numOuterIndexQubits : Int,
        numInnerQubits : Int,
        numReflectInner : Int,
        numPhaseGradientQubits : Int,
        power : Int,
    ) : (Qubit, Qubit[]) => Unit {
        let numSpinQubits = 2; // spinReg = [spinDQ, spinSF]
        (control, allQubits) => {
            // QPE convention: system qubits first, then block-encoding ancillas
            let systemReg = allQubits[0..numSystemQubits - 1];
            let outerReg = allQubits[numSystemQubits..numSystemQubits + numOuterQubits - 1];
            let innerReg = allQubits[numSystemQubits + numOuterQubits..numSystemQubits + numOuterQubits + numInnerQubits - 1];
            let spinReg = allQubits[numSystemQubits + numOuterQubits + numInnerQubits..numSystemQubits + numOuterQubits + numInnerQubits + numSpinQubits - 1];
            let phaseGradientReg = if numPhaseGradientQubits > 0 {
                allQubits[numSystemQubits + numOuterQubits + numInnerQubits + numSpinQubits..numSystemQubits + numOuterQubits + numInnerQubits + numSpinQubits + numPhaseGradientQubits - 1]
            } else {
                []
            };
            for _ in 0..power - 1 {
                if BeginEstimateCaching("Controlled_SOSSAWalkOp", SingleVariant()) {
                    Controlled SOSSAWalkStep(
                        [control],
                        (outerPrepareOp, innerPrepareOp, selectOp, numReflectInner, numOuterIndexQubits, outerReg, innerReg, spinReg, systemReg, phaseGradientReg),
                    );
                    EndEstimateCaching();
                }
            }
        }
    }

    /// Circuit entry point: allocates qubits and runs controlled walk.
    /// Register layout matches QPE convention: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    operation MakeControlledSOSSAWalkCircuit(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numOuterQubits : Int,
        numOuterIndexQubits : Int,
        numInnerQubits : Int,
        numReflectInner : Int,
        numPhaseGradientQubits : Int,
        power : Int,
    ) : Unit {
        let numSpinQubits = 2; // spinReg = [spinDQ, spinSF]
        let totalAncilla = numOuterQubits + numInnerQubits + numSpinQubits + numPhaseGradientQubits;

        use control = Qubit();
        use allQubits = Qubit[numSystemQubits + totalAncilla];
        let op = MakeControlledSOSSAWalkOp(
            outerPrepareOp,
            innerPrepareOp,
            selectOp,
            numSystemQubits,
            numOuterQubits,
            numOuterIndexQubits,
            numInnerQubits,
            numReflectInner,
            numPhaseGradientQubits,
            power,
        );
        op(control, allQubits);
    }

    /// Resource-estimation entry point: estimates the cost of numQueries
    /// controlled walk steps using RepeatEstimates.
    ///
    /// This avoids tracing through each walk step individually, which is
    /// expensive for large numQueries. Instead, it evaluates a single
    /// controlled walk step and multiplies the cost by numQueries.
    ///
    /// Register layout matches QPE convention: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    operation EstimateSOSSAWalkCircuit(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numOuterQubits : Int,
        numOuterIndexQubits : Int,
        numInnerQubits : Int,
        numReflectInner : Int,
        numPhaseGradientQubits : Int,
        numQueries : Int,
    ) : Unit {
        let numSpinQubits = 2;
        let totalAncilla = numOuterQubits + numInnerQubits + numSpinQubits + numPhaseGradientQubits;

        use control = Qubit[1];
        use allQubits = Qubit[numSystemQubits + totalAncilla];
        let systemReg = allQubits[0..numSystemQubits - 1];
        let outerReg = allQubits[numSystemQubits..numSystemQubits + numOuterQubits - 1];
        let innerReg = allQubits[numSystemQubits + numOuterQubits..numSystemQubits + numOuterQubits + numInnerQubits - 1];
        let spinReg = allQubits[numSystemQubits + numOuterQubits + numInnerQubits..numSystemQubits + numOuterQubits + numInnerQubits + numSpinQubits - 1];
        let phaseGradientReg = if numPhaseGradientQubits > 0 {
            allQubits[numSystemQubits + numOuterQubits + numInnerQubits + numSpinQubits..numSystemQubits + numOuterQubits + numInnerQubits + numSpinQubits + numPhaseGradientQubits - 1]
        } else {
            []
        };

        within { RepeatEstimates(numQueries); } apply {
            Controlled SOSSAWalkStep(
                control,
                (outerPrepareOp, innerPrepareOp, selectOp, numReflectInner, numOuterIndexQubits, outerReg, innerReg, spinReg, systemReg, phaseGradientReg),
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════════

    /// Select spin qubit and SWAP up/down registers (arXiv:2502.15882v1, Step 4).
    ///
    /// Coherently computes `spin` from (isSF, spinDQ, spinSF):
    ///   - DQ mode (isSF=0): spin ← spinDQ
    ///   - SF mode (isSF=1): spin ← spinSF
    /// Then SWAPs registerDown ↔ registerUp controlled on spin.
    operation SelectSpins(
        isSF : Qubit,
        spinDQ : Qubit,
        spinSF : Qubit,
        spin : Qubit,
        registerDown : Qubit[],
        registerUp : Qubit[]
    ) : Unit is Adj + Ctl {
        // DQ mode: copy spinDQ to spin (fires when isSF=0)
        within { X(isSF); } apply { CCNOT(isSF, spinDQ, spin); }
        // SF mode: copy spinSF to spin (fires when isSF=1)
        CCNOT(isSF, spinSF, spin);
        // SWAP up/down registers based on spin
        for i in 0..Length(registerDown) - 1 {
            Controlled SWAP([spin], (registerDown[i], registerUp[i]));
        }
    }

    /// Givens rotation chain with CNOT sandwich (arXiv:2502.15882v1, Appendix B.5).
    ///
    /// Each step G_{j,j+1}(θ) = CX(j→j+1) · Ry(-2θ, j) · CX(j→j+1) acts as a
    /// 2×2 rotation in the single-excitation subspace {|01⟩,|10⟩} of (target[j], target[j+1]).
    /// The full chain maps orbital content to qubit 0 for MajoranaOp.
    ///
    /// NOTE: This is the direct rotation (simulation) version. The production
    /// implementation should use QROM to load rotation angles into an ancilla
    /// register, then apply phase-gradient rotation (Rz via addition to a
    /// phase-gradient register). See MakeSelectPhaseGradient.
    ///
    /// DQ rotations: controlled on xoReg ∈ [0, N), unconditional on b.
    /// SF rotations: controlled on (xoReg, bReg) jointly.
    operation ApplyMultiControlledRotations(
        params : SelectParams,
        N : Int,
        numSF : Int,
        numBp1 : Int,
        numRotAngles : Int,
        xoBits : Int,
        isSF : Qubit,
        xoReg : Qubit[],
        bReg : Qubit[],
        sysRegDown : Qubit[],
        bEqBQubit : Qubit
    ) : Unit is Adj + Ctl {
        for j in 0..numRotAngles - 1 {
            CNOT(sysRegDown[j], sysRegDown[j + 1]);

            // DQ rotations: x_o in [0, N)
            // No extra control on sysRegDown[j+1]: the CNOT sandwich already
            // restricts the 1-excitation subspace, and the Givens rotation must
            // act on ALL sectors (0, 1, 2 excitations) so that its adjoint
            // properly uncomputes after MajoranaOp changes the particle number.
            for a in 0..N - 1 {
                let angle = params.OneBodyRotationAngles[a][j];
                ApplyControlledOnInt(a, Ry(-2.0 * angle, _), xoReg, sysRegDown[j]);
            }

            // SF rotations: x_o in [N, N+numSF), conditioned on b
            for xoIdx in 0..numSF - 1 {
                let xo = N + xoIdx;
                let r = xoIdx / params.numCopies;
                for b in 0..numBp1 - 1 {
                    let angleIdx = b * params.numRanks + r;
                    if angleIdx < Length(params.TwoBodyRotationAngles) and j < Length(params.TwoBodyRotationAngles[angleIdx]) {
                        let angle = params.TwoBodyRotationAngles[angleIdx][j];
                        let condValue = xo + b * (1 <<< xoBits);
                        ApplyControlledOnInt(condValue, Ry(-2.0 * angle, _), xoReg + bReg, sysRegDown[j]);
                    }
                }
            }

            CNOT(sysRegDown[j], sysRegDown[j + 1]);
        }
        // Set bEqB flag: 1 when (isSF AND b == B)
        ApplyControlledOnInt(params.numBases, q => Controlled X([isSF], q), bReg, bEqBQubit);
    }

    /// Givens rotation chain using split DQ/SF QROM bulk-load + phase gradient rotation.
    ///
    /// Loads ALL (N-1) rotation angles at once using two Controlled Select calls:
    ///   - DQ: Select(N entries) addressed by xoReg[0..⌈log₂N⌉-1], fires when isSF=0
    ///   - SF: Select(R*2^bBits entries) addressed by bReg++rBits, fires when isSF=1
    /// Then applies all Givens rotations from the loaded rotTarget register.
    ///
    /// Cost: N + R*2^bBits (Select) + (N-1) × Adder(bRot) (rotations).
    /// Matches paper cost formula (arXiv:2502.15882v1, Step 5).
    ///
    /// Reference: arXiv:2502.15882v1, Appendix B.5; Babbush et al. (arXiv:1805.03662).
    operation ApplyGivensRotationsQROM(
        params : SelectParams,
        N : Int,
        numSF : Int,
        numBp1 : Int,
        numRotAngles : Int,
        xoBits : Int,
        isSF : Qubit,
        xoReg : Qubit[],
        bReg : Qubit[],
        rBits : Qubit[],
        sysRegDown : Qubit[],
        phaseGradientReg : Qubit[],
        bEqBQubit : Qubit,
    ) : Unit is Adj + Ctl {
        let bRot = params.rotationBitPrecision;
        let bBits = Length(bReg);
        let R = params.numRanks;
        let nRotBits = numRotAngles * bRot;
        let nDQBits = Ceiling(Lg(IntAsDouble(if N > 1 { N } else { 2 })));

        // DQ table: N entries × (N-1)*bRot bits, addressed by xoReg[0..nDQBits-1]
        let dqData = BuildDQBulkRotationData(params, N, numRotAngles, bRot);

        // SF table: R*2^bBits entries × ((N-1)*bRot + 1) bits, addressed by bReg++rBits
        // The +1 bit is the bEqB flag (1 when b == B).
        let sfData = BuildSFBulkRotationData(params, R, numRotAngles, bRot, bBits);

        // Allocate rotation target register: (N-1)*bRot rotation bits + 1 bEqB flag bit.
        use rotTarget = Qubit[nRotBits + 1];

        // DQ load: fires when isSF=0, addressed by first ⌈log₂N⌉ bits of xoReg
        // DQ entries have only rotation bits (no bEqB), so target excludes last qubit.
        within { X(isSF); } apply {
            Controlled Select([isSF], (dqData, xoReg[0..nDQBits - 1], rotTarget[0..nRotBits - 1]));
        }
        // SF load: fires when isSF=1, addressed by (bReg ++ rBits)
        // SF entries include bEqB flag at position nRotBits.
        Controlled Select([isSF], (sfData, bReg + rBits, rotTarget));

        // Copy bEqB flag from QROM output to persistent qubit.
        // Cost: 1 CNOT (vs ⌈log₂(B+1)⌉ Toffoli for ApplyControlledOnInt).
        CNOT(rotTarget[nRotBits], bEqBQubit);

        // Apply all Givens rotations from the loaded register.
        // Uses DFTHC-style conjugation: CNOT(j+1→j) + S†H converts Rz→Ry
        // and conditions on particle-number subspace, all with an uncontrolled
        // adder (n Toffoli) instead of a controlled adder (2n Toffoli).
        // Reference: Sanders et al. (arXiv:2007.07391, §IIA1, Figure 4a).
        for j in 0..numRotAngles - 1 {
            within {
                CNOT(sysRegDown[j + 1], sysRegDown[j]);
            } apply {
                RyViaPhaseGradient(sysRegDown[j], rotTarget[j * bRot..(j + 1) * bRot - 1], phaseGradientReg);
            }
        }
    }

    /// Build DQ bulk rotation data: N entries, each containing all (N-1) quantized angles.
    /// Addressed by xoReg[0..⌈log₂N⌉-1] (the orbital index for one-body terms).
    internal function BuildDQBulkRotationData(
        params : SelectParams,
        N : Int,
        numRotAngles : Int,
        bRot : Int,
    ) : Bool[][] {
        mutable table : Bool[][] = [];
        for xo in 0..N - 1 {
            mutable bits : Bool[] = [];
            for j in 0..numRotAngles - 1 {
                let angle = if j < Length(params.OneBodyRotationAngles[xo]) {
                    params.OneBodyRotationAngles[xo][j]
                } else {
                    0.0
                };
                set bits += IntAsBoolArray(QuantizeGivensAngle(angle, bRot), bRot);
            }
            set table += [bits];
        }
        return table;
    }

    /// Build SF bulk rotation data: R*2^bBits entries, each containing all (N-1) quantized angles
    /// plus a 1-bit bEqB flag indicating b == numBases.
    /// Address layout: bReg (low bBits) ++ rBits (high), so addr = b + r * 2^bBits.
    /// Matches DFTHC paper cost formula: R*(B+1) entries.
    internal function BuildSFBulkRotationData(
        params : SelectParams,
        R : Int,
        numRotAngles : Int,
        bRot : Int,
        bBits : Int,
    ) : Bool[][] {
        let nInnerSlots = 1 <<< bBits; // 2^bBits
        let tableSize = R * nInnerSlots;

        mutable table : Bool[][] = [];
        for idx in 0..tableSize - 1 {
            let b = idx % nInnerSlots;   // basis = low bits (bReg is LSB)
            let r = idx / nInnerSlots;   // rank = high bits (rBits is MSB)

            mutable bits : Bool[] = [];
            for j in 0..numRotAngles - 1 {
                let angleIdx = b * params.numRanks + r;
                let angle = if r < R and angleIdx < Length(params.TwoBodyRotationAngles) and j < Length(params.TwoBodyRotationAngles[angleIdx]) {
                    params.TwoBodyRotationAngles[angleIdx][j]
                } else {
                    0.0
                };
                set bits += IntAsBoolArray(QuantizeGivensAngle(angle, bRot), bRot);
            }
            // Append bEqB flag: true when b == numBases (the identity term)
            set bits += [b == params.numBases];
            set table += [bits];
        }
        return table;
    }

    /// Quantize a Givens rotation angle for phase gradient application.
    ///
    /// RyViaPhaseGradient applies Ry(4π·x/2^b). To achieve Ry(-2θ):
    ///   4π·x/2^b = -2θ  →  x = -2^b · θ / (2π)  (mod 2^b)
    internal function QuantizeGivensAngle(angle : Double, bRot : Int) : Int {
        let scale = IntAsDouble(1 <<< bRot);
        let raw = Round(-scale * angle / (2.0 * PI()));
        ((raw % (1 <<< bRot)) + (1 <<< bRot)) % (1 <<< bRot)
    }

    /// Controlled Majorana Operator on single qubit (arXiv:2502.15882v1, Fig. 4 / Appendix B.6).
    ///
    /// - `sf_vs_dq`: 1 if SF (two-body), 0 if DQ (one-body)
    /// - `d_vs_q`: 0 for D1 (annihilation), 1 for Q1 (creation)
    /// - `bEqB`: 1 if b==B (identity term for SF), 0 otherwise
    /// - `spin`: computed spin qubit controlling up/down
    /// - `system_reg_0`: target qubit (qubit 0 after Givens rotation)
    operation MajoranaOp(
        sf_vs_dq : Qubit,
        d_vs_q : Qubit,
        bEqB : Qubit,
        spin : Qubit,
        system_reg_0 : Qubit
    ) : Unit is Adj + Ctl {
        // SF two-body (b < B): Z on system_reg_0 when sf_vs_dq=1 AND bEqB=0
        within { X(bEqB); } apply {
            Controlled Z([sf_vs_dq, bEqB], system_reg_0);
        }
        // DQ: X on system_reg_0 with spin-dependent Z
        within { X(sf_vs_dq); } apply {
            CNOT(sf_vs_dq, system_reg_0);
            Controlled Z([sf_vs_dq, spin], system_reg_0);
        }
        // DQ Q1 sign flip: Z(spin) when sf_vs_dq=0 AND d_vs_q=1
        within { X(sf_vs_dq); } apply {
            Controlled Z([sf_vs_dq, d_vs_q], spin);
        }
    }


    /// Build an outer PREPARE that applies PreparePureStateD (pure-state amplitude encoding).
    /// Returns a callable (Qubit[]) => Unit is Adj + Ctl.
    function MakeOuterPreparePureState(coefficients : Double[]) : (Qubit[]) => Unit is Adj + Ctl {
        // Reversed: PreparePureStateD is big-endian (qubit[0]=MSB), but
        // Select/ApplyControlledOnInt are little-endian (qubit[0]=LSB).
        (register) => PreparePureStateD(coefficients, Reversed(register))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test wrappers
    // ═══════════════════════════════════════════════════════════════════════════

    /// Generic wrapper: applies an operation to a freshly allocated register.
    operation TestApplyOuterPrep(op : (Qubit[]) => Unit is Adj + Ctl, n : Int) : Unit {
        let qs = QIR.Runtime.AllocateQubitArray(n);
        op(qs);
    }

    /// Wrapper: applies outer prep then inner prep on separate registers.
    operation TestApplyOuterInnerPrep(
        outerOp : (Qubit[]) => Unit is Adj + Ctl,
        innerOp : (Qubit[], Qubit[]) => Unit is Adj,
        nOuter : Int,
        nInner : Int,
    ) : Unit {
        let qs = QIR.Runtime.AllocateQubitArray(nOuter + nInner);
        let outerReg = qs[0..nOuter - 1];
        let innerReg = qs[nOuter..nOuter + nInner - 1];
        outerOp(outerReg);
        innerOp(outerReg, innerReg);
    }


    /// Test the full SELECT on an entry with known angles.
    operation TestSelectDQ(
        selectData : SelectParams,
        xoValue : Int,
    ) : Unit {
        let N = selectData.numOrbitals;
        let numPositiveOneBody = selectData.numPositiveOneBody;
        let numSF = selectData.numRanks * selectData.numCopies;
        let Xo = N + numSF;
        let xoBits = Ceiling(Lg(IntAsDouble(if Xo > 1 { Xo } else { 2 })));
        let numBp1 = selectData.numBases + 1;
        let bBits = Ceiling(Lg(IntAsDouble(if numBp1 > 1 { numBp1 } else { 2 })));
        let nFR = selectData.numFreeRiderBits;

        let nOuter = xoBits;
        let nInner = bBits + nFR;
        let nSpin = 2;
        let nSystem = 2 * N;
        let total = nOuter + nInner + nSpin + nSystem;
        let qs = QIR.Runtime.AllocateQubitArray(total);

        let outerReg = qs[0..nOuter - 1];
        let innerReg = qs[nOuter..nOuter + nInner - 1];
        let spinReg = qs[nOuter + nInner..nOuter + nInner + nSpin - 1];
        let systemReg = qs[nOuter + nInner + nSpin..total - 1];

        let xoReg = outerReg[0..xoBits - 1];
        for bit in 0..xoBits - 1 {
            if (xoValue >>> bit) &&& 1 == 1 {
                X(xoReg[bit]);
            }
        }
        H(spinReg[0]); // spinDQ

        let frStart = bBits;
        if xoValue >= N { X(innerReg[frStart]); }
        if xoValue >= numPositiveOneBody { X(innerReg[frStart + 1]); }

        X(systemReg[0]);

        SelectImpl(selectData, false, outerReg, innerReg, spinReg, systemReg, []);
    }


}
