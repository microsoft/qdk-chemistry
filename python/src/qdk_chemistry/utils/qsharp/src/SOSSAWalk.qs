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
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepareWithFreeRider;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;
    import QDKChemistry.Utils.SelectSwap.SelectSwap;

    // ═══════════════════════════════════════════════════════════════════════════
    // Type aliases for composable sub-operations
    // ═══════════════════════════════════════════════════════════════════════════

    /// Outer PREPARE: (outerReg: Qubit[]) => Unit is Adj + Ctl
    /// Inner PREPARE: (outerReg: Qubit[], innerReg: Qubit[]) => Unit is Adj + Ctl
    /// SELECT: (outerReg: Qubit[], innerReg: Qubit[], spinReg: Qubit[], systemReg: Qubit[]) => Unit is Adj + Ctl

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
        numD1 : Int,
        dqRotationAngles : Double[][],
        sfRotationAngles : Double[][],
        rotationBitPrecision : Int,
        /// Number of free-rider bits at the end of innerReg loaded by inner PREPARE QROM.
        /// Layout: [sf_vs_dq(1), d_vs_q(1), r_bits(⌈log₂ R⌉)].
        /// When > 0, SelectImpl reads isSF and dvsq from innerReg instead of computing them.
        numFreeRiderBits : Int,
    }

    /// Build a SELECT using QROM + phase gradient rotation.
    ///
    /// Givens rotations are applied via SelectSwap QROM angle load + RyViaPhaseGradient.
    /// This is the production implementation for fault-tolerant resource estimation.
    function MakeSelectPhaseGradient(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg, spinReg, systemReg) => {
            SelectImpl(params, true, outerReg, innerReg, spinReg, systemReg);
        }
    }

    /// Build a SELECT using direct rotation synthesis.
    ///
    /// Givens rotations are applied via multi-controlled Ry gates.
    /// Useful for simulation and testing (no ancilla overhead).
    function MakeSelectDirectRotation(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg, spinReg, systemReg) => {
            SelectImpl(params, false, outerReg, innerReg, spinReg, systemReg);
        }
    }

    /// Shared SELECT implementation (arXiv:2502.15882v1, Section 5 / Appendix B.5-B.6).
    ///
    /// Implements: within{SelectSpins} apply{ within{GivensRotations} apply{MajoranaOp} }
    ///
    /// isSF and dvsq are read from the free-rider register at the end of innerReg,
    /// loaded by inner PREPARE's conditional alias sampling QROM.
    /// H(spinSF) is applied externally in the walk step as part of inner preparation.
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
    operation SelectImpl(
        params : SelectParams,
        usePhaseGradient : Bool,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
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
                    ApplyGivensRotationsQROM(
                        params,
                        N,
                        numSF,
                        numBp1,
                        numRotAngles,
                        xoBits,
                        xoReg,
                        bReg,
                        sysRegDown
                    );
                } else {
                    ApplyConditionalGivensRotations(
                        params,
                        N,
                        numSF,
                        numBp1,
                        numRotAngles,
                        xoBits,
                        xoReg,
                        bReg,
                        sysRegDown
                    );
                }
                // Set bEqB flag: 1 when (isSF AND b == B)
                ApplyControlledOnInt(params.numBases, q => Controlled X([isSF], q), bReg, bEqBQubit);
            } apply {
                // Majorana operator (Fig. 4 / Appendix B.6)
                MajoranaOp(isSF, dvsq, bEqBQubit, spin, sysRegDown[0]);
                // DQ phase correction on spinSF
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
    operation SOSSAWalkStep(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
    ) : Unit is Adj + Ctl {
        body ... {
            // U: OuterPREP · H(spinDQ) · within{InnerPREP + H(spinSF)} apply{SELECT}
            outerPrepareOp(outerReg);
            H(spinReg[0]);
            within {
                innerPrepareOp(outerReg, innerReg);
                H(spinReg[1]); // H(spinSF): part of inner prep for reflection
            } apply {
                selectOp(outerReg, innerReg, spinReg, systemReg);
            }

            // Ref_B: inner reflection (includes spinSF so DQ/SF phase is detected)
            Reflect(innerReg + [spinReg[1]]);

            // U†: must undo inner BE first (needs outerReg superposition), then outer
            within {
                innerPrepareOp(outerReg, innerReg);
                H(spinReg[1]); // H(spinSF)
            } apply {
                Adjoint selectOp(outerReg, innerReg, spinReg, systemReg);
            }
            H(spinReg[0]);
            Adjoint outerPrepareOp(outerReg);

            // Ref_{a,B}: outer reflection
            Reflect(outerReg + innerReg + spinReg);
        }
        adjoint auto;
        controlled (ctls, ...) {
            // Only reflections are controlled for QPE.
            // U (uncontrolled)
            outerPrepareOp(outerReg);
            H(spinReg[0]);
            within {
                innerPrepareOp(outerReg, innerReg);
                H(spinReg[1]); // H(spinSF)
            } apply {
                selectOp(outerReg, innerReg, spinReg, systemReg);
            }

            // c-Ref_B
            Controlled Reflect(ctls, innerReg + [spinReg[1]]);

            // U† (uncontrolled): inner BE adjoint first, then outer
            within {
                innerPrepareOp(outerReg, innerReg);
                H(spinReg[1]); // H(spinSF)
            } apply {
                Adjoint selectOp(outerReg, innerReg, spinReg, systemReg);
            }
            H(spinReg[0]);
            Adjoint outerPrepareOp(outerReg);

            // c-Ref_{a,B}
            Controlled Reflect(ctls, outerReg + innerReg + spinReg);
        }
        controlled adjoint auto;
    }

    /// Creates a controlled SOSSA walk callable from pre-built sub-ops.
    function MakeControlledSOSSAWalkOp(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numOuterQubits : Int,
        numInnerQubits : Int,
        power : Int,
    ) : (Qubit, Qubit[]) => Unit {
        let numSpinQubits = 2; // spinReg = [spinDQ, spinSF]
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
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numOuterQubits : Int,
        numInnerQubits : Int,
        power : Int,
    ) : Unit {
        let numSpinQubits = 2; // spinReg = [spinDQ, spinSF]
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
    operation ApplyConditionalGivensRotations(
        params : SelectParams,
        N : Int,
        numSF : Int,
        numBp1 : Int,
        numRotAngles : Int,
        xoBits : Int,
        xoReg : Qubit[],
        bReg : Qubit[],
        sysRegDown : Qubit[]
    ) : Unit is Adj + Ctl {
        for j in 0..numRotAngles - 1 {
            CNOT(sysRegDown[j], sysRegDown[j + 1]);

            // DQ rotations: x_o in [0, N)
            for a in 0..N - 1 {
                let angle = params.dqRotationAngles[a][j];
                ApplyControlledOnInt(a, q => Ry(-2.0 * angle, q), xoReg, sysRegDown[j]);
            }

            // SF rotations: x_o in [N, N+numSF), conditioned on b
            for xoIdx in 0..numSF - 1 {
                let xo = N + xoIdx;
                let r = xoIdx / params.numCopies;
                for b in 0..numBp1 - 1 {
                    let angleIdx = b * params.numRanks + r;
                    if angleIdx < Length(params.sfRotationAngles) and j < Length(params.sfRotationAngles[angleIdx]) {
                        let angle = params.sfRotationAngles[angleIdx][j];
                        let condValue = xo + b * (1 <<< xoBits);
                        ApplyControlledOnInt(condValue, q => Ry(-2.0 * angle, q), xoReg + bReg, sysRegDown[j]);
                    }
                }
            }

            CNOT(sysRegDown[j], sysRegDown[j + 1]);
        }
    }

    /// Givens rotation chain using QROM angle load + phase gradient rotation.
    ///
    /// For each step j, loads the quantized angle θ_{xo,b,j} from a QROM table
    /// addressed by (xoReg ++ bReg), then applies Ry(-2θ) via phase gradient addition.
    /// The CNOT sandwich structure is the same as the direct version.
    ///
    /// Cost per step: 1 SelectSwap QROM (load) + 1 RippleCarryAdder (rotation) + 1 SelectSwap† (unload).
    /// Total cost: (N-1) × [QROM(2^(xoBits+bBits), bRot) + Adder(bRot)].
    ///
    /// Reference: arXiv:2502.15882v1, Appendix B.5; Sanders et al. (arXiv:2007.07391).
    operation ApplyGivensRotationsQROM(
        params : SelectParams,
        N : Int,
        numSF : Int,
        numBp1 : Int,
        numRotAngles : Int,
        xoBits : Int,
        xoReg : Qubit[],
        bReg : Qubit[],
        sysRegDown : Qubit[]
    ) : Unit is Adj + Ctl {
        let bRot = params.rotationBitPrecision;
        let bBits = Length(bReg);
        let angleTables = ComputeGivensAngleTables(
            params,
            N,
            numSF,
            numBp1,
            numRotAngles,
            xoBits,
            bBits
        );

        use phaseGradient = Qubit[bRot];
        use angleReg = Qubit[bRot];

        within {
            PreparePhaseGradientState(phaseGradient);
        } apply {
            for j in 0..numRotAngles - 1 {
                CNOT(sysRegDown[j], sysRegDown[j + 1]);
                within {
                    SelectSwap(-1, angleTables[j], xoReg + bReg, angleReg);
                } apply {
                    RyViaPhaseGradient(sysRegDown[j], angleReg, phaseGradient);
                }
                CNOT(sysRegDown[j], sysRegDown[j + 1]);
            }
        }
    }

    /// Compute QROM data tables for all Givens rotation steps.
    ///
    /// Returns Bool[numRotAngles][2^(xoBits+bBits)][bRot]:
    ///   tables[j][addr] = quantized angle for step j at address (xo, b).
    ///
    /// Address encoding: addr = xo + b * 2^xoBits (xoReg is LSB part).
    /// Angles are quantized so that RyViaPhaseGradient(x) = Ry(4π·x/2^b) = Ry(-2θ).
    internal function ComputeGivensAngleTables(
        params : SelectParams,
        N : Int,
        numSF : Int,
        numBp1 : Int,
        numRotAngles : Int,
        xoBits : Int,
        bBits : Int,
    ) : Bool[][][] {
        let bRot = params.rotationBitPrecision;
        let addrBits = xoBits + bBits;
        let numAddresses = 1 <<< addrBits;

        mutable tables : Bool[][][] = [];
        for j in 0..numRotAngles - 1 {
            mutable table : Bool[][] = [];
            for addr in 0..numAddresses - 1 {
                let xo = addr % (1 <<< xoBits);
                let b = addr / (1 <<< xoBits);

                mutable angle = 0.0;
                if xo < N {
                    // DQ entry: angle depends on xo only (same for all b)
                    if j < Length(params.dqRotationAngles[xo]) {
                        set angle = params.dqRotationAngles[xo][j];
                    }
                } elif xo < N + numSF {
                    // SF entry: angle depends on (xo, b) via rank index
                    let xoIdx = xo - N;
                    let r = xoIdx / params.numCopies;
                    let angleIdx = b * params.numRanks + r;
                    if angleIdx < Length(params.sfRotationAngles) and j < Length(params.sfRotationAngles[angleIdx]) {
                        set angle = params.sfRotationAngles[angleIdx][j];
                    }
                }
                // else: invalid address → angle = 0

                let quantized = QuantizeGivensAngle(angle, bRot);
                set table += [IntAsBoolArray(quantized, bRot)];
            }
            set tables += [table];
        }
        return tables;
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
        // DQ: X on system_reg_0 with spin-dependent Z (gives X when spin=0, iY when spin=1)
        within { X(sf_vs_dq); } apply {
            CNOT(sf_vs_dq, system_reg_0);
            Controlled Z([sf_vs_dq, spin], system_reg_0);
        }
        // DQ Q1 sign flip: Z(spin) when sf_vs_dq=0 AND d_vs_q=1
        // Distinguishes a† (D1, d_vs_q=0) from a (Q1, d_vs_q=1)
        within { X(sf_vs_dq); } apply {
            Controlled Z([sf_vs_dq, d_vs_q], spin);
        }
    }

    /// Apply a sequence of Givens rotations to target qubits (standalone, no CNOT sandwich).
    /// Used for testing individual orbital rotations.
    operation ApplyGivensSequence(angles : Double[], target : Qubit[]) : Unit is Adj + Ctl {
        let numAngles = Length(angles);
        let numQubits = Length(target);
        for j in 0..numAngles - 1 {
            if j + 1 < numQubits {
                CNOT(target[j], target[j + 1]);
                Ry(-2.0 * angles[j], target[j]);
                CNOT(target[j], target[j + 1]);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Exported helper operations for unit testing
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build an outer PREPARE that applies PreparePureStateD (pure-state amplitude encoding).
    /// Returns a callable (Qubit[]) => Unit is Adj + Ctl.
    function MakeOuterPreparePureState(coefficients : Double[]) : (Qubit[]) => Unit is Adj + Ctl {
        PreparePureStateD(coefficients, _)
    }

    /// Reflect about the |0⟩ state: R = 2|0⟩⟨0| - I.
    /// |0⟩ → +|0⟩, |x≠0⟩ → -|x≠0⟩.
    operation ReflectAboutZero(qs : Qubit[]) : Unit is Adj + Ctl {
        Reflect(qs);
    }

    /// Majorana D1 (double-qubit annihilation): X then CZ.
    /// On |spin⟩|sys⟩: applies X(sys[0]) then CZ(spin[0], sys[0]).
    operation MajoranaD1(spin : Qubit[], sys : Qubit[]) : Unit is Adj + Ctl {
        X(sys[0]);
        CZ(spin[0], sys[0]);
    }

    /// Majorana SF (spin-free two-body): Z on sys[0].
    operation MajoranaSF(sys : Qubit[]) : Unit is Adj + Ctl {
        Z(sys[0]);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test wrappers — allocate qubits via QIR.Runtime so they persist for
    // dump_machine (qubit values cannot cross the Python ↔ Q# boundary).
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

    /// Apply Givens rotation chain to |10...0⟩ and leave state for dump_machine.
    operation TestGivensRotation(angles : Double[], n : Int) : Unit {
        let qs = QIR.Runtime.AllocateQubitArray(n);
        X(qs[0]);
        ApplyGivensSequence(angles, qs);
    }

    /// Apply Adjoint Givens rotation to a prepared state (spread excitation) to verify round-trip.
    operation TestGivensRoundTrip(angles : Double[], n : Int) : Unit {
        let qs = QIR.Runtime.AllocateQubitArray(n);
        X(qs[0]);
        ApplyGivensSequence(angles, qs);
        Adjoint ApplyGivensSequence(angles, qs);
    }

    /// Apply SelectSpins and dump state to verify SWAP logic.
    operation TestSelectSpins(spinDQVal : Bool, spinSFVal : Bool, n : Int) : Unit {
        let qs = QIR.Runtime.AllocateQubitArray(2 * n + 4);
        // Register layout: isSF(1) + spinDQ(1) + spinSF(1) + spin(1) + sysDown(n) + sysUp(n)
        let isSF = qs[0];
        let spinDQ = qs[1];
        let spinSF = qs[2];
        let spin = qs[3];
        let sysDown = qs[4..4 + n - 1];
        let sysUp = qs[4 + n..4 + 2 * n - 1];

        if spinDQVal { X(spinDQ); }
        if spinSFVal { X(spinSF); }
        X(sysDown[0]);

        SelectSpins(isSF, spinDQ, spinSF, spin, sysDown, sysUp);
    }

    /// Apply SelectSpins in SF mode.
    operation TestSelectSpinsSF(spinSFVal : Bool, n : Int) : Unit {
        let qs = QIR.Runtime.AllocateQubitArray(2 * n + 4);
        let isSF = qs[0];
        let spinDQ = qs[1];
        let spinSF = qs[2];
        let spin = qs[3];
        let sysDown = qs[4..4 + n - 1];
        let sysUp = qs[4 + n..4 + 2 * n - 1];

        X(isSF);
        if spinSFVal { X(spinSF); }
        X(sysDown[0]);

        SelectSpins(isSF, spinDQ, spinSF, spin, sysDown, sysUp);
    }

    /// Test the full SELECT on an entry with known angles.
    operation TestSelectDQ(
        selectData : SelectParams,
        xoValue : Int,
    ) : Unit {
        let N = selectData.numOrbitals;
        let numD1 = selectData.numD1;
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
        if xoValue >= numD1 { X(innerReg[frStart + 1]); }

        X(systemReg[0]);

        SelectImpl(selectData, false, outerReg, innerReg, spinReg, systemReg);
    }

    /// Test SELECT round trip: SELECT†·SELECT should be identity.
    operation TestSelectRT(
        selectData : SelectParams,
        xoValue : Int,
    ) : Unit {
        let N = selectData.numOrbitals;
        let numD1 = selectData.numD1;
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
            if (xoValue >>> bit) &&& 1 == 1 { X(xoReg[bit]); }
        }

        let frStart = bBits;
        if xoValue >= N { X(innerReg[frStart]); }
        if xoValue >= numD1 { X(innerReg[frStart + 1]); }

        X(systemReg[0]);

        SelectImpl(selectData, false, outerReg, innerReg, spinReg, systemReg);
        Adjoint SelectImpl(selectData, false, outerReg, innerReg, spinReg, systemReg);
    }
}
