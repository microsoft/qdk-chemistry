// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Sum of Squares Spectral Amplification (SOSSA) walk operator.
///
/// Composable design: each sub-operation (OuterPrepare, InnerPrepare, Select)
/// is built independently as a Q# callable, and this module assembles them into
/// the block encoding B that phase estimation queries.
///
/// Walk operator (Low et al., Phys. Rev. X 15 (2025), inline above Eq. (9); App. A 2):
///   W = Ref_{a,B} · B,  B = U† · Ref_B · U
/// where U = OuterPREP · within{InnerPREP} apply{SELECT}.
///
/// Only B lives here. `Ref_{a,B}` is a plain reflection about the all-zero state of the
/// leading `SOSSAWalkAncillaCount(layout) - layout.numPhaseGradientQubits` ancillas, so
/// callers pair `MakeSOSSABlockEncodingOp` with the generic
/// `QDKChemistry.Utils.PrepSelPrep.MakeAncillaReflectionOp`.
namespace QDKChemistry.Utils.SOSSAWalk {

    import Std.Arrays.Padded;
    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyToEachCA;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;

    import Std.Math.AbsD;
    import Std.Math.BitSizeI;

    import Std.Math.PI;
    import Std.Math.Round;
    import Std.StatePreparation.PreparePureStateD;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepareWithFreeRider;
    import Std.Arithmetic.RippleCarryCGIncByLE;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState, QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;


    // ═══════════════════════════════════════════════════════════════════════════
    // Inner PREPARE
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
        let nIndexBits = BitSizeI(nCoeffs - 1);
        let mu = coefficientBitPrecision;
        let nFreeRider = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        let qromEnd = 2 * nIndexBits + 2 * mu + 2;
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
        let nIndexBits = BitSizeI((if nCoeffs > 1 { nCoeffs } else { 2 }) - 1);
        let nFreeRider = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        // innerReg layout: bReg[nIndexBits] + freeRiderReg[nFR]
        (outerReg, innerReg) => {
            let bReg = innerReg[0..nIndexBits - 1];
            let freeRiderReg = if nFreeRider > 0 {
                innerReg[nIndexBits..nIndexBits + nFreeRider - 1]
            } else {
                []
            };

            if nFreeRider > 0 {
                Select(freeRiderData, outerReg, freeRiderReg);
            }

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
    // SELECT
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
        let xoBits = BitSizeI((if Xo > 1 { Xo } else { 2 }) - 1);
        let numBp1 = params.numBases + 1;
        let bBits = BitSizeI((if numBp1 > 1 { numBp1 } else { 2 }) - 1);
        let numRotAngles = N - 1;

        // Register slicing
        let xoReg = outerReg[0..xoBits - 1];
        let spinDQ = spinReg[0];
        let spinSF = spinReg[1];
        let bReg = innerReg[0..bBits - 1];
        let sysRegDown = systemReg[0..N - 1];
        let sysRegUp = systemReg[N..2 * N - 1];

        // Free-rider data from inner PREPARE QROM: [sf_vs_dq(1), d_vs_q(1), r_bits...]
        let nInner = Length(innerReg);
        let nFR = params.numFreeRiderBits;
        let isSF = innerReg[nInner - nFR];       // sf_vs_dq
        let dvsq = innerReg[nInner - nFR + 1];   // d_vs_q

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
    // Walk step
    // ═══════════════════════════════════════════════════════════════════════════

    /// Apply the self-inverse block B = U† · Ref_B · U used by the SOSSA walk.
    ///
    /// `numOuterPrepareGradientQubits` is the prefix of `phaseGradientReg` that the outer
    /// PREPARE reads (nonzero only for a QROM PREPARE); it is appended to `outerReg` when
    /// the PREPARE is applied. The gradient is a persistent resource that every consumer
    /// leaves unchanged, so PREPARE and SELECT may both address it.
    operation SOSSABlockEncoding(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numReflectInner : Int,
        numOuterIndexQubits : Int,
        numOuterPrepareGradientQubits : Int,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj {
        let outerIndexReg = outerReg[0..numOuterIndexQubits - 1];
        let outerPrepareReg = outerReg + phaseGradientReg[0..numOuterPrepareGradientQubits - 1];

        outerPrepareOp(outerPrepareReg);
        H(spinReg[0]);
        within {
            innerPrepareOp(outerIndexReg, innerReg);
            H(spinReg[1]);
        } apply {
            selectOp(outerIndexReg, innerReg, spinReg, systemReg, phaseGradientReg);
        }

        Reflect(innerReg[0..numReflectInner - 1] + [spinReg[1]]);

        within {
            innerPrepareOp(outerIndexReg, innerReg);
            H(spinReg[1]);
        } apply {
            Adjoint selectOp(outerIndexReg, innerReg, spinReg, systemReg, phaseGradientReg);
        }
        H(spinReg[0]);
        Adjoint outerPrepareOp(outerPrepareReg);
    }

    /// Apply the SOSSA block encoding to a flat target register.
    ///
    /// Adapts `SOSSABlockEncoding` to the `Qubit[] => Unit is Adj` shape that the generic
    /// signed-power schedule consumes, by slicing the flat register with `layout`.
    ///
    /// The inner PREPARE's QROM output and free-rider bits are allocated here rather than
    /// taken from `allQubits`. Both are written and exactly uncompute inside this operation,
    /// because SELECT only reads them, so they never carry the success flag. Keeping them
    /// off the flat register is what makes the ancillas the caller reflects about a
    /// contiguous block, so a SOSSA walk pairs with the generic `MakeAncillaReflectionOp`.
    operation SOSSABlockEncodingOnRegister(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        allQubits : Qubit[],
    ) : Unit is Adj {
        let regs = SplitSOSSAWalkRegisters(layout, allQubits);
        use innerScratch = Qubit[SOSSAInnerScratchCount(layout)];
        SOSSABlockEncoding(
            outerPrepareOp,
            innerPrepareOp,
            selectOp,
            layout.numReflectInner,
            layout.numOuterIndexQubits,
            layout.numOuterPrepareGradientQubits,
            regs.outerReg,
            regs.innerReg + innerScratch,
            regs.spinReg,
            regs.systemReg,
            regs.phaseGradientReg,
        );
    }

    /// The SOSSA block encoding B as the `Qubit[] => Unit is Adj` callable QPE consumes.
    ///
    /// The register it takes is `[systemReg | outerReg | innerReg | spinReg | phaseGradientReg]`;
    /// the gradient tail is only present when `layout.numPhaseGradientQubits > 0`.
    function MakeSOSSABlockEncodingOp(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
    ) : (Qubit[] => Unit is Adj) {
        SOSSABlockEncodingOnRegister(outerPrepareOp, innerPrepareOp, selectOp, layout, _)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Register layout
    // ═══════════════════════════════════════════════════════════════════════════

    /// Number of spin qubits in the block encoding: spinReg = [spinDQ, spinSF].
    function NumSOSSASpinQubits() : Int {
        2
    }

    /// Sizes of the registers packed into the target register handed to a walk callable.
    ///
    /// Layout: allQubits = [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    ///
    /// `numInnerQubits` is the full width the inner PREPARE acts on. Only its first
    /// `numReflectInner` qubits — the index, uniform and inequality-flag registers that
    /// carry the success flag — sit in `allQubits`; the QROM output and free-rider bits
    /// behind them are scratch that `SOSSABlockEncodingOnRegister` allocates itself.
    ///
    /// `numOuterPrepareGradientQubits` is the prefix of the phase-gradient register the outer
    /// PREPARE reads; it is shared with SELECT rather than added to the total width.
    struct SOSSAWalkLayout {
        numSystemQubits : Int,
        numOuterQubits : Int,
        numOuterIndexQubits : Int,
        numInnerQubits : Int,
        numReflectInner : Int,
        numPhaseGradientQubits : Int,
        numOuterPrepareGradientQubits : Int,
    }

    /// The individual registers sliced out of the target register.
    struct SOSSAWalkRegisters {
        systemReg : Qubit[],
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        phaseGradientReg : Qubit[],
    }

    /// Width of the inner-PREPARE scratch that is allocated rather than passed in.
    function SOSSAInnerScratchCount(layout : SOSSAWalkLayout) : Int {
        layout.numInnerQubits - layout.numReflectInner
    }

    /// Number of block-encoding ancillas (everything except the system register).
    ///
    /// The reflected ancillas come first and the persistent phase gradient last, so a caller
    /// reflects about `SOSSAWalkAncillaCount(layout) - layout.numPhaseGradientQubits` qubits
    /// starting at `layout.numSystemQubits`.
    function SOSSAWalkAncillaCount(layout : SOSSAWalkLayout) : Int {
        layout.numOuterQubits + layout.numReflectInner + NumSOSSASpinQubits() + layout.numPhaseGradientQubits
    }

    /// Slice a flat target register into the SOSSA block-encoding registers.
    ///
    /// `innerReg` is only the reflected prefix; `SOSSABlockEncodingOnRegister` appends the
    /// scratch it allocates before handing the register to the inner PREPARE and SELECT.
    function SplitSOSSAWalkRegisters(layout : SOSSAWalkLayout, allQubits : Qubit[]) : SOSSAWalkRegisters {
        let outerStart = layout.numSystemQubits;
        let innerStart = outerStart + layout.numOuterQubits;
        let spinStart = innerStart + layout.numReflectInner;
        let gradientStart = spinStart + NumSOSSASpinQubits();
        new SOSSAWalkRegisters {
            systemReg = allQubits[0..outerStart - 1],
            outerReg = allQubits[outerStart..innerStart - 1],
            innerReg = allQubits[innerStart..spinStart - 1],
            spinReg = allQubits[spinStart..gradientStart - 1],
            phaseGradientReg = if layout.numPhaseGradientQubits > 0 {
                allQubits[gradientStart..gradientStart + layout.numPhaseGradientQubits - 1]
            } else {
                []
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // QPE-facing entry point
    // ═══════════════════════════════════════════════════════════════════════════

    /// Circuit entry point: allocates the flat register and applies the block encoding once.
    /// Register layout: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    operation MakeSOSSABlockEncodingCircuit(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
    ) : Unit {
        use allQubits = Qubit[layout.numSystemQubits + SOSSAWalkAncillaCount(layout)];
        if layout.numPhaseGradientQubits > 0 {
            let gradientStart = Length(allQubits) - layout.numPhaseGradientQubits;
            PreparePhaseGradientState(allQubits[gradientStart...]);
        }
        SOSSABlockEncodingOnRegister(outerPrepareOp, innerPrepareOp, selectOp, layout, allQubits);
        ResetAll(allQubits);
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
    ///
    /// This is above the paper's N + R*B: the SF Select is addressed by the full
    /// bReg, so its table is padded from B+1 entries up to 2^bBits =
    /// 2^⌈log₂(B+1)⌉. At FeMoco-54 (N=54, R=10, B=27) that is 374 vs 324
    /// entries. Collapsing the gap needs a Select that can address a
    /// non-power-of-two range, not a change here.
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
        let nDQBits = BitSizeI((if N > 1 { N } else { 2 }) - 1);

        // DQ table: N entries × (N-1)*bRot bits, addressed by xoReg[0..nDQBits-1]
        let dqData = BuildDQBulkRotationData(params, N, numRotAngles, bRot);

        // SF table: R*2^bBits entries × ((N-1)*bRot + 1) bits, addressed by bReg++rBits
        // The +1 bit is the bEqB flag (1 when b == B).
        let sfData = BuildSFBulkRotationData(params, R, numRotAngles, bRot, bBits);

        // Allocate rotation target register: (N-1)*bRot rotation bits + 1 bEqB flag bit.
        use rotTarget = Qubit[nRotBits + 1];

        // The two table loads are conjugated around their consumers so `rotTarget` is
        // uncomputed and released in |0⟩, matching how every other QROM read in this
        // project is written (QROMStatePrep.qs, SelectSwap.qs). The angles have to stay
        // live for the whole rotation chain, so the loads cannot be undone any earlier.
        within {
            // DQ load: fires when isSF=0, addressed by first ⌈log₂N⌉ bits of xoReg
            // DQ entries have only rotation bits (no bEqB), so target excludes last qubit.
            within { X(isSF); } apply {
                Controlled Select([isSF], (dqData, xoReg[0..nDQBits - 1], rotTarget[0..nRotBits - 1]));
            }
            // SF load: fires when isSF=1, addressed by (bReg ++ rBits)
            // SF entries include bEqB flag at position nRotBits.
            Controlled Select([isSF], (sfData, bReg + rBits, rotTarget));
        } apply {
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
    /// The DFTHC cost formula counts R*(B+1) useful entries; the table is padded up to the
    /// next power of two per rank (2^bBits ≥ B+1) because bReg addresses it directly.
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
        // Rejects NaN and ±∞ as well as absurd magnitudes: the comparison is false for
        // NaN, so this fails loudly instead of silently folding a non-finite angle into
        // an in-range bit pattern (NaN and +∞ both quantize to 2^bRot-1, -∞ to 0).
        // Givens angles come from Atan2/hypot and are in [-π, π].
        Fact(AbsD(angle) <= 4.0 * PI(), "QuantizeGivensAngle: angle must be finite and within [-4π, 4π]");
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
        let xoBits = BitSizeI((if Xo > 1 { Xo } else { 2 }) - 1);
        let numBp1 = selectData.numBases + 1;
        let bBits = BitSizeI((if numBp1 > 1 { numBp1 } else { 2 }) - 1);
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
