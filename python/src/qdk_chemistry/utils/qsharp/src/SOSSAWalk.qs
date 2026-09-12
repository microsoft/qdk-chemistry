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

namespace QDKChemistry.Utils.SOSSAWalk {

    import Std.Arrays.Padded;
    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyToEachCA;
    import Std.Canon.ApplyXorInPlace;
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
    import QDKChemistry.Utils.SelectSwap.ComputeOptimalLambda2D, QDKChemistry.Utils.SelectSwap.SelectSwapCost2D;
    import Std.Math.Ceiling;
    import Std.Math.Lg;

    // ═══════════════════════════════════════════════════════════════════════════
    // Parameters
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
        /// Must be at least 2.
        /// Layout: [sf_vs_dq(1), d_vs_q(1), r_bits(⌈log₂ R⌉)].
        /// SelectImpl reads isSF and dvsq from the first two bits.
        numFreeRiderBits : Int,
        /// Index into innerReg of the qubit the inner PREPARE loads the sign of the sampled
        /// (x_o, b) coefficient into, or -1 when the inner PREPARE supplies no sign bit.
        signQubitIndex : Int,
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
        numFreeRiderQubits : Int,
    }

    /// The individual registers sliced out of the target register.
    struct SOSSAWalkRegisters {
        systemReg : Qubit[],
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        phaseGradientReg : Qubit[],
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Classical helpers
    // ═══════════════════════════════════════════════════════════════════════════

    /// Index at which each register starts, followed by the end of the last one.
    internal function SOSSAWalkRegisterBounds(layout : SOSSAWalkLayout) : Int[] {
        let outerStart = layout.numSystemQubits;
        let innerStart = outerStart + layout.numOuterQubits;
        let spinStart = innerStart + layout.numReflectInner;
        let gradientStart = spinStart + 2;
        [outerStart, innerStart, spinStart, gradientStart, gradientStart + layout.numPhaseGradientQubits]
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

    /// Whether the SF rotation table is cheaper addressed `rBits ++ bReg` than `bReg ++ rBits`.
    internal function SFTableRankAddressedFirst(
        numRanks : Int,
        numBases : Int,
        bBits : Int,
        rankBits : Int,
    ) : Bool {
        (numBases + 1) * (1 <<< rankBits) < numRanks * (1 <<< bBits)
    }

    /// Build SF bulk rotation data: all (N-1) quantized angles per entry, plus a 1-bit bEqB
    /// flag indicating b == numBases.
    internal function BuildSFBulkRotationData(
        params : SelectParams,
        R : Int,
        numRotAngles : Int,
        bRot : Int,
        bBits : Int,
        rankBits : Int,
        rankFirst : Bool,
    ) : Bool[][] {
        let bSlots = 1 <<< bBits;
        let rSlots = 1 <<< rankBits;
        let tableSize = if rankFirst { (params.numBases + 1) * rSlots } else { R * bSlots };

        mutable table : Bool[][] = [];
        for idx in 0..tableSize - 1 {
            let b = if rankFirst { idx / rSlots } else { idx % bSlots };
            let r = if rankFirst { idx % rSlots } else { idx / bSlots };

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

    /// One-bit sign table for the inner PREPARE, addressed by `bReg + outerReg`.
    ///
    /// Entry `b + x_o * 2^nIndexBits` is true when the `(x_o, b)` LCU coefficient is
    /// negative. Padding entries are false, matching the zero amplitude they carry.
    internal function BuildInnerSignTable(
        innerCoefficients : Double[][],
        nIndexBits : Int,
    ) : Bool[][] {
        let nCoeffs = Length(innerCoefficients[0]);
        let nPadded = 1 <<< nIndexBits;
        mutable table : Bool[][] = [];
        for row in innerCoefficients {
            for b in 0..nPadded - 1 {
                set table += [[b < nCoeffs and row[b] < 0.0]];
            }
        }
        return table;
    }

    /// Quantize a Givens rotation angle for phase gradient application.
    ///
    /// RyViaPhaseGradient applies Ry(4π·x/2^b). The gated Givens rotation is built as the
    /// controlled CRy(2θ) = Ry(θ)·CNOT·Ry(-θ)·CNOT from two uncontrolled Ry(θ), so each word
    /// encodes the half-angle θ:
    ///   4π·x/2^b = θ  →  x = 2^b · θ / (4π)  (mod 2^b)
    internal function QuantizeGivensAngle(angle : Double, bRot : Int) : Int {
        Fact(AbsD(angle) <= 4.0 * PI(), "QuantizeGivensAngle: angle must be finite and within [-4π, 4π]");
        let scale = IntAsDouble(1 <<< bRot);
        let raw = Round(scale * angle / (4.0 * PI()));
        ((raw % (1 <<< bRot)) + (1 <<< bRot)) % (1 <<< bRot)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Quantum operations
    // ═══════════════════════════════════════════════════════════════════════════

    /// SELECT implementation (arXiv:2502.15882v1, Appendix B.3, B.5-B.6).
    ///
    /// Implements: within{SelectSpins} apply{ within{GivensRotations} apply{MajoranaOp} }
    ///
    /// isSF and dvsq are read from the free-rider register at the end of innerReg,
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

        // Free-rider data from inner PREPARE QROM: [sf_vs_dq(1), d_vs_q(1), r_bits...].
        // The two leading bits are read unconditionally below, so they are required rather
        // than optional: with nFR = 0 the reads below would run off the end of innerReg.
        let nInner = Length(innerReg);
        let nFR = params.numFreeRiderBits;
        Fact(nFR >= 2, "SelectImpl requires at least two free-rider bits (sf_vs_dq and d_vs_q)");
        let isSF = innerReg[nInner - nFR];       // sf_vs_dq
        let dvsq = innerReg[nInner - nFR + 1];   // d_vs_q

        use spin = Qubit();
        use bEqBQubit = Qubit();

        // sign of the sampled (x_o, b) coefficient.
        if params.signQubitIndex >= 0 {
            Z(innerReg[params.signQubitIndex]);
        }

        // The Majorana operator runs in the Givens-rotated basis, so it is passed into the
        // rotation step instead of being wrapped around it.
        let majoranaStep : (Unit => Unit is Adj + Ctl) = () => {
            MajoranaOp(isSF, dvsq, bEqBQubit, spin, spinSF, sysRegDown[0]);
        };

        within {
            SelectSpins(isSF, spinDQ, spinSF, spin, sysRegDown, sysRegUp);
        } apply {
            // Givens rotations: basis change to localize amplitude on qubit 0
            if usePhaseGradient {
                let rBits = if nFR > 2 { innerReg[nInner - nFR + 2..nInner - 1] } else { [] };
                WithGivensRotationsQROM(
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
                    bEqBQubit,
                    majoranaStep
                );
            } else {
                within {
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
                } apply {
                    // Majorana operator (Fig. 4 / Appendix B.6)
                    majoranaStep();
                }
            }
        }
    }

    /// Apply the self-inverse block B = U† · Ref_B · U used by the SOSSA walk.
    ///
    /// `numOuterPrepareGradientQubits` is the prefix of `phaseGradientReg` that the outer
    /// PREPARE reads (nonzero only for a QROM PREPARE); it is appended to `outerReg` when
    /// the PREPARE is applied. The gradient is a persistent resource that every consumer
    /// leaves unchanged, so PREPARE and SELECT may both address it.
    operation SOSSABlockEncoding(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numReflectInner : Int,
        numOuterIndexQubits : Int,
        numOuterPrepareGradientQubits : Int,
        numFreeRiderQubits : Int,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj {
        let outerIndexReg = outerReg[0..numOuterIndexQubits - 1];
        let outerPrepareReg = outerReg + phaseGradientReg[0..numOuterPrepareGradientQubits - 1];
        let freeRiderReg = if numFreeRiderQubits > 0 {
            innerReg[Length(innerReg) - numFreeRiderQubits...]
        } else {
            []
        };

        within {
            outerPrepareOp(outerPrepareReg);
            H(spinReg[0]);
        } apply {
            within {
                freeRiderOp(outerIndexReg, freeRiderReg);
            } apply {
                within {
                    within {
                        innerPrepareOp(outerIndexReg, innerReg);
                        H(spinReg[1]);
                    } apply {
                        selectOp(outerIndexReg, innerReg, spinReg, systemReg, phaseGradientReg);
                    }
                } apply {
                    Reflect(innerReg[0..numReflectInner - 1] + [spinReg[1]]);
                }
            }
        }
    }

    /// Apply the SOSSA block encoding to a flat target register.
    ///
    /// Adapts `SOSSABlockEncoding` to the `Qubit[] => Unit is Adj` shape that the generic
    /// signed-power schedule consumes, by slicing the flat register with `layout`.
    /// The inner PREPARE's QROM output and free-rider bits are allocated here rather than
    /// taken from `allQubits`. Both are written and exactly uncompute inside this operation.
    operation SOSSABlockEncodingOnRegister(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        allQubits : Qubit[],
    ) : Unit is Adj {
        let bounds = SOSSAWalkRegisterBounds(layout);
        let outerStart = bounds[0];
        let innerStart = bounds[1];
        let spinStart = bounds[2];
        let gradientStart = bounds[3];
        let regs = new SOSSAWalkRegisters {
            systemReg = allQubits[0..outerStart - 1],
            outerReg = allQubits[outerStart..innerStart - 1],
            innerReg = allQubits[innerStart..spinStart - 1],
            spinReg = allQubits[spinStart..gradientStart - 1],
            phaseGradientReg = if bounds[4] > gradientStart {
                allQubits[gradientStart..bounds[4] - 1]
            } else {
                []
            }
        };
        use innerScratch = Qubit[layout.numInnerQubits - layout.numReflectInner];
        SOSSABlockEncoding(
            outerPrepareOp,
            freeRiderOp,
            innerPrepareOp,
            selectOp,
            layout.numReflectInner,
            layout.numOuterIndexQubits,
            layout.numOuterPrepareGradientQubits,
            layout.numFreeRiderQubits,
            regs.outerReg,
            regs.innerReg + innerScratch,
            regs.spinReg,
            regs.systemReg,
            regs.phaseGradientReg,
        );
    }

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
        Controlled ApplyToEachCA([spin], (SWAP, Zipped(registerDown, registerUp)));
    }

    /// Givens rotation chain with CNOT sandwich (arXiv:2502.15882v1, Appendix B.5).
    ///
    /// Each step G_{j,j+1}(θ) = CX(j→j+1) · Ctrl_{j+1}[Ry(2θ, j)] · CX(j→j+1) acts as a
    /// 2×2 rotation in the single-excitation subspace {|01⟩,|10⟩} of (target[j], target[j+1]).
    /// The control on target[j+1] keeps the rotation out of the |00⟩/|11⟩ sector, so the chain
    /// preserves particle number; it maps orbital content to qubit 0 for MajoranaOp.
    ///
    /// NOTE: This is the direct rotation (simulation) version. The production
    /// implementation should use QROM to load rotation angles into an ancilla
    /// register, then apply phase-gradient rotation (Rz via addition to a
    /// phase-gradient register). See MakeSelectPhaseGradient.
    ///
    /// DQ rotations: controlled on xoReg ∈ [0, N) and the neighbor target[j+1], unconditional on b.
    /// SF rotations: controlled on (xoReg, bReg) and the neighbor target[j+1] jointly.
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
        for j in numRotAngles - 1..-1..0 {
            CNOT(sysRegDown[j], sysRegDown[j + 1]);

            // DQ rotations: x_o in [0, N)
            for a in 0..N - 1 {
                let angle = params.OneBodyRotationAngles[a][j];
                ApplyControlledOnInt(
                    a + (1 <<< xoBits),
                    Ry(2.0 * angle, _),
                    xoReg + [sysRegDown[j + 1]],
                    sysRegDown[j]
                );
            }

            // SF rotations: x_o in [N, N+numSF), conditioned on b
            for xoIdx in 0..numSF - 1 {
                let xo = N + xoIdx;
                let r = xoIdx / params.numCopies;
                for b in 0..numBp1 - 1 {
                    let angleIdx = b * params.numRanks + r;
                    if angleIdx < Length(params.TwoBodyRotationAngles) and j < Length(params.TwoBodyRotationAngles[angleIdx]) {
                        let angle = params.TwoBodyRotationAngles[angleIdx][j];
                        let condValue = xo + b * (1 <<< xoBits) + (1 <<< (xoBits + Length(bReg)));
                        ApplyControlledOnInt(
                            condValue,
                            Ry(2.0 * angle, _),
                            xoReg + bReg + [sysRegDown[j + 1]],
                            sysRegDown[j]
                        );
                    }
                }
            }

            CNOT(sysRegDown[j], sysRegDown[j + 1]);
        }
        // Set bEqB flag: 1 when (isSF AND b == B)
        ApplyControlledOnInt(params.numBases, q => Controlled X([isSF], q), bReg, bEqBQubit);
    }

    /// Applies the Givens basis change from a QROM-loaded angle word, runs `action` in that
    /// rotated basis, then undoes the chain and releases the angle word.
    ///
    /// Loads ALL (N-1) rotation angles at once using two Select calls:
    ///   - SF: Select over min(R*2^bBits, (B+1)*2^rankBits) entries, uncontrolled
    ///   - DQ: Select(N entries) addressed by xoReg[0..⌈log₂N⌉-1], fires when isSF=0
    ///
    /// Cost: (L_SF - 2) + unlookup(L_SF) for SF, 2*(N-1) for DQ, and 2*(N-1) uncontrolled
    /// Adder(bRot) for the rotations -- each Givens rotation is the neighbor-gated CRy(2θ)
    /// built as Ry(θ)·CNOT·Ry(-θ)·CNOT from two uncontrolled Ry(θ) sharing one angle word, so it
    /// preserves particle number (matching the direct path) without a controlled adder. The SF
    /// unlookup is measurement-based and costs O(sqrt(L)), which is the paper's R + B phase fixup.
    ///
    /// L_SF is above the paper's R*B because `Select` pads whichever register addresses the
    /// low bits out to a power of two; `SFTableRankAddressedFirst` picks the cheaper of the
    /// two orderings, which is all that can be done without a non-power-of-two stride.
    ///
    /// Reference: arXiv:2502.15882v1, Appendix B.5 and B step 7; Babbush et al. (arXiv:1805.03662).
    operation WithGivensRotationsQROM(
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
        action : (Unit => Unit is Adj + Ctl),
    ) : Unit is Adj + Ctl {
        let bRot = params.rotationBitPrecision;
        let bBits = Length(bReg);
        let R = params.numRanks;
        let nRotBits = numRotAngles * bRot;
        let nDQBits = BitSizeI((if N > 1 { N } else { 2 }) - 1);

        // DQ table: N entries × (N-1)*bRot bits, addressed by xoReg[0..nDQBits-1]
        let dqData = BuildDQBulkRotationData(params, N, numRotAngles, bRot);

        // SF table: addressed by (bReg ++ rBits) or (rBits ++ bReg), whichever is smaller.
        let rankFirst = SFTableRankAddressedFirst(R, params.numBases, bBits, Length(rBits));
        let sfData = BuildSFBulkRotationData(params, R, numRotAngles, bRot, bBits, Length(rBits), rankFirst);
        let sfAddress = if rankFirst { rBits + bReg } else { bReg + rBits };

        // Allocate rotation target register: (N-1)*bRot rotation bits + 1 bEqB flag bit.
        use rotTarget = Qubit[nRotBits + 1];

        within {
            // SF load fires only on the SF branch (isSF=1), so the DQ branch keeps `rotTarget`
            // clear and needs no separate unload. `Select` tolerates R or B+1 not being a power
            // of two -- it never reads an address at or above `Length(data)` -- so the raw table
            // is used directly, without padding out to 2^(address qubits).
            Controlled Select([isSF], (sfData, sfAddress, rotTarget));

            // DQ load: fires when isSF=0, addressed by first ⌈log₂N⌉ bits of xoReg.
            within { X(isSF); } apply {
                Controlled Select([isSF], (dqData, xoReg[0..nDQBits - 1], rotTarget[0..nRotBits - 1]));
            }
        } apply {
            within {
                CNOT(rotTarget[nRotBits], bEqBQubit);

                for j in numRotAngles - 1..-1..0 {
                    let word = rotTarget[j * bRot..(j + 1) * bRot - 1];
                    within {
                        CNOT(sysRegDown[j], sysRegDown[j + 1]);
                    } apply {
                        // arXiv:2605.30455 FIG. 40. Implementation of a controlled RZ(2θ)
                        // gate using two parallel RZ(θ) gates without controls.
                        within {
                            CNOT(sysRegDown[j + 1], sysRegDown[j]);
                        } apply {
                            Adjoint RyViaPhaseGradient(sysRegDown[j], word, phaseGradientReg);
                        }
                        RyViaPhaseGradient(sysRegDown[j], word, phaseGradientReg);
                    }
                }
            } apply {
                action();
            }
        }
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
        majoranaSel : Qubit,
        system_reg_0 : Qubit
    ) : Unit is Adj + Ctl {
        // SF two-body (b < B): Z on system_reg_0 when sf_vs_dq=1 AND bEqB=0
        within { X(bEqB); } apply {
            Controlled Z([sf_vs_dq, bEqB], system_reg_0);
        }
        // DQ: a_k = (X + iY)/2 as a two-term LCU on majoranaSel, which the inner
        // reflection projects. X on the |0> branch, ZX = iY on the |1> branch.
        within { X(sf_vs_dq); } apply {
            CNOT(sf_vs_dq, system_reg_0);
            Controlled Z([sf_vs_dq, majoranaSel], system_reg_0);
        }
        // Q1 uses a_k^dagger = (X - iY)/2, so the sign flip belongs on the branch that
        // carries the iY term -- the Majorana selector, not the spin selector.
        within { X(sf_vs_dq); } apply {
            Controlled Z([sf_vs_dq, d_vs_q], majoranaSel);
        }
    }


    // ═══════════════════════════════════════════════════════════════════════════
    // Factories and circuit entry points
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build an inner PREPARE using conditional alias sampling (2D QROM).
    ///
    /// Uses ConditionalAliasSamplingPrepareWithFreeRider to prepare:
    ///   |x_o⟩|0⟩ → |x_o⟩ Σ_b √(p̃_{x_o,b}) e^{iπ·sign} |b⟩|garbage⟩
    ///
    /// Pass `freeRiderData = []` to leave the free-rider word to `MakeFreeRiderLoadOp`. That
    /// pays off only when the lookup takes the select-swap path, where the word widens the
    /// QROAM output the swap network is charged for, four times per block encoding. On the
    /// unary-iteration path the cost does not depend on the output width, so carrying it here
    /// is free and a separate load would be pure overhead.
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
        // A single inner entry still gets a one-qubit b register, matching MakeInnerPrepareDirect
        // and the Python layout. Letting this fall to zero would leave the alias PREPARE treating
        // innerReg[0] as its uniform register while SELECT treats it as b.
        let nIndexBits = BitSizeI((if nCoeffs > 1 { nCoeffs } else { 2 }) - 1);
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

    /// Load the free-rider word (G, r) for the current x_o.
    ///
    /// It is a function of x_o alone, so the block encoding loads it once around both SELECT
    /// calls rather than letting each inner PREPARE carry it: one `Select` round trip against
    /// four widened QROAM round trips.
    function MakeFreeRiderLoadOp(freeRiderData : Bool[][]) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, freeRiderReg) => {
            if Length(freeRiderData) > 0 and Length(freeRiderReg) > 0 {
                Select(freeRiderData, outerReg, freeRiderReg);
            }
        }
    }

    /// Build the inner alias-sampling PREPARE and its free-rider loader together.
    ///
    /// Carrying the free-rider word widens the QROAM output charged on each of the four
    /// inner-table lookups in one block encoding. Loading it separately costs one `Select`
    /// round trip over the outer conditions but lets the inner table use a narrower output
    /// and potentially a different swap width.
    function MakeInnerPrepareAliasSamplingOracles(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
    ) : (
        ((Qubit[], Qubit[]) => Unit is Adj),
        ((Qubit[], Qubit[]) => Unit is Adj + Ctl)
    ) {
        let numConditions = Length(innerCoefficients);
        let numInnerSlots = 1 <<< BitSizeI(Length(innerCoefficients[0]) - 1);
        let numWordBits = coefficientBitPrecision + BitSizeI(numInnerSlots - 1) + 2;
        let numExtraBits = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        let inlineBits = numWordBits + numExtraBits;
        let inlineLambda = ComputeOptimalLambda2D(numConditions, numInnerSlots, inlineBits, true);
        let separateLambda = ComputeOptimalLambda2D(numConditions, numInnerSlots, numWordBits, true);
        let inlineCost = 4 * SelectSwapCost2D(inlineLambda, numConditions, numInnerSlots, inlineBits, true);
        let separateCost = 4 * SelectSwapCost2D(
            separateLambda,
            numConditions,
            numInnerSlots,
            numWordBits,
            true
        ) + SelectSwapCost2D(0, numConditions, 1, 1, true);
        let loadSeparately = numExtraBits > 0 and separateCost < inlineCost;
        let inlineData = if loadSeparately { [] } else { freeRiderData };
        let separateData = if loadSeparately { freeRiderData } else { [] };

        (
            MakeInnerPrepareAliasSampling(innerCoefficients, inlineData, coefficientBitPrecision),
            MakeFreeRiderLoadOp(separateData)
        )
    }

    /// Build an inner PREPARE using direct controlled preparation.
    ///
    /// innerReg layout: bReg[nIndexBits] + signQubit[1] + freeRiderReg[nFR]. The sign qubit
    /// mirrors the alias-sampling QROM's sign output, so SELECT finds the LCU sign of the
    /// sampled `(x_o, b)` in the same kind of place whichever inner backend is in use.
    function MakeInnerPrepareDirect(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][]
    ) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        let nCoeffs = Length(innerCoefficients[0]);
        let nIndexBits = BitSizeI((if nCoeffs > 1 { nCoeffs } else { 2 }) - 1);
        let signData = BuildInnerSignTable(innerCoefficients, nIndexBits);
        (outerReg, innerReg) => {
            let bReg = innerReg[0..nIndexBits - 1];

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
            Select(signData, bReg + outerReg, [innerReg[nIndexBits]]);
        }
    }

    /// Build a SELECT using QROM + phase gradient rotation.
    function MakeSelectPhaseGradient(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg, spinReg, systemReg, phaseGradientReg) => {
            SelectImpl(params, true, outerReg, innerReg, spinReg, systemReg, phaseGradientReg);
        }
    }

    /// Build a SELECT using direct rotation synthesis.
    function MakeSelectDirectRotation(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg, spinReg, systemReg, phaseGradientReg) => {
            SelectImpl(params, false, outerReg, innerReg, spinReg, systemReg, phaseGradientReg);
        }
    }

    /// The SOSSA block encoding B as the `Qubit[] => Unit is Adj` callable QPE consumes.
    ///
    /// Register layout: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    /// the gradient tail is only present when `layout.numPhaseGradientQubits > 0`.
    function MakeSOSSABlockEncodingOp(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
    ) : (Qubit[] => Unit is Adj) {
        SOSSABlockEncodingOnRegister(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, _)
    }

    /// Circuit entry point: allocates the flat register and applies the block encoding once.
    /// Register layout: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    operation MakeSOSSABlockEncodingCircuit(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
    ) : Unit {
        let bounds = SOSSAWalkRegisterBounds(layout);
        use allQubits = Qubit[layout.numSystemQubits + bounds[4] - bounds[0]];
        if layout.numPhaseGradientQubits > 0 {
            let gradientStart = Length(allQubits) - layout.numPhaseGradientQubits;
            PreparePhaseGradientState(allQubits[gradientStart...]);
        }
        SOSSABlockEncodingOnRegister(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, allQubits);
        ResetAll(allQubits);
    }


    // ═══════════════════════════════════════════════════════════════════════════
    // Test wrappers
    // ═══════════════════════════════════════════════════════════════════════════

    /// Adapter: outer PREPARE then inner PREPARE, over one flat register.
    function MakeOuterInnerPrepOp(
        outerOp : (Qubit[]) => Unit is Adj + Ctl,
        innerOp : (Qubit[], Qubit[]) => Unit is Adj,
        nOuter : Int,
    ) : Qubit[] => Unit {
        (qs) => {
            let outerReg = qs[0..nOuter - 1];
            outerOp(outerReg);
            innerOp(outerReg, qs[nOuter...]);
        }
    }


    /// Test the full SELECT on an entry with known angles.
    operation TestSelectDQ(
        selectData : SelectParams,
        xoValue : Int,
        bValue : Int,
        usePhaseGradient : Bool,
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
        // The gradient register is allocated only on the QROM path. It is conjugated back to
        // |0...0>, so the direct path's state is recovered from the QROM dump by restricting
        // to the gradient=|0...0> subspace.
        let nGradient = if usePhaseGradient { selectData.rotationBitPrecision } else { 0 };
        let total = nOuter + nInner + nSpin + nSystem + nGradient;
        let qs = QIR.Runtime.AllocateQubitArray(total);

        let outerReg = qs[0..nOuter - 1];
        let innerReg = qs[nOuter..nOuter + nInner - 1];
        let spinReg = qs[nOuter + nInner..nOuter + nInner + nSpin - 1];
        let systemReg = qs[nOuter + nInner + nSpin..nOuter + nInner + nSpin + nSystem - 1];
        let gradientReg = qs[total - nGradient...];

        let xoReg = outerReg[0..xoBits - 1];
        ApplyXorInPlace(xoValue, xoReg);
        H(spinReg[0]); // spinDQ

        ApplyXorInPlace(bValue, innerReg[0..bBits - 1]);

        let frStart = bBits;
        if nFR >= 2 {
            if xoValue >= N { X(innerReg[frStart]); }
            if xoValue >= numPositiveOneBody { X(innerReg[frStart + 1]); }

            // Rank as the inner PREPARE's free-rider data would carry it: 0 for one-body.
            let rValue = if xoValue >= N { (xoValue - N) / selectData.numCopies } else { 0 };
            ApplyXorInPlace(rValue, innerReg[frStart + 2..frStart + nFR - 1]);
        }

        X(systemReg[0]);

        if usePhaseGradient {
            // Conjugated so the gradient register returns to |0...0> and does not contribute
            // its own amplitudes to the comparison against the direct path.
            within {
                PreparePhaseGradientState(gradientReg);
            } apply {
                SelectImpl(selectData, true, outerReg, innerReg, spinReg, systemReg, gradientReg);
            }
        } else {
            SelectImpl(selectData, false, outerReg, innerReg, spinReg, systemReg, []);
        }
    }


}
