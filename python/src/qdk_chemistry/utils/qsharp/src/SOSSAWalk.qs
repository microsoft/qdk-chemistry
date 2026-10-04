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

    import Std.Arrays.All;
    import Std.Arrays.Flattened;
    import Std.Arrays.ForEach;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Padded;
    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyToEachCA;
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.IntAsDouble;
    import Std.Convert.ResultArrayAsBoolArray;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;

    import Std.Math.AbsD;
    import Std.Math.MaxI;
    import Std.Math.MinI;
    import Std.Math.PI;
    import Std.Math.Round;
    import Std.Measurement.MeasureEachZ;
    import Std.Measurement.MResetX;
    import Std.StatePreparation.PreparePureStateD;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepareWithFreeRider;
    import Std.Arithmetic.RippleCarryCGIncByLE;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState, QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;
    import QDKChemistry.Utils.SelectSwap.ApplyBranchPhaseFixup, QDKChemistry.Utils.SelectSwap.ComputeOptimalLambda2D, QDKChemistry.Utils.SelectSwap.SelectSwapCost2D;
    import QDKChemistry.Utils.SelectSwap.ComputeOptimalSwapBits, QDKChemistry.Utils.SelectSwap.SelectSwapAliased;
    import QDKChemistry.Utils.SelectSwapDirty.ComputeOptimalDirtySwapBits, QDKChemistry.Utils.SelectSwapDirty.DirtyQROAMBorrowedQubits, QDKChemistry.Utils.SelectSwapDirty.SelectSwapDirty;
    import QDKChemistry.Utils.SelectSwap.LookupSelect, QDKChemistry.Utils.SelectSwap.LookupSelectSwap, QDKChemistry.Utils.SelectSwap.LookupDirtySelectSwap;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;

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
        /// Number of Givens angles held in the rotation register at once (the paper's lambda).
        /// Zero, or any value at or above `N - 1`, keeps the whole angle word resident, which is
        /// the fastest setting and the historical behaviour. Smaller values stream the angles in
        /// batches, trading Toffolis for a rotation register of `rotationBatchSize * b_rot`
        /// qubits instead of `(N - 1) * b_rot`.
        /// :cite:`Low2026` Appendix E 3.
        rotationBatchSize : Int,
        /// Which loader a streamed rotation batch uses: `LookupSelect()`,
        /// `LookupSelectSwap()`, or `LookupDirtySelectSwap()`.
        /// Only consulted when the angles are actually streamed; a resident angle word is
        /// already wide enough that no swap network fits.
        rotationLookupMethod : Int,
        /// Ceiling on the swap width any QROAM in the walk may take. -1 leaves each loader's
        /// own selector alone; 0 forces plain unary-iteration lookups; a positive value caps
        /// whatever the selector would have chosen. Capping only reroutes identical data, so
        /// it trades Toffolis for width and never touches the state prepared.
        maxSwapBits : Int,
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
        BuildDQRotationBatch(params, N, 0, numRotAngles - 1, bRot)
    }

    /// Angles `first..last` of the DQ table, quantized and concatenated low-index first.
    ///
    /// `BuildDQBulkRotationData` is the whole-range case; a streamed batch asks for a window.
    internal function BuildDQRotationBatch(
        params : SelectParams,
        N : Int,
        first : Int,
        last : Int,
        bRot : Int,
    ) : Bool[][] {
        MappedOverRange(
            xo -> Flattened(
                MappedOverRange(
                    j -> IntAsBoolArray(
                        QuantizeGivensAngle(
                            if j < Length(params.OneBodyRotationAngles[xo]) {
                                params.OneBodyRotationAngles[xo][j]
                            } else {
                                0.0
                            },
                            bRot
                        ),
                        bRot
                    ),
                    first..last
                )
            ),
            0..N - 1
        )
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
            let angleIdx = b * params.numRanks + r;

            let angleBits = Flattened(
                MappedOverRange(
                    j -> IntAsBoolArray(
                        QuantizeGivensAngle(
                            if r < R and angleIdx < Length(params.TwoBodyRotationAngles) and j < Length(params.TwoBodyRotationAngles[angleIdx]) {
                                params.TwoBodyRotationAngles[angleIdx][j]
                            } else {
                                0.0
                            },
                            bRot
                        ),
                        bRot
                    ),
                    0..numRotAngles - 1
                )
            );
            // Append bEqB flag: true when b == numBases (the identity term)
            set table += [angleBits + [b == params.numBases]];
        }
        return table;
    }

    /// Angles `first..last` of the SF table, without the trailing `bEqB` flag.
    ///
    /// The bulk table carries that flag for free, because unary iteration costs the same at any
    /// output width. A streamed batch is loaded once per batch, so the flag is set arithmetically
    /// by the caller instead of being paid for on every batch.
    internal function BuildSFRotationBatch(
        params : SelectParams,
        R : Int,
        first : Int,
        last : Int,
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
            let angleIdx = b * params.numRanks + r;

            set table += [
                Flattened(
                    MappedOverRange(
                        j -> IntAsBoolArray(
                            QuantizeGivensAngle(
                                if r < R and angleIdx < Length(params.TwoBodyRotationAngles) and j < Length(params.TwoBodyRotationAngles[angleIdx]) {
                                    params.TwoBodyRotationAngles[angleIdx][j]
                                } else {
                                    0.0
                                },
                                bRot
                            ),
                            bRot
                        ),
                        first..last
                    )
                )
            ];
        }
        return table;
    }

    /// Number of Givens angles a single streamed batch holds, clamped to the sane range.
    ///
    /// Returns `numRotAngles` — a single resident batch, the historical behaviour — for the
    /// zero/negative/oversized settings.
    internal function RotationBatchSize(params : SelectParams, numRotAngles : Int) : Int {
        let requested = params.rotationBatchSize;
        if requested <= 0 or requested >= numRotAngles {
            numRotAngles
        } else {
            requested
        }
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
        let xoBits = MaxI(1, AddressQubits(Xo));
        let numBp1 = params.numBases + 1;
        let bBits = MaxI(1, AddressQubits(numBp1));
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
                    sysRegUp,
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
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
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
                        // `systemReg` is lent to the inner PREPARE's QROAM: SELECT is the only
                        // thing that touches the wavefunction, and it runs in the `apply` below,
                        // so these qubits are provably idle for the whole of PREPARE. A dirty
                        // load hands them back untouched, so the `within` uncompute still sees
                        // exactly the state it would have seen from a clean load.
                        innerPrepareOp(outerIndexReg, innerReg, systemReg);
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
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
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

    /// Loads the branch-selected Givens angle word; the adjoint erases it by measurement.
    /// Both tables write the same `target` and measuring consumes it, so the adjoint measures once
    /// and repairs each branch's phase; letting `within` uncompute instead costs a second lookup.
    ///
    /// `lookupMethod` picks the loader and the two widths size it, per table. A width of 0 always
    /// means a plain unary-iteration `Select`, whichever method asked for it, so a cost model that
    /// declines its network degrades to the cheapest correct thing. The widths are separate
    /// because the two tables differ in row count, often by an order of magnitude, so one shared
    /// width would hold the larger table to the smaller one's optimum.
    ///
    /// All three loaders leave `target` in exactly the state a plain `Select` would, including at
    /// surplus addresses, which is why the measurement-based adjoint below is shared unchanged.
    internal operation ControlledSelectWithUnlookup(
        sfData : Bool[][],
        sfAddress : Qubit[],
        dqData : Bool[][],
        dqAddress : Qubit[],
        isSF : Qubit,
        dirty : Qubit[],
        lookupMethod : Int,
        sfSwapBits : Int,
        dqSwapBits : Int,
        target : Qubit[],
    ) : Unit is Adj {
        body (...) {
            // `Select` tolerates a table length that is not a power of two -- it never reads an
            // address at or above `Length(data)` -- so both tables are used as built.
            LoadRotationWord(lookupMethod, sfSwapBits, sfData, sfAddress, isSF, dirty, target);
            let dqTarget = target[0..Length(dqData[0]) - 1];
            within { X(isSF); } apply {
                LoadRotationWord(lookupMethod, dqSwapBits, dqData, dqAddress, isSF, dirty, dqTarget);
            }
        }
        adjoint (...) {
            let measured = ResultArrayAsBoolArray(ForEach(MResetX, target));
            // Each branch contributes the parity of its own word, and the two contributions
            // compose, so the fixups are independent and their order does not matter.
            ApplyBranchPhaseFixup(measured, sfData, true, [isSF] + sfAddress);
            ApplyBranchPhaseFixup(measured, dqData, false, [isSF] + dqAddress);
        }
    }

    /// One branch's forward load, controlled on `isSF`, using the selected loader.
    internal operation LoadRotationWord(
        lookupMethod : Int,
        numSwapBits : Int,
        data : Bool[][],
        address : Qubit[],
        isSF : Qubit,
        dirty : Qubit[],
        target : Qubit[],
    ) : Unit is Adj + Ctl {
        if numSwapBits == 0 {
            Controlled Select([isSF], (data, address, target));
        } elif lookupMethod == LookupSelectSwap() {
            Controlled SelectSwapAliased([isSF], (numSwapBits, data, address, target));
        } else {
            Controlled SelectSwapDirty([isSF], (numSwapBits, data, address, dirty, target));
        }
    }

    /// Applies the Givens basis change from a QROM-loaded angle word, runs `action` in that
    /// rotated basis, then undoes the chain and releases the angle word.
    ///
    /// Loads ALL (N-1) rotation angles at once using two Select calls:
    ///   - SF: Select over min(R*2^bBits, (B+1)*2^rankBits) entries, fires when isSF=1
    ///   - DQ: Select(N entries) addressed by xoReg[0..⌈log₂N⌉-1], fires when isSF=0
    ///
    /// Cost: each table is loaded once, by a controlled `Select` that fires only on its own
    /// branch, so the SF table contributes L_SF - 2 and the DQ table N - 1. Both are then erased
    /// together by a single measurement plus one phase fixup per branch, at O(sqrt(2*L_SF)) and
    /// O(sqrt(2*N)). The rotations add 2*(N-1) uncontrolled
    /// Adder(bRot) -- each Givens rotation is the neighbor-gated CRy(2θ) built as
    /// Ry(θ)·CNOT·Ry(-θ)·CNOT from two uncontrolled Ry(θ) sharing one angle word, so it
    /// preserves particle number (matching the direct path) without a controlled adder.
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
        sysRegUp : Qubit[],
        phaseGradientReg : Qubit[],
        bEqBQubit : Qubit,
        action : (Unit => Unit is Adj + Ctl),
    ) : Unit is Adj + Ctl {
        let bRot = params.rotationBitPrecision;
        let bBits = Length(bReg);
        let R = params.numRanks;
        let nDQBits = MaxI(1, AddressQubits(N));
        let dqAddress = xoReg[0..nDQBits - 1];

        // SF table: addressed by (bReg ++ rBits) or (rBits ++ bReg), whichever is smaller.
        let rankFirst = SFTableRankAddressedFirst(R, params.numBases, bBits, Length(rBits));
        let sfAddress = if rankFirst { rBits + bReg } else { bReg + rBits };

        let batch = RotationBatchSize(params, numRotAngles);

        if batch >= numRotAngles {
            let nRotBits = numRotAngles * bRot;
            let dqData = BuildDQBulkRotationData(params, N, numRotAngles, bRot);
            let sfData = BuildSFBulkRotationData(params, R, numRotAngles, bRot, bBits, Length(rBits), rankFirst);

            // Allocate rotation target register: (N-1)*bRot rotation bits + 1 bEqB flag bit.
            use rotTarget = Qubit[nRotBits + 1];

            within {
                // The resident word is already the full angle table, so no loader choice applies:
                // widths of 0 pin both branches to the plain lookup whatever the method says.
                ControlledSelectWithUnlookup(
                    sfData,
                    sfAddress,
                    dqData,
                    dqAddress,
                    isSF,
                    [],
                    LookupSelect(),
                    0,
                    0,
                    rotTarget
                );
            } apply {
                within {
                    CNOT(rotTarget[nRotBits], bEqBQubit);
                    ApplyGivensRotationWords(rotTarget, 0, numRotAngles - 1, bRot, sysRegDown, phaseGradientReg);
                } apply {
                    action();
                }
            }
        } else {
            let numBatches = (numRotAngles + batch - 1) / batch;
            within {
                // The bEqB flag rides the bulk table for free, but a streamed table would pay for
                // it once per batch, so it is computed directly from b and isSF instead.
                ApplyControlledOnInt(params.numBases, q => Controlled X([isSF], q), bReg, bEqBQubit);

                // Descending batch order keeps the global angle order of the resident path: the
                // rotation chain must run from the highest index down to zero.
                for t in numBatches - 1..-1..0 {
                    let first = t * batch;
                    let last = MinI(first + batch, numRotAngles) - 1;
                    ApplyGivensRotationBatch(
                        params,
                        N,
                        R,
                        first,
                        last,
                        bRot,
                        bBits,
                        rankFirst,
                        isSF,
                        dqAddress,
                        sfAddress,
                        sysRegDown,
                        sysRegUp,
                        phaseGradientReg
                    );
                }
            } apply {
                action();
            }
        }
    }

    /// Applies the neighbour-gated Givens rotations for angles `first..last`, reading the angle
    /// words out of `rotTarget` (which holds exactly that window, low index first).
    internal operation ApplyGivensRotationWords(
        rotTarget : Qubit[],
        first : Int,
        last : Int,
        bRot : Int,
        sysRegDown : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj + Ctl {
        for j in last..-1..first {
            let word = rotTarget[(j - first) * bRot..(j - first + 1) * bRot - 1];
            within {
                CNOT(sysRegDown[j], sysRegDown[j + 1]);
            } apply {
                // :cite:`Low2026` FIG. 40. Implementation of a controlled RZ(2θ)
                // gate using two parallel RZ(θ) gates without controls.
                within {
                    CNOT(sysRegDown[j + 1], sysRegDown[j]);
                } apply {
                    Adjoint RyViaPhaseGradient(sysRegDown[j], word, phaseGradientReg);
                }
                RyViaPhaseGradient(sysRegDown[j], word, phaseGradientReg);
            }
        }
    }

    /// Loads one streamed window of Givens angles, applies its rotations, and releases the word.
    ///
    /// The rotations survive the release because they act on `sysRegDown`, so only
    /// `(last - first + 1) * bRot` angle qubits are ever resident instead of `(N - 1) * bRot`.
    /// The cost is that the window has to be looked up again on the way out, which `within`
    /// arranges when the caller's conjugation is inverted.
    ///
    /// Qubits lent to the swap network are the spin-up half of the wavefunction plus the part of
    /// the spin-down half this window does not rotate — live data that `SelectSwapDirty` restores
    /// exactly. :cite:`Low2026` Appendix E 3 notes these registers as the dirty-ancilla source.
    internal operation ApplyGivensRotationBatch(
        params : SelectParams,
        N : Int,
        R : Int,
        first : Int,
        last : Int,
        bRot : Int,
        bBits : Int,
        rankFirst : Bool,
        isSF : Qubit,
        dqAddress : Qubit[],
        sfAddress : Qubit[],
        sysRegDown : Qubit[],
        sysRegUp : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj + Ctl {
        let width = last - first + 1;
        let m = width * bRot;
        let dqData = BuildDQRotationBatch(params, N, first, last, bRot);
        let sfData = BuildSFRotationBatch(params, R, first, last, bRot, bBits, Length(sfAddress) - bBits, rankFirst);

        // Angles first..last rotate sysRegDown[first..last+1]; everything else may be borrowed.
        let untouched = sysRegDown[0..first - 1] + sysRegDown[last + 2..N - 1];
        let dirty = sysRegUp + untouched;
        // Each table is sized on its own word width: the SF word spans the whole target but the
        // DQ word is only a prefix of it, and both the scratch a clean network allocates and the
        // block a dirty one borrows scale with that width.
        let sfSwapBits = RotationSwapWidth(
            params.rotationLookupMethod,
            Length(sfData),
            Length(sfData[0]),
            Length(dirty),
            params.maxSwapBits
        );
        let dqSwapBits = RotationSwapWidth(
            params.rotationLookupMethod,
            Length(dqData),
            Length(dqData[0]),
            Length(dirty),
            params.maxSwapBits
        );

        use rotTarget = Qubit[m];
        within {
            ControlledSelectWithUnlookup(
                sfData,
                sfAddress,
                dqData,
                dqAddress,
                isSF,
                dirty,
                params.rotationLookupMethod,
                sfSwapBits,
                dqSwapBits,
                rotTarget
            );
        } apply {
            ApplyGivensRotationWords(rotTarget, first, last, bRot, sysRegDown, phaseGradientReg);
        }
    }

    /// Swap width one streamed batch table should use under the selected lookup method.
    ///
    /// Each method's own cost model decides, and each returns 0 when no network beats the plain
    /// lookup, so an unprofitable shape falls back to `Select` instead of paying for scratch or
    /// borrowing it cannot use. `maxSwapBits` then caps that choice: -1 leaves it alone, and any
    /// other value is an upper bound, so the cap can only ever narrow the register.
    internal function RotationSwapWidth(
        lookupMethod : Int,
        numData : Int,
        numBits : Int,
        availableDirty : Int,
        maxSwapBits : Int,
    ) : Int {
        let selected = if lookupMethod == LookupSelectSwap() {
            ComputeOptimalSwapBits(numData, numBits)
        } elif lookupMethod == LookupDirtySelectSwap() {
            ComputeOptimalDirtySwapBits(numData, numBits, availableDirty)
        } else {
            0
        };
        if maxSwapBits < 0 { selected } else { MinI(selected, maxSwapBits) }
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
    ///
    /// `maxSwapBits` caps the QROAM swap width: -1 leaves the loader's own selector alone, 0
    /// forces a plain unary-iteration load, and a positive value is an upper bound on whatever
    /// the selector chose. The swap network allocates scratch proportional to `2^k` times the
    /// loaded word, which is where a qubit-limited caller wants a say.
    function MakeInnerPrepareAliasSampling(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
        maxSwapBits : Int,
        lookupMethod : Int,
    ) : (Qubit[], Qubit[], Qubit[]) => Unit is Adj {
        let nCoeffs = Length(innerCoefficients[0]);
        // A single inner entry still gets a one-qubit b register, matching MakeInnerPrepareDirect
        // and the Python layout. Letting this fall to zero would leave the alias PREPARE treating
        // innerReg[0] as its uniform register while SELECT treats it as b.
        let nIndexBits = MaxI(1, AddressQubits(nCoeffs));
        let mu = coefficientBitPrecision;
        let nFreeRider = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        let qromEnd = 2 * nIndexBits + 2 * mu + 2;
        // A swap network cannot be wider than the table it routes, and `SwappedLoadShape`
        // asserts as much. Clamp rather than fault, so an over-large request degrades to the
        // widest usable network instead of crashing inside Q#.
        let swapBits = if maxSwapBits > 0 { MinI(maxSwapBits, nIndexBits) } else { maxSwapBits };
        (outerReg, innerReg, dirty) => {
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
                freeRiderReg,
                swapBits,
                dirty,
                lookupMethod
            );
        }
    }

    /// Load the free-rider word (G, r) for the current x_o.
    ///
    /// It is a function of x_o alone, so the block encoding loads it once around both inner
    /// PREPARE/uncompute pairs rather than letting each pair carry it in the alias QROAM output.
    function MakeFreeRiderLoadOp(freeRiderData : Bool[][]) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, freeRiderReg) => {
            if Length(freeRiderData) > 0 and Length(freeRiderReg) > 0 {
                Select(freeRiderData, outerReg, freeRiderReg);
            }
        }
    }

    /// Whether the free-rider word should be loaded separately from the inner alias tables.
    ///
    /// `SelectSwapCost2D` already covers one lookup/uncompute pair. One SOSSA block applies
    /// two inner PREPARE/uncompute pairs and, if split out, one free-rider lookup pair.
    ///
    /// `maxSwapBits` must cap what the lookups will actually use, or this compares the cost
    /// of two layouts neither of which gets built.
    internal function ShouldLoadFreeRiderSeparately(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
        maxSwapBits : Int,
    ) : Bool {
        let numConditions = Length(innerCoefficients);
        let nIndexBits = MaxI(1, AddressQubits(Length(innerCoefficients[0])));
        let numInnerSlots = 1 <<< nIndexBits;
        let numWordBits = coefficientBitPrecision + nIndexBits + 2;
        let numExtraBits = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        let inlineBits = numWordBits + numExtraBits;
        // Mirrors the cap in `MakeInnerPrepareAliasSampling`, including its clamp to the table's
        // address width, so this compares the cost of the layouts that will actually be built.
        let cap = MinI(MaxI(0, maxSwapBits), nIndexBits);
        let inlineSelected = ComputeOptimalLambda2D(numConditions, numInnerSlots, inlineBits, true);
        let separateSelected = ComputeOptimalLambda2D(numConditions, numInnerSlots, numWordBits, true);
        let inlineLambda = if maxSwapBits < 0 { inlineSelected } else { MinI(inlineSelected, cap) };
        let separateLambda = if maxSwapBits < 0 { separateSelected } else { MinI(separateSelected, cap) };
        let innerPreparePairsPerBlock = 2;
        let freeRiderPairsPerBlock = 1;
        let inlineCost = innerPreparePairsPerBlock * SelectSwapCost2D(
            inlineLambda,
            numConditions,
            numInnerSlots,
            inlineBits,
            true
        );
        let separateCost = innerPreparePairsPerBlock * SelectSwapCost2D(
            separateLambda,
            numConditions,
            numInnerSlots,
            numWordBits,
            true
        ) + freeRiderPairsPerBlock * SelectSwapCost2D(0, numConditions, 1, numExtraBits, true);
        numExtraBits > 0 and separateCost < inlineCost
    }

    /// Build the inner alias-sampling PREPARE and its free-rider loader together.
    ///
    /// Carrying the free-rider word widens the QROAM output charged on each of the two inner
    /// PREPARE/uncompute pairs in one block encoding. Loading it separately costs one `Select`
    /// round trip over the outer conditions but lets the inner table use a narrower output
    /// and potentially a different swap width.
    function MakeInnerPrepareAliasSamplingOracles(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
        maxSwapBits : Int,
        lookupMethod : Int,
    ) : (
        ((Qubit[], Qubit[], Qubit[]) => Unit is Adj),
        ((Qubit[], Qubit[]) => Unit is Adj + Ctl)
    ) {
        let loadSeparately = ShouldLoadFreeRiderSeparately(
            innerCoefficients,
            freeRiderData,
            coefficientBitPrecision,
            maxSwapBits
        );
        let inlineData = if loadSeparately { [] } else { freeRiderData };
        let separateData = if loadSeparately { freeRiderData } else { [] };

        (
            MakeInnerPrepareAliasSampling(
                innerCoefficients,
                inlineData,
                coefficientBitPrecision,
                maxSwapBits,
                lookupMethod
            ),
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
    ) : (Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        let nCoeffs = Length(innerCoefficients[0]);
        let nIndexBits = MaxI(1, AddressQubits(nCoeffs));
        let signData = BuildInnerSignTable(innerCoefficients, nIndexBits);
        // Direct preparation has no lookup table, so there is nothing to borrow for; the
        // lender is accepted and ignored purely to keep one inner-PREPARE signature.
        (outerReg, innerReg, _) => {
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
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
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
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
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
    ///
    /// Lends the inner PREPARE nothing, so this exercises the clean lookup. The fidelity it
    /// checks is the state on `qs`, which a borrowed load leaves alone by construction.
    function MakeOuterInnerPrepOp(
        outerOp : (Qubit[]) => Unit is Adj + Ctl,
        innerOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        nOuter : Int,
    ) : Qubit[] => Unit {
        (qs) => {
            let outerReg = qs[0..nOuter - 1];
            outerOp(outerReg);
            innerOp(outerReg, qs[nOuter...], []);
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
        let xoBits = MaxI(1, AddressQubits(Xo));
        let numBp1 = selectData.numBases + 1;
        let bBits = MaxI(1, AddressQubits(numBp1));
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

    function TestShouldLoadFreeRiderSeparately(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
        maxSwapBits : Int,
    ) : Bool {
        ShouldLoadFreeRiderSeparately(
            innerCoefficients,
            freeRiderData,
            coefficientBitPrecision,
            maxSwapBits
        )
    }

    /// Checks that loading and then erasing the branched angle word leaves the address
    /// registers untouched, including the relative phase between the two branches.
    ///
    /// Conjugating by H turns any phase that is not global into a nonzero measurement, which is
    /// what a fixup table that disagreed with the forward load would produce. Tables whose row
    /// count is not a power of two are the case worth covering: `Select` aliases the surplus
    /// addresses onto real rows instead of leaving them unloaded.
    ///
    /// The borrowed register is conjugated along with the addresses, so a dirty forward pass that
    /// returned it in a different state -- or merely entangled with the angle word -- fails here
    /// too. Swap width 0 runs the plain loader for that branch, whatever `lookupMethod` says.
    operation TestBranchedRotationWordRoundTrip(
        sfData : Bool[][],
        dqData : Bool[][],
        numSFAddressQubits : Int,
        numDQAddressQubits : Int,
        lookupMethod : Int,
        sfSwapBits : Int,
        dqSwapBits : Int,
    ) : Bool {
        use isSF = Qubit();
        use sfAddress = Qubit[numSFAddressQubits];
        use dqAddress = Qubit[numDQAddressQubits];
        use target = Qubit[Length(sfData[0])];
        // Only a dirty load borrows; the clean paths allocate their own scratch internally.
        let numDirty = if lookupMethod == LookupDirtySelectSwap() {
            MaxI(
                if sfSwapBits == 0 { 0 } else { DirtyQROAMBorrowedQubits(sfSwapBits, Length(sfData[0])) },
                if dqSwapBits == 0 { 0 } else { DirtyQROAMBorrowedQubits(dqSwapBits, Length(dqData[0])) }
            )
        } else {
            0
        };
        use dirty = Qubit[numDirty];
        let addressReg = [isSF] + sfAddress + dqAddress;
        let conjugated = addressReg + dirty;

        ApplyToEachCA(H, conjugated);
        ControlledSelectWithUnlookup(
            sfData,
            sfAddress,
            dqData,
            dqAddress,
            isSF,
            dirty,
            lookupMethod,
            sfSwapBits,
            dqSwapBits,
            target
        );
        Adjoint ControlledSelectWithUnlookup(
            sfData,
            sfAddress,
            dqData,
            dqAddress,
            isSF,
            dirty,
            lookupMethod,
            sfSwapBits,
            dqSwapBits,
            target
        );
        ApplyToEachCA(H, conjugated);

        let results = MeasureEachZ(conjugated + target);
        ResetAll(conjugated + target);
        All(result -> result == Zero, results)
    }

}
