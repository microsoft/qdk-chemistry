// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Sum of Squares Spectral Amplification (SOSSA) walk operator.
///
/// Composable design: each sub-operation (OuterPrepare, InnerPrepare, Select)
/// is built independently as a Q# callable, and this module assembles them into
/// the block encoding B that unary-iteration phase estimation queries, and into the
/// controlled walk that standard and iterative phase estimation query.
///
/// Walk operator (Low et al., Phys. Rev. X 15 (2025), inline above Eq. (9); App. A 2):
///   W = Ref_{a,B} · B,  B = U† · Ref_B · U
/// where U = OuterPREP · within{InnerPREP} apply{SELECT}.

namespace QDKChemistry.Utils.SOSSAWalk {

    import Std.Arrays.All;
    import Std.Arrays.Any;
    import Std.Arrays.Chunks;
    import Std.Arrays.Filtered;
    import Std.Arrays.Flattened;
    import Std.Arrays.Fold;
    import Std.Arrays.ForEach;
    import Std.Arrays.IndexRange;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.Mapped;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Padded;
    import Std.Arrays.Reversed;
    import Std.Arrays.Partitioned;
    import Std.Arrays.Subarray;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyToEachA;
    import Std.Canon.ApplyToEachCA;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.IntAsDouble;
    import Std.Convert.ResultArrayAsBoolArray;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;

    import Std.Math.AbsD;
    import Std.Math.ArcTan2;
    import Std.Math.Ceiling;
    import Std.Math.Cos;
    import Std.Math.Lg;
    import Std.Math.MaxI;
    import Std.Math.MinI;
    import Std.Math.PI;
    import Std.Math.Round;
    import Std.Math.Sin;
    import Std.Math.Sqrt;
    import Std.Measurement.MResetX;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import Std.ResourceEstimation.SingleVariant;
    import Std.StatePreparation.PreparePureStateD;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepareWithFreeRider, QDKChemistry.Utils.AliasSampling.ConditionalAliasSwapBits;
    import Std.Arithmetic.ApplyIfLessOrEqualL;
    import Std.Arithmetic.IncByLUsingIncByLE;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepareOneHot;
    import Std.Arithmetic.RippleCarryCGIncByLE;
    import Std.Convert.IntAsBigInt;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState, QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;
    import QDKChemistry.Utils.SelectSwap.ApplyBranchPhaseFixup, QDKChemistry.Utils.SelectSwap.SelectSwapCost2D;
    import QDKChemistry.Utils.SelectSwap.ComputeOptimalSwapBits, QDKChemistry.Utils.SelectSwap.SelectSwapAliased;
    import QDKChemistry.Utils.SelectSwap.SelectSwapForwardCost, QDKChemistry.Utils.SelectSwapDirty.SelectSwapDirtyCost1D;
    import QDKChemistry.Utils.SelectSwap.AliasToAddressSpace, QDKChemistry.Utils.SelectSwap.DimensionsForSelect;
    import QDKChemistry.Utils.SelectSwap.MaxSelectableSwapBits, QDKChemistry.Utils.SelectSwap.SwapDataOutputs;
    import QDKChemistry.Utils.SelectSwapDirty.ComputeOptimalDirtySwapBits, QDKChemistry.Utils.SelectSwapDirty.SelectSwapDirty;
    import QDKChemistry.Utils.SelectSwapDirty.SelectSwapDirtyCost2D;
    import QDKChemistry.Utils.SkewTree.SkewTreeBoxSize, QDKChemistry.Utils.SkewTree.SkewTreeLookup, QDKChemistry.Utils.SkewTree.SkewTreeOutputTable;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;
    import QDKChemistry.Utils.UnaryIteration.ExactAnd;

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
        /// Whether every Givens angle is pre-reduced to [0, π), so its word drops the two always-zero top bits.
        compressedAngleWords : Bool,
        /// Givens angles held at once (λ, :cite:`Low2026` E3); 0 or ≥ N-1 keeps them all resident.
        rotationBatchSize : Int,
        /// Whether streamed batches and inner alias tables borrow idle system qubits for swaps.
        borrowDirty : Bool,
        /// Cap on any walk QROAM's swap width: -1 lets each selector choose, 0 forces plain `Select`.
        maxSwapBits : Int,
        /// Whether streamed batches share J + 1 telescoping lookups, borrowing `sysRegUp` when `borrowDirty` (:cite:`Low2026` E3).
        amortizeBatchLookups : Bool,
        /// Whether each streamed angle word is erased by measurement right after its rotation.
        measureAngleWords : Bool,
        /// Whether unswapped Givens-angle loads use a box-restricted skew tree instead of `Select`.
        useSkewTree : Bool,
        /// Whether x_o is ordered [SF at 2^nR·c + r | D1 | Q1] and SELECT recomputes the generator
        /// type, rank, spin and b == B from it instead of keeping them live (:cite:`Low2026` E2).
        recomputeOuterRegisters : Bool,
        /// Whether the N-1 Givens rotations run as the log-depth binary tree (:cite:`Low2026` Eq. D6)
        /// instead of the nearest-neighbour sweep; the rotation angles are always given as sweep angles.
        givensTree : Bool,
        /// Whether each phase-gradient Givens step is one controlled adder of a full-angle word
        /// (:cite:`Low2026` FIG. 39, Lemma 6) instead of two uncontrolled adders of a half-angle word.
        givensControlledAdder : Bool,
        /// Number of free-rider bits at the end of innerReg loaded by inner PREPARE QROM.
        /// Must be at least 2 unless `recomputeOuterRegisters`, which reads none.
        /// Layout: [sf_vs_dq(1), d_vs_q(1), r_bits(⌈log₂ R⌉)].
        /// SelectImpl reads isSF and dvsq from the first two bits.
        numFreeRiderBits : Int,
        /// Index into innerReg of the qubit the inner PREPARE loads the sign of the sampled
        /// (x_o, b) coefficient into, or -1 when the inner PREPARE supplies no sign bit.
        signQubitIndex : Int,
        /// Distinct nonzero Givens angle vectors that the one-hot SELECT (arXiv:2611.09822) loads one qubit each for.
        oneHotAngles : Double[][],
        /// `oneHotAngles` index of each DQ row x_o, or -1 for none; empty keeps the binary angle word.
        oneHotDQIndex : Int[],
        /// `oneHotAngles` index of each SF row b * R + r, or -1 for none.
        oneHotSFIndex : Int[],
        /// Whether spin swaps apply a Bell-basis parity phase (arXiv:2611.09822) instead of N controlled SWAPs.
        bellParitySpinSwap : Bool,
        /// Whether each one-hot SELECT qubit stands for two angle vectors, chosen by one more qubit (:cite:`Shirobokov2026` Eq. 23).
        oneHotPairedVectors : Bool,
        /// Whether SELECT leaves its trailing parity swap to the walk, which shares it with the following SELECT†.
        walkSharesSpinParity : Bool,
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
        /// Whether one Bell-basis parity serves the spin swaps that end SELECT and begin SELECT†;
        /// the SELECT must then be built with `walkSharesSpinParity`.
        sharedSpinParity : Bool,
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

    /// Slice the flat target register into the registers `layout` describes.
    internal function SOSSAWalkRegistersOf(layout : SOSSAWalkLayout, allQubits : Qubit[]) : SOSSAWalkRegisters {
        let bounds = SOSSAWalkRegisterBounds(layout);
        let outerStart = bounds[0];
        let innerStart = bounds[1];
        let spinStart = bounds[2];
        let gradientStart = bounds[3];
        new SOSSAWalkRegisters {
            systemReg = allQubits[0..outerStart - 1],
            outerReg = allQubits[outerStart..innerStart - 1],
            innerReg = allQubits[innerStart..spinStart - 1],
            spinReg = allQubits[spinStart..gradientStart - 1],
            phaseGradientReg = if bounds[4] > gradientStart {
                allQubits[gradientStart..bounds[4] - 1]
            } else {
                []
            }
        }
    }

    /// Angles `first..last` of the DQ table, quantized to `bRot` bits, stored in `wordBits`-bit
    /// words and concatenated low-index first; the resident word is the whole range `0..N-2`.
    internal function BuildDQRotationBatch(
        params : SelectParams,
        N : Int,
        first : Int,
        last : Int,
        bRot : Int,
        wordBits : Int,
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
                            bRot,
                            params.givensControlledAdder
                        ),
                        wordBits
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

    /// Angles `first..last` of the SF table, plus a trailing `b == B` flag bit if `withBEqBFlag`
    /// (free on the resident word; streamed batches set it arithmetically instead).
    internal function BuildSFRotationBatch(
        params : SelectParams,
        R : Int,
        first : Int,
        last : Int,
        bRot : Int,
        wordBits : Int,
        bBits : Int,
        rankBits : Int,
        rankFirst : Bool,
        withBEqBFlag : Bool,
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
                            bRot,
                            params.givensControlledAdder
                        ),
                        wordBits
                    ),
                    first..last
                )
            );
            set table += [if withBEqBFlag { angleBits + [b == params.numBases] } else { angleBits }];
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
    /// With `fullAngle` the word drives one controlled Ry(2θ) instead, so x = 2^b · 2θ / (4π), which is
    /// the half-angle word at b + 1 bits with its top bit dropped (that bit only flips both Ry signs).
    internal function QuantizeGivensAngle(angle : Double, bRot : Int, fullAngle : Bool) : Int {
        Fact(AbsD(angle) <= 4.0 * PI(), "QuantizeGivensAngle: angle must be finite and within [-4π, 4π]");
        let scale = IntAsDouble(if fullAngle { 2 <<< bRot } else { 1 <<< bRot });
        let raw = Round(scale * angle / (4.0 * PI()));
        ((raw % (1 <<< bRot)) + (1 <<< bRot)) % (1 <<< bRot)
    }

    /// Mode pairs `(p, c)` of the N-1 Givens rotations indexed by angle: `(j, j+1)` for the sweep,
    /// or the binary tree of :cite:`Low2026` Eq. D6 with contiguous-block labels, `(p, p + s)` for
    /// p a multiple of 2s, largest stride s first; each layer's Jordan-Wigner spans are disjoint,
    /// giving rotation depth ⌈log₂ N⌉.
    function GivensRotationPairs(N : Int, tree : Bool) : (Int, Int)[] {
        if not tree {
            return MappedOverRange(j -> (j, j + 1), 0..N - 2);
        }
        mutable stride = 1;
        while 2 * stride < N {
            set stride *= 2;
        }
        mutable pairs = [];
        while stride >= 1 {
            for p in 0..2 * stride..N - 1 - stride {
                set pairs += [(p, p + stride)];
            }
            set stride /= 2;
        }
        pairs
    }

    /// Tree angles that rotate the same unit vector onto mode 0 as the sweep angles `sweepAngles`.
    ///
    /// The vector is rebuilt from the sweep (u_j = cos θ_j Π_{k<j} sin θ_k), then zeroed edge by
    /// edge, deepest first, with the sweep's rule θ = atan2(u_c, u_p) merging child c into parent p.
    function SweepToTreeAngles(N : Int, sweepAngles : Double[]) : Double[] {
        mutable u = [0.0, size = N];
        mutable r = 1.0;
        for j in 0..N - 2 {
            let theta = if j < Length(sweepAngles) { sweepAngles[j] } else { 0.0 };
            set u w/= j <- r * Cos(theta);
            set r *= Sin(theta);
        }
        set u w/= N - 1 <- r;
        let pairs = GivensRotationPairs(N, true);
        mutable treeAngles = [0.0, size = N - 1];
        for e in N - 2..-1..0 {
            let (p, c) = pairs[e];
            set treeAngles w/= e <- ArcTan2(u[c], u[p]);
            set u w/= p <- Sqrt(u[p] * u[p] + u[c] * u[c]);
            set u w/= c <- 0.0;
        }
        treeAngles
    }

    /// Swap width for one streamed batch table: the clean or dirty selector's best width within
    /// `maxSwapBits` (-1 for any), or 0 when a skew tree costing `skewTreeCost` (-1 for none) is no dearer.
    internal function RotationSwapWidth(
        numData : Int,
        numBits : Int,
        availableDirty : Int,
        maxSwapBits : Int,
        skewTreeCost : Int,
    ) : Int {
        let width = if availableDirty == 0 {
            ComputeOptimalSwapBits(numData, numBits, maxSwapBits)
        } else {
            ComputeOptimalDirtySwapBits(numData, numBits, availableDirty, maxSwapBits)
        };
        let widthCost = if availableDirty == 0 {
            SelectSwapForwardCost(width, numData, numBits)
        } else {
            SelectSwapDirtyCost1D(width, numData, numBits)
        };
        if skewTreeCost >= 0 and skewTreeCost <= widthCost { 0 } else { width }
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
        let recompute = params.recomputeOuterRegisters;
        // E2 orders x_o as [SF at 2^nR·c + r | D1 | Q1], so SF spans [0, sfSpan) and r is its low nR bits.
        let nR = if params.numRanks > 0 { AddressQubits(params.numRanks) } else { 0 };
        let sfSpan = (1 <<< nR) * params.numCopies;
        let Xo = if recompute { sfSpan + N } else { N + numSF };
        let xoBits = MaxI(1, AddressQubits(Xo));
        let numBp1 = params.numBases + 1;
        let bBits = MaxI(1, AddressQubits(numBp1));
        let numRotAngles = N - 1;
        let rotationParams = if params.givensTree {
            new SelectParams {
                ...params,
                OneBodyRotationAngles = Mapped(angles -> SweepToTreeAngles(N, angles), params.OneBodyRotationAngles),
                TwoBodyRotationAngles = Mapped(angles -> SweepToTreeAngles(N, angles), params.TwoBodyRotationAngles),
                oneHotAngles = Mapped(angles -> SweepToTreeAngles(N, angles), params.oneHotAngles)
            }
        } else {
            params
        };

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
        // Recomputing from x_o reads none of them.
        let nInner = Length(innerReg);
        let nFR = params.numFreeRiderBits;
        Fact(recompute or nFR >= 2, "SelectImpl requires at least two free-rider bits (sf_vs_dq and d_vs_q)");

        // `spin` by default; the recomputed [x_o < sfSpan] flag when recomputing from x_o.
        use flag = Qubit();
        use bEqBQubit = Qubit[if recompute { 0 } else { 1 }];
        let isSF = if recompute { flag } else { innerReg[nInner - nFR] };
        let rBits = if recompute { xoReg[0..nR - 1] } elif nFR > 2 { innerReg[nInner - nFR + 2..nInner - 1] } else { [] };

        // sign of the sampled (x_o, b) coefficient.
        if params.signQubitIndex >= 0 {
            Z(innerReg[params.signQubitIndex]);
        }
        // Q1's a^dagger sign on the Majorana selector, which nothing below touches (:cite:`Low2026` Eq. E12).
        if recompute and params.numPositiveOneBody < N {
            ApplyIfLessOrEqualL(Z, IntAsBigInt(sfSpan + params.numPositiveOneBody), xoReg, spinSF);
        }

        // The Majorana operator runs in the Givens-rotated basis, so it is passed into the
        // rotation step instead of being wrapped around it.
        let majoranaStep : (Unit => Unit is Adj + Ctl) = () => {
            if recompute {
                // SF: Z unless b == B, the identity term, applied as a multi-controlled Z.
                Controlled Z([isSF], sysRegDown[0]);
                ApplyControlledOnInt(params.numBases, Controlled Z([isSF], _), bReg, sysRegDown[0]);
                within { X(isSF); } apply {
                    CNOT(isSF, sysRegDown[0]);
                    Controlled Z([isSF, spinSF], sysRegDown[0]);
                }
            } else {
                MajoranaOp(isSF, innerReg[nInner - nFR + 1], bEqBQubit[0], flag, spinSF, sysRegDown[0]);
            }
        };

        // Givens rotations: basis change to localize amplitude on qubit 0, around the Majorana operator.
        let rotatedMajorana : (Unit => Unit is Adj + Ctl) = () => {
            if usePhaseGradient {
                WithGivensRotationsQROM(
                    rotationParams,
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
                        rotationParams,
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
                        bEqBQubit
                    );
                } apply {
                    // Majorana operator (Fig. 4 / Appendix B.6)
                    majoranaStep();
                }
            }
        };

        // Adding 2^(xoBits+1) - sfSpan sets the carry to [x_o < sfSpan], leaves r in the low
        // nR bits of an SF index and turns a DQ index into its one-body index a; spinDQ then
        // holds the active spin, and spinSF stays the DQ Majorana selector.
        let recomputeSpins : (Unit => Unit is Adj + Ctl) = () => {
            IncByLUsingIncByLE(RippleCarryCGIncByLE, IntAsBigInt((1 <<< (xoBits + 1)) - sfSpan), xoReg + [isSF]);
            Controlled SWAP([isSF], (spinDQ, spinSF));
        };
        if params.bellParitySpinSwap {
            let activeSpin = if recompute { spinDQ } else { flag };
            within {
                if recompute {
                    recomputeSpins();
                } else {
                    SelectSpins(isSF, spinDQ, spinSF, flag, [], []);
                }
            } apply {
                use parity = Qubit();
                within {
                    ComputeSpinParity(sysRegDown, sysRegUp, parity);
                } apply {
                    CZ(activeSpin, parity);
                }
                rotatedMajorana();
                if not params.walkSharesSpinParity {
                    within {
                        ComputeSpinParity(sysRegDown, sysRegUp, parity);
                    } apply {
                        CZ(activeSpin, parity);
                    }
                }
            }
        } else {
            within {
                if recompute {
                    recomputeSpins();
                    Controlled ApplyToEachCA([spinDQ], (SWAP, Zipped(sysRegDown, sysRegUp)));
                } else {
                    SelectSpins(isSF, spinDQ, spinSF, flag, sysRegDown, sysRegUp);
                }
            } apply {
                rotatedMajorana();
            }
        }
    }

    /// Write into `parity` the number of singlet pairs (registerDown[i], registerUp[i]) mod 2 (arXiv:2611.09822).
    ///
    /// A pair SWAP is -1 on the singlet and +1 on the triplets, so CZ(spin, parity) between this
    /// and its adjoint is the controlled SWAP of the two registers for N Toffolis. The adjoint
    /// erases `parity` with an X measurement and a Clifford phase fix-up, so it costs no Toffolis.
    internal operation ComputeSpinParity(registerDown : Qubit[], registerUp : Qubit[], parity : Qubit) : Unit is Adj {
        body ... {
            within {
                ApplyToEachA(CNOT, Zipped(registerDown, registerUp));
                ApplyToEachA(H, registerDown);
            } apply {
                for (down, up) in Zipped(registerDown, registerUp) {
                    CCNOT(down, up, parity);
                }
            }
        }
        adjoint ... {
            within {
                ApplyToEachA(CNOT, Zipped(registerDown, registerUp));
                ApplyToEachA(H, registerDown);
            } apply {
                if MResetX(parity) == One {
                    ApplyToEachA(CZ, Zipped(registerDown, registerUp));
                }
            }
        }
    }

    /// Apply CZ(spin, parity) with spin = isSF ? spinSF : spinDQ, using one AND with measurement uncompute.
    internal operation ApplySpinParityPhase(isSF : Qubit, spinDQ : Qubit, spinSF : Qubit, parity : Qubit) : Unit is Adj {
        use differs = Qubit();
        within {
            CNOT(spinDQ, spinSF);
            AND(isSF, spinSF, differs);
        } apply {
            CZ(differs, parity);
        }
        CZ(spinDQ, parity);
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
        sharedSpinParity : Bool,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj {
        ApplySOSSABlockEncoding(
            [],
            outerPrepareOp,
            freeRiderOp,
            innerPrepareOp,
            selectOp,
            numReflectInner,
            numOuterIndexQubits,
            numOuterPrepareGradientQubits,
            numFreeRiderQubits,
            sharedSpinParity,
            outerReg,
            innerReg,
            spinReg,
            systemReg,
            phaseGradientReg,
        );
    }

    /// Apply B = U† · Ref_B · U with Ref_B controlled on `controls`.
    ///
    /// U runs unconditionally: with the controls off U† · U = I, so this is controlled-B
    /// without a controlled PREPARE, which the inner alias-sampling PREPARE does not provide.
    /// With no controls it is the uncontrolled B.
    internal operation ApplySOSSABlockEncoding(
        controls : Qubit[],
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numReflectInner : Int,
        numOuterIndexQubits : Int,
        numOuterPrepareGradientQubits : Int,
        numFreeRiderQubits : Int,
        sharedSpinParity : Bool,
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
                // With `sharedSpinParity`, SELECT omits its trailing spin swap and one parity serves it and
                // SELECT†'s leading swap: the walk leaves the system unchanged between them (arXiv:2611.09822).
                use parity = Qubit[sharedSpinParity ? 1 | 0];
                within {
                    within {
                        // Only SELECT touches `systemReg`, and it runs in `apply`, so it is idle and lendable here.
                        innerPrepareOp(outerIndexReg, innerReg, systemReg);
                        H(spinReg[1]);
                    } apply {
                        selectOp(outerIndexReg, innerReg, spinReg, systemReg, phaseGradientReg);
                        if sharedSpinParity {
                            let N = Length(systemReg) / 2;
                            ComputeSpinParity(systemReg[0..N - 1], systemReg[N...], parity[0]);
                            ApplySpinParityPhase(freeRiderReg[0], spinReg[0], spinReg[1], parity[0]);
                        }
                    }
                } apply {
                    let reflected = innerReg[0..numReflectInner - 1] + [spinReg[1]];
                    if Length(controls) == 0 {
                        Reflect(reflected);
                    } else {
                        Controlled Reflect(controls, reflected);
                    }
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
        let regs = SOSSAWalkRegistersOf(layout, allQubits);
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
            layout.sharedSpinParity,
            regs.outerReg,
            regs.innerReg + innerScratch,
            regs.spinReg,
            regs.systemReg,
            regs.phaseGradientReg,
        );
    }

    /// Apply controlled-W^power, W = Ref_a · B, to a flat target register.
    ///
    /// Ref_a reflects about the zero state of `[outerReg | innerReg | spinReg]`, the same
    /// register the unary-iteration QPE reflects; the phase-gradient tail is not reflected.
    /// Both reflections are controlled and U runs unconditionally, see `ApplySOSSABlockEncoding`.
    /// Register layout: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    /// `cacheName` keys the resource-estimation cache of one walk step. It must identify the walk,
    /// so that distinct walks in one program are estimated separately and the powers of one walk share it.
    /// W and W† cost the same, so both directions share the entry. The adjoint is written out because
    /// a generated one would run `EndEstimateCaching` before the step and cache an empty estimate.
    operation ControlledSOSSAWalkOnRegister(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        cacheName : String,
        power : Int,
        control : Qubit,
        allQubits : Qubit[],
    ) : Unit is Adj {
        body ... {
            for _ in 1..power {
                if BeginEstimateCaching(cacheName, SingleVariant()) {
                    ControlledSOSSAWalkStep(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, control, allQubits);
                    EndEstimateCaching();
                }
            }
        }
        adjoint ... {
            for _ in 1..power {
                if BeginEstimateCaching(cacheName, SingleVariant()) {
                    Adjoint ControlledSOSSAWalkStep(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, control, allQubits);
                    EndEstimateCaching();
                }
            }
        }
    }

    /// Apply one step of controlled-W, see `ControlledSOSSAWalkOnRegister`.
    operation ControlledSOSSAWalkStep(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        control : Qubit,
        allQubits : Qubit[],
    ) : Unit is Adj {
        let regs = SOSSAWalkRegistersOf(layout, allQubits);
        {
            // The block encoding returns the inner scratch to |0⟩, so release it before Ref_a.
            use innerScratch = Qubit[layout.numInnerQubits - layout.numReflectInner];
            ApplySOSSABlockEncoding(
                [control],
                outerPrepareOp,
                freeRiderOp,
                innerPrepareOp,
                selectOp,
                layout.numReflectInner,
                layout.numOuterIndexQubits,
                layout.numOuterPrepareGradientQubits,
                layout.numFreeRiderQubits,
                layout.sharedSpinParity,
                regs.outerReg,
                regs.innerReg + innerScratch,
                regs.spinReg,
                regs.systemReg,
                regs.phaseGradientReg,
            );
        }
        Controlled Reflect([control], regs.outerReg + regs.innerReg + regs.spinReg);
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
    /// When recomputing from x_o, xoReg holds a (DQ) or r in its low bits (SF) beside `isSF`,
    /// and `bEqBQubit` is empty.
    /// With `params.givensTree` the pair (j, j+1) becomes the tree edge `GivensRotationPairs` gives.
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
        rBits : Qubit[],
        sysRegDown : Qubit[],
        bEqBQubit : Qubit[]
    ) : Unit is Adj + Ctl {
        let recompute = params.recomputeOuterRegisters;
        let nR = Length(rBits);
        let bBits = Length(bReg);
        let pairs = GivensRotationPairs(N, params.givensTree);
        for j in numRotAngles - 1..-1..0 {
            let (p, c) = pairs[j];
            // Tree pairs need not be adjacent; the CZ string keeps each gate a fermionic Givens rotation.
            for s in p + 1..c - 1 {
                CZ(sysRegDown[s], sysRegDown[c]);
            }
            CNOT(sysRegDown[p], sysRegDown[c]);

            // DQ rotations: x_o in [0, N)
            for a in 0..N - 1 {
                let angle = params.OneBodyRotationAngles[a][j];
                if recompute {
                    ApplyControlledOnInt(
                        a + (1 <<< (xoBits + 1)),
                        Ry(2.0 * angle, _),
                        xoReg + [isSF, sysRegDown[c]],
                        sysRegDown[p]
                    );
                } else {
                    ApplyControlledOnInt(
                        a + (1 <<< xoBits),
                        Ry(2.0 * angle, _),
                        xoReg + [sysRegDown[c]],
                        sysRegDown[p]
                    );
                }
            }

            if recompute {
                // SF rotations depend on (r, b) only, so the copies c share one rotation.
                for r in 0..params.numRanks - 1 {
                    for b in 0..numBp1 - 1 {
                        let angleIdx = b * params.numRanks + r;
                        if angleIdx < Length(params.TwoBodyRotationAngles) and j < Length(params.TwoBodyRotationAngles[angleIdx]) {
                            let condValue = r + (b <<< nR) + (1 <<< (nR + bBits)) + (1 <<< (nR + bBits + 1));
                            ApplyControlledOnInt(
                                condValue,
                                Ry(2.0 * params.TwoBodyRotationAngles[angleIdx][j], _),
                                rBits + bReg + [isSF, sysRegDown[c]],
                                sysRegDown[p]
                            );
                        }
                    }
                }
            } else {
                // SF rotations: x_o in [N, N+numSF), conditioned on b
                for xoIdx in 0..numSF - 1 {
                    let xo = N + xoIdx;
                    let r = xoIdx / params.numCopies;
                    for b in 0..numBp1 - 1 {
                        let angleIdx = b * params.numRanks + r;
                        if angleIdx < Length(params.TwoBodyRotationAngles) and j < Length(params.TwoBodyRotationAngles[angleIdx]) {
                            let angle = params.TwoBodyRotationAngles[angleIdx][j];
                            let condValue = xo + b * (1 <<< xoBits) + (1 <<< (xoBits + bBits));
                            ApplyControlledOnInt(
                                condValue,
                                Ry(2.0 * angle, _),
                                xoReg + bReg + [sysRegDown[c]],
                                sysRegDown[p]
                            );
                        }
                    }
                }
            }

            CNOT(sysRegDown[p], sysRegDown[c]);
            for s in p + 1..c - 1 {
                CZ(sysRegDown[s], sysRegDown[c]);
            }
        }
        // Set bEqB flag: 1 when (isSF AND b == B)
        for flag in bEqBQubit {
            ApplyControlledOnInt(params.numBases, q => Controlled X([isSF], q), bReg, flag);
        }
    }

    /// Loads the branch-selected Givens angle word; the adjoint erases it by measurement.
    /// Both tables write the same `target` and measuring consumes it, so the adjoint measures once
    /// and repairs each branch's phase; letting `within` uncompute instead costs a second lookup.
    /// Each table has its own swap width; every route loads `target` exactly as plain `Select` does
    /// on in-range addresses. With `useSkewTree`, unswapped loads walk a skew tree over the box
    /// `sfFields` (SF) or `Length(dqData)` (DQ) instead of padding to a power of two.
    internal operation ControlledSelectWithUnlookup(
        sfData : Bool[][],
        dqData : Bool[][],
        sfSwapBits : Int,
        dqSwapBits : Int,
        useSkewTree : Bool,
        sfFields : (Int, Int)[],
        sfAddress : Qubit[],
        dqAddress : Qubit[],
        isSF : Qubit,
        dirty : Qubit[],
        target : Qubit[],
    ) : Unit is Adj {
        body (...) {
            // `Select` tolerates a table length that is not a power of two -- it never reads an
            // address at or above `Length(data)` -- so both tables are used as built.
            LoadRotationWord(sfData, sfSwapBits, useSkewTree, sfFields, sfAddress, isSF, dirty, target);
            let dqTarget = target[0..Length(dqData[0]) - 1];
            within { X(isSF); } apply {
                LoadRotationWord(dqData, dqSwapBits, useSkewTree, [], dqAddress, isSF, dirty, dqTarget);
            }
        }
        adjoint (...) {
            let measured = ResultArrayAsBoolArray(ForEach(MResetX, target));
            ApplyRotationWordFixup(measured, sfData, dqData, sfSwapBits, dqSwapBits, useSkewTree, sfFields, sfAddress, dqAddress, isSF);
        }
    }

    /// Repairs the phase that measuring the branch-selected angle word in the X basis leaves behind.
    /// Each branch contributes the parity of its own word, so the two fixups are independent.
    /// A skew tree's word is linear in its rows, so its phases follow from the row parities alone.
    internal operation ApplyRotationWordFixup(
        measured : Bool[],
        sfData : Bool[][],
        dqData : Bool[][],
        sfSwapBits : Int,
        dqSwapBits : Int,
        useSkewTree : Bool,
        sfFields : (Int, Int)[],
        sfAddress : Qubit[],
        dqAddress : Qubit[],
        isSF : Qubit,
    ) : Unit {
        if useSkewTree and sfSwapBits == 0 {
            let parities = Mapped(row -> [MeasuredParity(measured, row)], sfData);
            ApplyBranchPhaseFixup([true], SkewTreeOutputTable(parities, sfFields, Length(sfAddress)), true, [isSF] + sfAddress);
        } else {
            ApplyBranchPhaseFixup(measured, sfData, true, [isSF] + sfAddress);
        }
        if useSkewTree and dqSwapBits == 0 {
            let parities = Mapped(row -> [MeasuredParity(measured, row)], dqData);
            ApplyBranchPhaseFixup([true], SkewTreeOutputTable(parities, [], Length(dqAddress)), false, [isSF] + dqAddress);
        } else {
            ApplyBranchPhaseFixup(measured, dqData, false, [isSF] + dqAddress);
        }
    }

    /// Parity of `word` against the measured bits it overlaps.
    internal function MeasuredParity(measured : Bool[], word : Bool[]) : Bool {
        mutable parity = false;
        for (measuredBit, wordBit) in Zipped(measured, word) {
            set parity = parity != (measuredBit and wordBit);
        }
        parity
    }

    /// One branch's forward load, controlled on `isSF`.
    internal operation LoadRotationWord(
        data : Bool[][],
        numSwapBits : Int,
        useSkewTree : Bool,
        fields : (Int, Int)[],
        address : Qubit[],
        isSF : Qubit,
        dirty : Qubit[],
        target : Qubit[],
    ) : Unit is Adj + Ctl {
        if numSwapBits == 0 and useSkewTree {
            Controlled SkewTreeLookup([isSF], (data, fields, address, target));
        } elif numSwapBits == 0 {
            Controlled Select([isSF], (data, address, target));
        } elif IsEmpty(dirty) {
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
    /// With `givensControlledAdder`, each rotation is instead one controlled Adder(bRot) on a
    /// full-angle word (:cite:`Low2026` FIG. 39), costing 2*bRot - 1 Toffolis, or 3*bRot - 2 for
    /// the first step of a streamed batch, which has no erased word to borrow as clean ancillas.
    ///
    /// L_SF is above the paper's R*B because `Select` pads whichever register addresses the
    /// low bits out to a power of two; `SFTableRankAddressedFirst` picks the cheaper of the
    /// two orderings. `params.useSkewTree` removes the padding instead: a skew tree over the
    /// box b ≤ B, r < R loads (B+1)*R entries at the same cost per entry.
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
        bEqBQubit : Qubit[],
        action : (Unit => Unit is Adj + Ctl),
    ) : Unit is Adj + Ctl {
        let bRot = params.rotationBitPrecision;
        // Angles pre-reduced to [0, π) leave the top two bits of a half-angle word zero, or the top bit of a
        // full-angle (controlled adder) word, so they are not stored.
        let wordBits = if not params.compressedAngleWords { bRot } elif params.givensControlledAdder { bRot - 1 } else { bRot - 2 };
        let bBits = Length(bReg);
        let R = params.numRanks;
        let nDQBits = MaxI(1, AddressQubits(N));
        let dqAddress = xoReg[0..nDQBits - 1];

        // SF table: addressed by (bReg ++ rBits) or (rBits ++ bReg), whichever is smaller.
        let rankFirst = SFTableRankAddressedFirst(R, params.numBases, bBits, Length(rBits));
        let sfAddress = if rankFirst { rBits + bReg } else { bReg + rBits };
        // In-range SF addresses form the box b ≤ B, r < R, low field first, which a skew tree loads unpadded.
        let sfFields = if rankFirst {
            [(Length(rBits), R), (bBits, params.numBases + 1)]
        } else {
            [(bBits, params.numBases + 1), (Length(rBits), R)]
        };

        let batch = params.rotationBatchSize;
        let pairs = GivensRotationPairs(N, params.givensTree);

        if Length(params.oneHotDQIndex) > 0 {
            // One-hot SELECT (arXiv:2611.09822): load one qubit per distinct angle vector plus the b == B
            // flag, then fan each angle word out of the one-hot register with CNOTs. Paired, qubit q stands
            // for vectors 2q and 2q + 1 and one more qubit says which (:cite:`Shirobokov2026` Eq. 23).
            let numVectors = Length(params.oneHotAngles);
            let paired = params.oneHotPairedVectors;
            Fact(not paired or params.givensControlledAdder, "Paired one-hot vectors need the controlled-adder Givens step");
            let numSlots = paired ? (numVectors + 1) / 2 | numVectors;
            let numSelect = paired ? 1 | 0;
            // The b == B flag rides on the SF row only when there is a flag qubit to receive it.
            let numFlags = Length(bEqBQubit);
            let oneHotRow = (v, flag) -> MappedOverRange(i -> v >= 0 and i == (paired ? v / 2 | v), 0..numSlots - 1) + (paired ? [v >= 0 and v % 2 == 1] | []) + (numFlags > 0 ? [flag] | []);
            let dqData = MappedOverRange(xo -> oneHotRow(params.oneHotDQIndex[xo], false), 0..N - 1);
            let rSlots = 1 <<< Length(rBits);
            let bSlots = 1 <<< bBits;
            let sfData = MappedOverRange(
                idx -> {
                    let b = if rankFirst { idx / rSlots } else { idx % bSlots };
                    let r = if rankFirst { idx % rSlots } else { idx / bSlots };
                    let angleIdx = b * R + r;
                    oneHotRow(if r < R and angleIdx < Length(params.oneHotSFIndex) { params.oneHotSFIndex[angleIdx] } else { -1 }, b == params.numBases)
                },
                0..(if rankFirst { numBp1 * rSlots } else { R * bSlots }) - 1
            );
            use oneHot = Qubit[numSlots + numSelect + numFlags];
            within {
                ControlledSelectWithUnlookup(sfData, dqData, 0, 0, params.useSkewTree, sfFields, sfAddress, dqAddress, isSF, [], oneHot);
            } apply {
                within {
                    for flag in bEqBQubit {
                        CNOT(oneHot[numSlots + numSelect], flag);
                    }
                    for j in numRotAngles - 1..-1..0 {
                        let words = MappedOverRange(
                            v -> {
                                let angles = params.oneHotAngles[v];
                                IntAsBoolArray(QuantizeGivensAngle(j < Length(angles) ? angles[j] | 0.0, bRot, params.givensControlledAdder), wordBits)
                            },
                            0..numVectors - 1
                        );
                        if paired {
                            let (p, c) = pairs[j];
                            ApplyPairedOneHotGivensStep(
                                p,
                                c,
                                sysRegDown,
                                oneHot[0..numSlots - 1],
                                oneHot[numSlots],
                                MappedOverRange(q -> words[2 * q], 0..numSlots - 1),
                                MappedOverRange(q -> 2 * q + 1 < numVectors ? words[2 * q + 1] | [false, size = wordBits], 0..numSlots - 1),
                                phaseGradientReg
                            );
                        } else {
                            use word = Qubit[wordBits];
                            within {
                                for v in 0..numVectors - 1 {
                                    for k in 0..wordBits - 1 {
                                        if words[v][k] {
                                            CNOT(oneHot[v], word[k]);
                                        }
                                    }
                                }
                            } apply {
                                ApplyGivensRotationWords(word, j, j, wordBits, pairs, sysRegDown, phaseGradientReg, params.givensControlledAdder);
                            }
                        }
                    }
                } apply {
                    action();
                }
            }
        } elif batch <= 0 or batch >= numRotAngles {
            // Zero, or any size at or above N - 1, keeps the whole angle word resident.
            let nRotBits = numRotAngles * wordBits;
            // The b == B flag rides on the SF word only when there is a flag qubit to receive it.
            let withFlag = Length(bEqBQubit) > 0;
            let dqData = BuildDQRotationBatch(params, N, 0, numRotAngles - 1, bRot, wordBits);
            let sfData = BuildSFRotationBatch(params, R, 0, numRotAngles - 1, bRot, wordBits, bBits, Length(rBits), rankFirst, withFlag);

            // Allocate rotation target register: (N-1)*wordBits rotation bits plus the bEqB flag bit, if any.
            use rotTarget = Qubit[nRotBits + Length(bEqBQubit)];

            within {
                // The resident word is already the full angle table, so no swap network fits.
                ControlledSelectWithUnlookup(sfData, dqData, 0, 0, params.useSkewTree, sfFields, sfAddress, dqAddress, isSF, [], rotTarget);
            } apply {
                within {
                    if withFlag {
                        CNOT(rotTarget[nRotBits], bEqBQubit[0]);
                    }
                    ApplyGivensRotationWords(rotTarget, 0, numRotAngles - 1, wordBits, pairs, sysRegDown, phaseGradientReg, params.givensControlledAdder);
                } apply {
                    action();
                }
            }
        } elif params.amortizeBatchLookups {
            let numBatches = (numRotAngles + batch - 1) / batch;
            let width = batch * wordBits;
            let sfTables = MappedOverRange(
                t -> PaddedRows(
                    BuildSFRotationBatch(params, R, t * batch, MinI(t * batch + batch, numRotAngles) - 1, bRot, wordBits, bBits, Length(rBits), rankFirst, false),
                    width
                ),
                0..numBatches - 1
            );
            let dqTables = MappedOverRange(
                t -> PaddedRows(BuildDQRotationBatch(params, N, t * batch, MinI(t * batch + batch, numRotAngles) - 1, bRot, wordBits), width),
                0..numBatches - 1
            );
            // Lenders must survive every batch's rotations, which touch all of sysRegDown.
            let lenders = if params.borrowDirty { sysRegUp } else { [] };
            let sfSwapBits = AmortizedDirtySwapBits(Length(sfTables[0]), width, Length(lenders), params.maxSwapBits);
            let dqSwapBits = AmortizedDirtySwapBits(Length(dqTables[0]), width, Length(lenders), params.maxSwapBits);
            within {
                for flag in bEqBQubit {
                    ApplyControlledOnInt(params.numBases, q => Controlled X([isSF], q), bReg, flag);
                }
                // Descending batches keep the resident path's high-to-low rotation order.
                AmortizedRotationWordSweep(
                    sfTables,
                    dqTables,
                    MappedOverRange(i -> numBatches - 1 - i, 0..numBatches - 1),
                    sfSwapBits,
                    dqSwapBits,
                    sfAddress,
                    dqAddress,
                    isSF,
                    lenders,
                    params.measureAngleWords,
                    (t, word) => ApplyGivensRotationWords(word, t * batch, MinI(t * batch + batch, numRotAngles) - 1, wordBits, pairs, sysRegDown, phaseGradientReg, params.givensControlledAdder)
                );
            } apply {
                action();
            }
        } else {
            let numBatches = (numRotAngles + batch - 1) / batch;
            within {
                // Computed from b and isSF rather than loaded, which a streamed table would pay for per batch.
                for flag in bEqBQubit {
                    ApplyControlledOnInt(params.numBases, q => Controlled X([isSF], q), bReg, flag);
                }

                // Descending batches keep the resident path's high-to-low rotation order.
                for t in numBatches - 1..-1..0 {
                    let first = t * batch;
                    let last = MinI(first + batch, numRotAngles) - 1;
                    let dqData = BuildDQRotationBatch(params, N, first, last, bRot, wordBits);
                    let sfData = BuildSFRotationBatch(params, R, first, last, bRot, wordBits, bBits, Length(rBits), rankFirst, false);

                    // Qubits outside every pair span [p, c] of this window are live but untouched, so they are lent (:cite:`Low2026` E3).
                    let spans = MappedOverRange(e -> pairs[e], first..last);
                    let untouchedModes = Filtered(i -> All((p, c) -> i < p or i > c, spans), MappedOverRange(i -> i, 0..N - 1));
                    let untouched = Subarray(untouchedModes, sysRegDown);
                    let dirty = if params.borrowDirty { sysRegUp + untouched } else { [] };
                    // Size each table's swap width on its own word: the DQ word is only a prefix of the SF word.
                    // A singly controlled skew-tree load costs one Toffoli per box node past the first.
                    let sfSkewCost = if params.useSkewTree { SkewTreeBoxSize(Length(sfData), sfFields, Length(sfAddress)) - 1 } else { -1 };
                    let dqSkewCost = if params.useSkewTree { SkewTreeBoxSize(Length(dqData), [], Length(dqAddress)) - 1 } else { -1 };
                    let sfSwapBits = RotationSwapWidth(Length(sfData), Length(sfData[0]), Length(dirty), params.maxSwapBits, sfSkewCost);
                    let dqSwapBits = RotationSwapWidth(Length(dqData), Length(dqData[0]), Length(dirty), params.maxSwapBits, dqSkewCost);

                    // Only this window's angles are resident; the enclosing `within` looks them up again to unwind.
                    if params.givensControlledAdder {
                        // Its words are erased by measurement right after their steps, whatever measureAngleWords says.
                        ApplyStreamedControlledAdderBatch(
                            sfData,
                            dqData,
                            sfSwapBits,
                            dqSwapBits,
                            params.useSkewTree,
                            sfFields,
                            sfAddress,
                            dqAddress,
                            isSF,
                            dirty,
                            first,
                            last,
                            wordBits,
                            pairs,
                            sysRegDown,
                            phaseGradientReg
                        );
                    } elif params.measureAngleWords {
                        MeasuredRotationWordBatch(
                            sfData,
                            dqData,
                            sfSwapBits,
                            dqSwapBits,
                            params.useSkewTree,
                            sfFields,
                            sfAddress,
                            dqAddress,
                            isSF,
                            dirty,
                            first,
                            last,
                            wordBits,
                            pairs,
                            sysRegDown,
                            phaseGradientReg
                        );
                    } else {
                        use rotTarget = Qubit[(last - first + 1) * wordBits];
                        within {
                            ControlledSelectWithUnlookup(
                                sfData,
                                dqData,
                                sfSwapBits,
                                dqSwapBits,
                                params.useSkewTree,
                                sfFields,
                                sfAddress,
                                dqAddress,
                                isSF,
                                dirty,
                                rotTarget
                            );
                        } apply {
                            ApplyGivensRotationWords(rotTarget, first, last, wordBits, pairs, sysRegDown, phaseGradientReg, false);
                        }
                    }
                }
            } apply {
                action();
            }
        }
    }

    /// Applies the Givens rotations for angles `first..last` on their mode `pairs`, reading the
    /// `wordBits`-bit angle words out of `rotTarget` (which holds exactly that window, low index
    /// first); a word shorter than the gradient is zero-extended by the adder. With `controlledAdder`
    /// each step is one controlled adder on fresh clean ancillas, as resident words cannot be erased early.
    internal operation ApplyGivensRotationWords(
        rotTarget : Qubit[],
        first : Int,
        last : Int,
        wordBits : Int,
        pairs : (Int, Int)[],
        sysRegDown : Qubit[],
        phaseGradientReg : Qubit[],
        controlledAdder : Bool,
    ) : Unit is Adj {
        for j in last..-1..first {
            let word = rotTarget[(j - first) * wordBits..(j - first + 1) * wordBits - 1];
            let (p, c) = pairs[j];
            if controlledAdder {
                use clean = Qubit[Length(phaseGradientReg) - 1];
                ApplyControlledAdderGivensStep(p, c, sysRegDown, word, phaseGradientReg, clean);
            } else {
                within {
                    // Tree pairs need not be adjacent; the CZ string keeps each gate a fermionic Givens rotation.
                    for s in p + 1..c - 1 {
                        CZ(sysRegDown[s], sysRegDown[c]);
                    }
                    CNOT(sysRegDown[p], sysRegDown[c]);
                } apply {
                    // :cite:`Low2026` FIG. 40: controlled RZ(2θ) from two parallel uncontrolled RZ(θ) gates.
                    within {
                        CNOT(sysRegDown[c], sysRegDown[p]);
                    } apply {
                        Adjoint RyViaPhaseGradient(sysRegDown[p], word, phaseGradientReg);
                    }
                    RyViaPhaseGradient(sysRegDown[p], word, phaseGradientReg);
                }
            }
        }
    }

    /// One fermionic Givens step on modes (p, c): Ry(4π·x/2^b) on p controlled on c, made from a single
    /// adder of `word` into the phase gradient controlled on c (:cite:`Low2026` FIG. 39, Lemma 6), costing
    /// 2b - 2 Toffolis with b - 1 `clean` ancillas (Gidney) or 3b - 3 without (Muñoz-Coreas-Thapliyal), one below
    /// Lemma 7 because the gradient's |−⟩ top qubit makes its controlled flip a CZ.
    internal operation ApplyControlledAdderGivensStep(
        p : Int,
        c : Int,
        sysRegDown : Qubit[],
        word : Qubit[],
        phaseGradientReg : Qubit[],
        clean : Qubit[],
    ) : Unit is Adj {
        let n = Length(phaseGradientReg);
        Fact(Length(word) <= n, "ApplyControlledAdderGivensStep: the word must not be wider than the phase gradient");
        // A compressed word is zero-extended at its top to the gradient's width.
        use pad = Qubit[n - Length(word)];
        let ctrl = sysRegDown[c];
        let a = word + pad;
        let g = phaseGradientReg;
        within {
            GivensPhaseGradientFrame(p, c, sysRegDown, g);
        } apply {
            if Length(clean) >= n - 1 {
                // Carries c_{i+1} = maj(a_i, g_i, c_i) are ANDed into `clean`, then g_i ^= ctrl·(a_i ⊕ c_i) unwinds them.
                for i in 0..n - 2 {
                    if i > 0 {
                        CNOT(clean[i - 1], a[i]);
                        CNOT(clean[i - 1], g[i]);
                    }
                    AND(a[i], g[i], clean[i]);
                    if i > 0 {
                        CNOT(clean[i - 1], clean[i]);
                    }
                }
                // The gradient's top qubit is |−⟩, so flipping it on ctrl·(a ⊕ c) is the phase CZ(ctrl, a ⊕ c).
                CZ(ctrl, a[n - 1]);
                if n > 1 {
                    CZ(ctrl, clean[n - 2]);
                }
                for i in n - 2..-1..0 {
                    if i > 0 {
                        CNOT(clean[i - 1], clean[i]);
                    }
                    Adjoint AND(a[i], g[i], clean[i]);
                    CCNOT(ctrl, a[i], g[i]);
                    if i > 0 {
                        CNOT(clean[i - 1], a[i]);
                        CNOT(clean[i - 1], g[i]);
                    }
                }
            } else {
                for i in 1..n - 1 {
                    CNOT(a[i], g[i]);
                }
                for i in n - 2..-1..1 {
                    CNOT(a[i], a[i + 1]);
                }
                for i in 0..n - 2 {
                    CCNOT(g[i], a[i], a[i + 1]);
                }
                for i in n - 1..-1..1 {
                    if i == n - 1 {
                        CZ(ctrl, a[i]);
                    } else {
                        CCNOT(ctrl, a[i], g[i]);
                    }
                    CCNOT(g[i - 1], a[i - 1], a[i]);
                }
                for i in 1..n - 2 {
                    CNOT(a[i], a[i + 1]);
                }
                for i in 1..n - 1 {
                    CNOT(a[i], g[i]);
                }
                if n == 1 {
                    CZ(ctrl, a[0]);
                } else {
                    CCNOT(ctrl, a[0], g[0]);
                }
            }
        }
    }

    /// Basis change around a phase-gradient Givens step on modes (p, c), after which adding x into the
    /// gradient while c is set applies Ry(4π·x/2^b) on p (the RyViaPhaseGradient frame with a CZ string).
    internal operation GivensPhaseGradientFrame(p : Int, c : Int, sysRegDown : Qubit[], phaseGradientReg : Qubit[]) : Unit is Adj {
        for s in p + 1..c - 1 {
            CZ(sysRegDown[s], sysRegDown[c]);
        }
        CNOT(sysRegDown[p], sysRegDown[c]);
        Adjoint S(sysRegDown[p]);
        H(sysRegDown[p]);
        X(sysRegDown[p]);
        for g in phaseGradientReg {
            CNOT(sysRegDown[p], g);
        }
    }

    /// One controlled-adder Givens step whose full-angle word comes from a paired one-hot register: bit k
    /// enters the gradient as (c ⊕ u ⊕ x1)(u ⊕ x0) ⊕ x01 with u = c·s (:cite:`Shirobokov2026` Eq. 23), where
    /// x0, x1 and x01 are the parities of the `oneHotPairs` qubits whose first, second or both words set it,
    /// then one uncontrolled adder costs b - 2 Toffolis, for 2b - 1 per step or one less per always-zero bit.
    internal operation ApplyPairedOneHotGivensStep(
        p : Int,
        c : Int,
        sysRegDown : Qubit[],
        oneHotPairs : Qubit[],
        second : Qubit,
        firstWords : Bool[][],
        secondWords : Bool[][],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj {
        let n = Length(phaseGradientReg);
        let ctrl = sysRegDown[c];
        let g = phaseGradientReg;
        within {
            GivensPhaseGradientFrame(p, c, sysRegDown, g);
        } apply {
            use u = Qubit();
            use t = Qubit[n];
            within {
                AND(ctrl, second, u);
                for k in 0..n - 1 {
                    if Any(word -> k < Length(word) and word[k], firstWords + secondWords) {
                        within {
                            CNOT(u, ctrl);
                            for q in IndexRange(oneHotPairs) {
                                if secondWords[q][k] {
                                    CNOT(oneHotPairs[q], ctrl);
                                }
                                if firstWords[q][k] {
                                    CNOT(oneHotPairs[q], u);
                                }
                            }
                        } apply {
                            AND(u, ctrl, t[k]);
                        }
                        for q in IndexRange(oneHotPairs) {
                            if firstWords[q][k] and secondWords[q][k] {
                                CNOT(oneHotPairs[q], t[k]);
                            }
                        }
                    }
                }
            } apply {
                // Gidney's adder of t into g; carry[i] holds c_{i+1} and g[i], t[i] hold g ⊕ c_i, t ⊕ c_i while it is live.
                use carry = Qubit[MaxI(0, n - 2)];
                for i in 0..n - 2 {
                    if i > 0 {
                        CNOT(carry[i - 1], t[i]);
                        CNOT(carry[i - 1], g[i]);
                    }
                    if i < n - 2 {
                        AND(t[i], g[i], carry[i]);
                        if i > 0 {
                            CNOT(carry[i - 1], carry[i]);
                        }
                    }
                }
                // The gradient's top qubit is |−⟩, so flipping it on t ⊕ c_{n-1} is a phase, and
                // c_{n-1} = c_{n-2} ⊕ (t ⊕ c_{n-2})(g ⊕ c_{n-2}) needs no Toffoli.
                Z(t[n - 1]);
                if n > 1 {
                    CZ(t[n - 2], g[n - 2]);
                }
                if n > 2 {
                    Z(carry[n - 3]);
                }
                for i in n - 2..-1..0 {
                    if i < n - 2 {
                        if i > 0 {
                            CNOT(carry[i - 1], carry[i]);
                        }
                        Adjoint AND(t[i], g[i], carry[i]);
                    }
                    CNOT(t[i], g[i]);
                    if i > 0 {
                        CNOT(carry[i - 1], t[i]);
                        CNOT(carry[i - 1], g[i]);
                    }
                }
            }
        }
    }

    /// Streams one window of full-angle words through controlled-adder Givens steps, erasing each word
    /// right after its step so its qubits are the clean ancillas of the next (:cite:`Low2026` Lemma 6);
    /// the adjoint loads the window again and unrotates rather than inverting the measurements.
    internal operation ApplyStreamedControlledAdderBatch(
        sfData : Bool[][],
        dqData : Bool[][],
        sfSwapBits : Int,
        dqSwapBits : Int,
        useSkewTree : Bool,
        sfFields : (Int, Int)[],
        sfAddress : Qubit[],
        dqAddress : Qubit[],
        isSF : Qubit,
        dirty : Qubit[],
        first : Int,
        last : Int,
        wordBits : Int,
        pairs : (Int, Int)[],
        sysRegDown : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj {
        body (...) {
            StreamControlledAdderBatch(sfData, dqData, sfSwapBits, dqSwapBits, useSkewTree, sfFields, sfAddress, dqAddress, isSF, dirty, first, last, wordBits, pairs, sysRegDown, phaseGradientReg, true);
        }
        adjoint (...) {
            StreamControlledAdderBatch(sfData, dqData, sfSwapBits, dqSwapBits, useSkewTree, sfFields, sfAddress, dqAddress, isSF, dirty, first, last, wordBits, pairs, sysRegDown, phaseGradientReg, false);
        }
    }

    /// Loads the window, applies its steps (descending if `forward`, else ascending and adjoint), measures each
    /// word out after use and repairs the branch phases once at the end; the first step has no clean ancillas.
    internal operation StreamControlledAdderBatch(
        sfData : Bool[][],
        dqData : Bool[][],
        sfSwapBits : Int,
        dqSwapBits : Int,
        useSkewTree : Bool,
        sfFields : (Int, Int)[],
        sfAddress : Qubit[],
        dqAddress : Qubit[],
        isSF : Qubit,
        dirty : Qubit[],
        first : Int,
        last : Int,
        wordBits : Int,
        pairs : (Int, Int)[],
        sysRegDown : Qubit[],
        phaseGradientReg : Qubit[],
        forward : Bool,
    ) : Unit {
        use rotTarget = Qubit[(last - first + 1) * wordBits];
        ControlledSelectWithUnlookup(sfData, dqData, sfSwapBits, dqSwapBits, useSkewTree, sfFields, sfAddress, dqAddress, isSF, dirty, rotTarget);
        mutable measured = [false, size = Length(rotTarget)];
        mutable clean : Qubit[] = [];
        let order = if forward { last..-1..first } else { first..last };
        for j in order {
            let offset = (j - first) * wordBits;
            let word = rotTarget[offset..offset + wordBits - 1];
            let (p, c) = pairs[j];
            if forward {
                ApplyControlledAdderGivensStep(p, c, sysRegDown, word, phaseGradientReg, clean);
            } else {
                Adjoint ApplyControlledAdderGivensStep(p, c, sysRegDown, word, phaseGradientReg, clean);
            }
            let bits = ResultArrayAsBoolArray(ForEach(MResetX, word));
            for k in 0..wordBits - 1 {
                set measured w/= offset + k <- bits[k];
            }
            if IsEmpty(clean) {
                // A compressed word is one bit short of the gradient, exactly the b - 1 ancillas a step needs.
                set clean = word[0..Length(phaseGradientReg) - 2];
            }
        }
        // The fixups are diagonal on the address, which no step touches, so one pass after all erasures suffices.
        ApplyRotationWordFixup(measured, sfData, dqData, sfSwapBits, dqSwapBits, useSkewTree, sfFields, sfAddress, dqAddress, isSF);
    }

    /// Pads every row of `table` with false bits out to `width`.
    internal function PaddedRows(table : Bool[][], width : Int) : Bool[][] {
        MappedOverRange(i -> Padded(-width, false, table[i]), 0..Length(table) - 1)
    }

    /// Row-wise XOR of two equally shaped tables.
    internal function XorTables(a : Bool[][], b : Bool[][]) : Bool[][] {
        MappedOverRange(i -> MappedOverRange(c -> a[i][c] != b[i][c], 0..Length(a[i]) - 1), 0..Length(a) - 1)
    }

    /// Table XORed in by round `step` of an amortized sweep over `tables` taken in `order`.
    /// Unitary rounds load `θ_k ⊕ θ_{k-1}` so the sweep telescopes back to zero; measured rounds
    /// load `θ_k` and finish with `⊕_k θ_k`, which restores the lenders (:cite:`Low2026` E14).
    internal function AmortizedRoundTable(tables : Bool[][][], order : Int[], step : Int, measure : Bool) : Bool[][] {
        let numBatches = Length(order);
        let zero = XorTables(tables[0], tables[0]);
        if measure {
            if step < numBatches { tables[order[step]] } else { Fold(XorTables, zero, tables) }
        } else {
            let current = if step < numBatches { tables[order[step]] } else { zero };
            let previous = if step > 0 { tables[order[step - 1]] } else { zero };
            XorTables(current, previous)
        }
    }

    /// Toffoli cost of one amortized round: `ceil(d/K) - 1` for the controlled `Select` plus
    /// `2*b*(K-1)` for the swap network and its inverse (:cite:`Low2026` E13).
    internal function AmortizedDirtyRoundCost(numSwapBits : Int, numData : Int, numBits : Int) : Int {
        let blockSize = 1 <<< numSwapBits;
        (numData + blockSize - 1) / blockSize - 1 + 2 * numBits * (blockSize - 1)
    }

    /// Best amortized swap width whose `b*(K-1)` lenders fit in `availableDirty`; 0 means plain `Select`.
    internal function AmortizedDirtySwapBits(numData : Int, numBits : Int, availableDirty : Int, maxSwapBits : Int) : Int {
        let maxK = MaxSelectableSwapBits(Ceiling(Lg(IntAsDouble(numData))), maxSwapBits);
        mutable best = AmortizedDirtyRoundCost(0, numData, numBits);
        mutable bestBits = 0;
        for swapBits in 1..maxK {
            if numBits * ((1 <<< swapBits) - 1) <= availableDirty {
                let cost = AmortizedDirtyRoundCost(swapBits, numData, numBits);
                if cost < best {
                    set best = cost;
                    set bestBits = swapBits;
                }
            }
        }
        bestBits
    }

    /// XORs `data[address]` into `output` when `control` is set, with `output` as swap slot 0
    /// and `dirty` lending the other `K - 1` slots, which keep the rest of the row.
    internal operation AmortizedLookupRound(
        data : Bool[][],
        numSwapBits : Int,
        address : Qubit[],
        control : Qubit,
        dirty : Qubit[],
        output : Qubit[],
    ) : Unit {
        if numSwapBits == 0 {
            Controlled Select([control], (data, address, output));
        } else {
            let m = Length(output);
            let blockSize = 1 <<< numSwapBits;
            let nRequired = DimensionsForSelect(data, address);
            Fact(numSwapBits <= nRequired, "Too many bits for SWAP network");
            let slots = [output] + Chunks(m, dirty[...m * (blockSize - 1) - 1]);
            let parts = Partitioned([numSwapBits, nRequired - numSwapBits], address[...nRequired - 1]);
            let aliased = AliasToAddressSpace(data, nRequired);
            let rows = MappedOverRange(
                s -> Flattened(aliased[s * blockSize..(s + 1) * blockSize - 1]),
                0..(Length(data) + blockSize - 1) / blockSize - 1
            );
            // The inverse butterfly parks `output` in slot `address mod K`, which the row fills with `data[address]`.
            within {
                Adjoint SwapDataOutputs(parts[0], slots);
            } apply {
                Controlled Select([control], (rows, parts[1], Flattened(slots)));
            }
        }
    }

    /// Streams the batch words of both branches through one shared word in `Length(order) + 1`
    /// lookups (`Length(order)` when measured without lenders), running `visit(t, word)` with
    /// batch `t` resident (:cite:`Low2026` E13/E14).
    /// The adjoint replays the batches in reverse with `Adjoint visit`; neither direction runs a
    /// lookup backwards, because `Adjoint Select` would measure a word that holds lender garbage.
    internal operation AmortizedRotationWordSweep(
        sfTables : Bool[][][],
        dqTables : Bool[][][],
        order : Int[],
        sfSwapBits : Int,
        dqSwapBits : Int,
        sfAddress : Qubit[],
        dqAddress : Qubit[],
        isSF : Qubit,
        lenders : Qubit[],
        measure : Bool,
        visit : ((Int, Qubit[]) => Unit is Adj),
    ) : Unit is Adj {
        body (...) {
            AmortizedRotationWordSweepImpl(sfTables, dqTables, order, sfSwapBits, dqSwapBits, sfAddress, dqAddress, isSF, lenders, measure, visit, false);
        }
        adjoint (...) {
            AmortizedRotationWordSweepImpl(sfTables, dqTables, Reversed(order), sfSwapBits, dqSwapBits, sfAddress, dqAddress, isSF, lenders, measure, visit, true);
        }
    }

    internal operation AmortizedRotationWordSweepImpl(
        sfTables : Bool[][][],
        dqTables : Bool[][][],
        order : Int[],
        sfSwapBits : Int,
        dqSwapBits : Int,
        sfAddress : Qubit[],
        dqAddress : Qubit[],
        isSF : Qubit,
        lenders : Qubit[],
        measure : Bool,
        visit : ((Int, Qubit[]) => Unit is Adj),
        adjointVisit : Bool,
    ) : Unit {
        let numBatches = Length(order);
        // A measured sweep without lenders has nothing to restore, so it skips the closing round.
        let lastStep = if measure and sfSwapBits == 0 and dqSwapBits == 0 { numBatches - 1 } else { numBatches };
        let sfRounds = MappedOverRange(step -> AmortizedRoundTable(sfTables, order, step, measure), 0..lastStep);
        let dqRounds = MappedOverRange(step -> AmortizedRoundTable(dqTables, order, step, measure), 0..lastStep);
        use word = Qubit[Length(sfTables[0][0])];
        mutable measured : Bool[] = [];
        for step in 0..lastStep {
            AmortizedLookupRound(sfRounds[step], sfSwapBits, sfAddress, isSF, lenders, word);
            within { X(isSF); } apply {
                AmortizedLookupRound(dqRounds[step], dqSwapBits, dqAddress, isSF, lenders, word);
            }
            if step < numBatches {
                if adjointVisit { Adjoint visit(order[step], word); } else { visit(order[step], word); }
            }
            if measure {
                // In place: `word` is slot 0 of the next round's butterfly, so it cannot be released early.
                set measured += ResultArrayAsBoolArray(ForEach(MResetX, word));
            }
        }
        if measure {
            // Each round's X outcomes leave a parity against the word it held; one fixup per branch repairs them all.
            let sfAll = MappedOverRange(i -> Flattened(MappedOverRange(step -> sfRounds[step][i], 0..lastStep)), 0..Length(sfTables[0]) - 1);
            let dqAll = MappedOverRange(i -> Flattened(MappedOverRange(step -> dqRounds[step][i], 0..lastStep)), 0..Length(dqTables[0]) - 1);
            ApplyBranchPhaseFixup(measured, sfAll, true, [isSF] + sfAddress);
            ApplyBranchPhaseFixup(measured, dqAll, false, [isSF] + dqAddress);
        }
    }

    /// Loads one batch of angle words like `ControlledSelectWithUnlookup`, but erases each word by
    /// measurement right after its rotation and releases it, so later words reuse its qubits.
    internal operation MeasuredRotationWordBatch(
        sfData : Bool[][],
        dqData : Bool[][],
        sfSwapBits : Int,
        dqSwapBits : Int,
        useSkewTree : Bool,
        sfFields : (Int, Int)[],
        sfAddress : Qubit[],
        dqAddress : Qubit[],
        isSF : Qubit,
        dirty : Qubit[],
        first : Int,
        last : Int,
        bRot : Int,
        pairs : (Int, Int)[],
        sysRegDown : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj {
        body (...) {
            MeasuredRotationWordBatchImpl(sfData, dqData, sfSwapBits, dqSwapBits, useSkewTree, sfFields, sfAddress, dqAddress, isSF, dirty, first, last, bRot, pairs, sysRegDown, phaseGradientReg, false);
        }
        adjoint (...) {
            MeasuredRotationWordBatchImpl(sfData, dqData, sfSwapBits, dqSwapBits, useSkewTree, sfFields, sfAddress, dqAddress, isSF, dirty, first, last, bRot, pairs, sysRegDown, phaseGradientReg, true);
        }
    }

    internal operation MeasuredRotationWordBatchImpl(
        sfData : Bool[][],
        dqData : Bool[][],
        sfSwapBits : Int,
        dqSwapBits : Int,
        useSkewTree : Bool,
        sfFields : (Int, Int)[],
        sfAddress : Qubit[],
        dqAddress : Qubit[],
        isSF : Qubit,
        dirty : Qubit[],
        first : Int,
        last : Int,
        bRot : Int,
        pairs : (Int, Int)[],
        sysRegDown : Qubit[],
        phaseGradientReg : Qubit[],
        reverse : Bool,
    ) : Unit {
        let numWords = last - first + 1;
        // The word rotated first is allocated innermost: highest index forward, lowest in reverse.
        let allocationOrder = if reverse { MappedOverRange(i -> numWords - 1 - i, 0..numWords - 1) } else { MappedOverRange(i -> i, 0..numWords - 1) };
        let outcomes = LoadRotateAndMeasureWords(
            allocationOrder,
            [],
            sfData,
            dqData,
            sfSwapBits,
            dqSwapBits,
            useSkewTree,
            sfFields,
            sfAddress,
            dqAddress,
            isSF,
            dirty,
            first,
            bRot,
            pairs,
            sysRegDown,
            phaseGradientReg,
            reverse
        );
        // Outcomes come back innermost first, i.e. reversed allocation order, so this restores index order.
        let measured = Flattened(if reverse { outcomes } else { Reversed(outcomes) });
        ApplyRotationWordFixup(measured, sfData, dqData, sfSwapBits, dqSwapBits, useSkewTree, sfFields, sfAddress, dqAddress, isSF);
    }

    /// Allocates the words in `pending` one scope each, loads the batch once all exist, then
    /// rotates, measures and releases them innermost first; returns the X outcomes innermost first.
    internal operation LoadRotateAndMeasureWords(
        pending : Int[],
        allocated : Qubit[][],
        sfData : Bool[][],
        dqData : Bool[][],
        sfSwapBits : Int,
        dqSwapBits : Int,
        useSkewTree : Bool,
        sfFields : (Int, Int)[],
        sfAddress : Qubit[],
        dqAddress : Qubit[],
        isSF : Qubit,
        dirty : Qubit[],
        first : Int,
        bRot : Int,
        pairs : (Int, Int)[],
        sysRegDown : Qubit[],
        phaseGradientReg : Qubit[],
        reverse : Bool,
    ) : Bool[][] {
        if IsEmpty(pending) {
            // Words are allocated in index order forward and reverse index order backward.
            let target = Flattened(if reverse { Reversed(allocated) } else { allocated });
            ControlledSelectWithUnlookup(sfData, dqData, sfSwapBits, dqSwapBits, useSkewTree, sfFields, sfAddress, dqAddress, isSF, dirty, target);
            return [];
        }
        let j = pending[0];
        use word = Qubit[bRot];
        let inner = LoadRotateAndMeasureWords(
            pending[1...],
            allocated + [word],
            sfData,
            dqData,
            sfSwapBits,
            dqSwapBits,
            useSkewTree,
            sfFields,
            sfAddress,
            dqAddress,
            isSF,
            dirty,
            first,
            bRot,
            pairs,
            sysRegDown,
            phaseGradientReg,
            reverse
        );
        if reverse {
            Adjoint ApplyGivensRotationWords(word, first + j, first + j, bRot, pairs, sysRegDown, phaseGradientReg, false);
        } else {
            ApplyGivensRotationWords(word, first + j, first + j, bRot, pairs, sysRegDown, phaseGradientReg, false);
        }
        let bits = ResultArrayAsBoolArray(ForEach(MResetX, word));
        return inner + [bits];
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
    /// `maxSwapBits` caps the QROAM swap width (-1 free, 0 plain); `borrowDirty` lends `systemReg`.
    function MakeInnerPrepareAliasSampling(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
        maxSwapBits : Int,
        borrowDirty : Bool,
    ) : (Qubit[], Qubit[], Qubit[]) => Unit is Adj {
        let nCoeffs = Length(innerCoefficients[0]);
        // Keep at least one b-register qubit, matching MakeInnerPrepareDirect and the Python layout.
        let nIndexBits = MaxI(1, AddressQubits(nCoeffs));
        let mu = coefficientBitPrecision;
        let nFreeRider = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        let qromEnd = 2 * nIndexBits + 2 * mu + 2;
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
                maxSwapBits,
                outerReg,
                indexReg,
                uniformReg,
                flagQubit,
                qromOut,
                freeRiderReg,
                if borrowDirty { dirty } else { [] }
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
    /// Both layouts are costed at the width `ConditionalAliasSwapBits` builds the PREPARE with.
    internal function ShouldLoadFreeRiderSeparately(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
        maxSwapBits : Int,
        availableDirty : Int,
    ) : Bool {
        let numConditions = Length(innerCoefficients);
        let nIndexBits = MaxI(1, AddressQubits(Length(innerCoefficients[0])));
        let numInnerSlots = 1 <<< nIndexBits;
        let numWordBits = coefficientBitPrecision + nIndexBits + 2;
        let numExtraBits = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        let innerPreparePairsPerBlock = 2;
        let freeRiderPairsPerBlock = 1;
        mutable innerCosts = [];
        for numBits in [numWordBits + numExtraBits, numWordBits] {
            let lambda = ConditionalAliasSwapBits(numConditions, numInnerSlots, numBits, maxSwapBits, availableDirty);
            let cost = if availableDirty > 0 {
                SelectSwapDirtyCost2D(lambda, numConditions, numInnerSlots, numBits, true)
            } else {
                SelectSwapCost2D(lambda, numConditions, numInnerSlots, numBits, true)
            };
            set innerCosts += [innerPreparePairsPerBlock * cost];
        }
        let separateCost = innerCosts[1]
            + freeRiderPairsPerBlock * SelectSwapCost2D(0, numConditions, 1, numExtraBits, true);
        numExtraBits > 0 and separateCost < innerCosts[0]
    }

    /// Build the inner alias-sampling PREPARE and its free-rider loader together.
    ///
    /// Carrying the free-rider word widens the QROAM output charged on each of the two inner
    /// PREPARE/uncompute pairs in one block encoding. Loading it separately costs one `Select`
    /// round trip over the outer conditions but lets the inner table use a narrower output
    /// and potentially a different swap width.
    /// `availableDirty` is the lender size the walk passes the inner PREPARE, or 0 for clean scratch.
    function MakeInnerPrepareAliasSamplingOracles(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
        maxSwapBits : Int,
        availableDirty : Int,
    ) : (
        ((Qubit[], Qubit[], Qubit[]) => Unit is Adj),
        ((Qubit[], Qubit[]) => Unit is Adj + Ctl)
    ) {
        let loadSeparately = ShouldLoadFreeRiderSeparately(
            innerCoefficients,
            freeRiderData,
            coefficientBitPrecision,
            maxSwapBits,
            availableDirty
        );
        let inlineData = if loadSeparately { [] } else { freeRiderData };
        let separateData = if loadSeparately { freeRiderData } else { [] };

        (
            MakeInnerPrepareAliasSampling(
                innerCoefficients,
                inlineData,
                coefficientBitPrecision,
                maxSwapBits,
                availableDirty > 0
            ),
            MakeFreeRiderLoadOp(separateData)
        )
    }

    /// Decode `address` into `oneHot` and CNOT each condition's free-rider word out of it.
    ///
    /// Each address bit doubles the decoded prefix with one AND per new one-hot qubit, X_o - 1 in
    /// all, so the adjoint frees the register by measurement with no Toffoli; `ctls` gate the seed.
    operation LoadOneHotWithFreeRider(
        freeRiderData : Bool[][],
        address : Qubit[],
        oneHot : Qubit[],
        freeRider : Qubit[],
    ) : Unit is Adj + Ctl {
        body (...) {
            Controlled LoadOneHotWithFreeRider([], (freeRiderData, address, oneHot, freeRider));
        }
        controlled (ctls, ...) {
            Controlled X(ctls, oneHot[0]);
            for k in 0..AddressQubits(Length(oneHot)) - 1 {
                for i in 0..MinI(1 <<< k, Length(oneHot) - (1 <<< k)) - 1 {
                    ExactAnd(oneHot[i], address[k], oneHot[i + (1 <<< k)]);
                    CNOT(oneHot[i + (1 <<< k)], oneHot[i]);
                }
            }
            for (bits, condition) in Zipped(freeRiderData, oneHot) {
                for (bit, target) in Zipped(bits, freeRider) {
                    if bit {
                        CNOT(condition, target);
                    }
                }
            }
        }
    }

    /// Build the one-hot inner PREPARE and the loader that decodes x_o for a whole block encoding.
    ///
    /// innerReg layout: indexReg[nIdx] + uniformReg[μ] + flagQubit[1] + qromOutput[μ + nIdx + 2]
    /// + oneHot[X_o] + freeRiderReg[nFR]. The loader owns oneHot + freeRiderReg, so both inner
    /// PREPARE/uncompute pairs of a block read one decode, as in arXiv:2611.09822 App. I.
    function MakeInnerPrepareOneHotOracles(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
    ) : (
        ((Qubit[], Qubit[], Qubit[]) => Unit is Adj),
        ((Qubit[], Qubit[]) => Unit is Adj + Ctl)
    ) {
        let numConditions = Length(innerCoefficients);
        let nIndexBits = MaxI(1, AddressQubits(Length(innerCoefficients[0])));
        let mu = coefficientBitPrecision;
        let qromEnd = 2 * nIndexBits + 2 * mu + 2;
        (
            (_, innerReg, _) => ConditionalAliasSamplingPrepareOneHot(
                innerCoefficients,
                mu,
                innerReg[qromEnd + 1..qromEnd + numConditions],
                innerReg[0..nIndexBits - 1],
                innerReg[nIndexBits..nIndexBits + mu - 1],
                innerReg[nIndexBits + mu],
                innerReg[nIndexBits + mu + 1..qromEnd]
            ),
            (outerReg, freeRiderReg) => LoadOneHotWithFreeRider(
                freeRiderData,
                outerReg,
                freeRiderReg[...numConditions - 1],
                freeRiderReg[numConditions...]
            )
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
        // No lookup table here; the lender is accepted only to keep one inner-PREPARE signature.
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

    /// Controlled-W^power as the `(control, targets)` callable standard and iterative QPE consume.
    ///
    /// Target layout: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    function MakeControlledSOSSAWalkOp(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        cacheName : String,
        power : Int,
    ) : ((Qubit, Qubit[]) => Unit is Adj) {
        ControlledSOSSAWalkOnRegister(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, cacheName, power, _, _)
    }

    /// Circuit entry point: allocates the control and the flat target register and applies
    /// controlled-W^power once.
    operation MakeControlledSOSSAWalkCircuit(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        cacheName : String,
        power : Int,
    ) : Unit {
        let bounds = SOSSAWalkRegisterBounds(layout);
        use control = Qubit();
        use allQubits = Qubit[bounds[4]];
        if layout.numPhaseGradientQubits > 0 {
            PreparePhaseGradientState(allQubits[bounds[3]...]);
        }
        ControlledSOSSAWalkOnRegister(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, cacheName, power, control, allQubits);
        ResetAll([control] + allQubits);
    }
}
