// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Rotation-word lookup, streaming, and measurement fix-ups for SOSSA.

namespace QDKChemistry.Utils.SOSSAWalk {

    import Std.Arrays.All;
    import Std.Arrays.Chunks;
    import Std.Arrays.Filtered;
    import Std.Arrays.Flattened;
    import Std.Arrays.Fold;
    import Std.Arrays.ForEach;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.Mapped;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Padded;
    import Std.Arrays.Reversed;
    import Std.Arrays.Partitioned;
    import Std.Arrays.Subarray;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.IntAsDouble;
    import Std.Convert.ResultArrayAsBoolArray;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Math.MaxI;
    import Std.Math.MinI;
    import Std.Measurement.MResetX;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.SelectSwap.ApplyBranchPhaseFixup;
    import QDKChemistry.Utils.SelectSwap.ComputeOptimalSwapBits;
    import QDKChemistry.Utils.SelectSwap.SelectSwapAliased;
    import QDKChemistry.Utils.SelectSwap.SelectSwapForwardCost;
    import QDKChemistry.Utils.SelectSwapDirty.SelectSwapDirtyCost1D;
    import QDKChemistry.Utils.SelectSwap.AliasToAddressSpace;
    import QDKChemistry.Utils.SelectSwap.DimensionsForSelect;
    import QDKChemistry.Utils.SelectSwap.MaxSelectableSwapBits;
    import QDKChemistry.Utils.SelectSwap.SwapDataOutputs;
    import QDKChemistry.Utils.SelectSwapDirty.ComputeOptimalDirtySwapBits;
    import QDKChemistry.Utils.SelectSwapDirty.SelectSwapDirty;
    import QDKChemistry.Utils.SkewTree.SkewTreeBoxSize;
    import QDKChemistry.Utils.SkewTree.SkewTreeLookup;
    import QDKChemistry.Utils.SkewTree.SkewTreeOutputTable;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;

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
}
