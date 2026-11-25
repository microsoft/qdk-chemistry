// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Inner PREPARE and free-rider loading for SOSSA.

namespace QDKChemistry.Utils.SOSSAWalk {

    import Std.Arrays.Padded;
    import Std.Arrays.Reversed;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Core.Length;
    import Std.Math.MaxI;
    import Std.Math.MinI;
    import Std.StatePreparation.PreparePureStateD;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepareWithFreeRider;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSwapBits;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepareOneHot;
    import QDKChemistry.Utils.SelectSwap.SelectSwapCost2D;
    import QDKChemistry.Utils.SelectSwapDirty.SelectSwapDirtyCost2D;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;
    import QDKChemistry.Utils.UnaryIteration.ExactAnd;

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
}
