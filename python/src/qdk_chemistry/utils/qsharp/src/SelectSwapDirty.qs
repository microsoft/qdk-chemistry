// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SELECT-SWAP loaders that borrow live caller qubits instead of allocating scratch.
/// Reference: Berry et al. (arXiv:1902.02134) App. A Thm 1; Low et al. (arXiv:1812.00954).
namespace QDKChemistry.Utils.SelectSwapDirty {

    import Std.Arrays.All;
    import Std.Arrays.Chunks;
    import Std.Arrays.Flattened;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Partitioned;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyToEachCA;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Math.MaxI;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.SelectSwap.AliasToAddressSpace;
    import QDKChemistry.Utils.SelectSwap.DimensionsForSelect;
    import QDKChemistry.Utils.SelectSwap.SwapDataOutputs;
    import QDKChemistry.Utils.SelectSwap.MaxSelectableSwapBits;
    import QDKChemistry.Utils.SelectSwap.MeasurementUnlookupCost;
    import QDKChemistry.Utils.SelectSwap.SelectSwap2D;

    /// Toffoli cost of one borrowed `SelectSwap2D` load and its uncompute: Berry et al.'s
    /// `2*ceil(d/K) + 4*b*(K-1)` forward pass, plus one measurement erasure of the target.
    internal function SelectSwapDirtyCost2D(
        lambda : Int,
        numOuterData : Int,
        numInnerData : Int,
        numBits : Int,
        outerAddressAlwaysValid : Bool,
    ) : Int {
        let outerAddressBits = Ceiling(Lg(IntAsDouble(numOuterData)));
        let innerAddressBits = Ceiling(Lg(IntAsDouble(numInnerData)));
        let outerBlocks = if outerAddressAlwaysValid { numOuterData } else { 2^outerAddressBits };
        let eraseCost = MeasurementUnlookupCost(outerAddressBits + innerAddressBits);

        if lambda == 0 {
            // Width 0 is one plain `Select`, not the two-pass `K = 1` limit of the swap formula.
            MaxI(0, outerBlocks * 2^innerAddressBits - 2) + eraseCost
        } else {
            let numEntries = outerBlocks * 2^(innerAddressBits - lambda);
            // Both passes are uncontrolled, so each is `numEntries - 2`, and free once one entry is left.
            let selectCost = 2 * MaxI(0, numEntries - 2);
            let swapCost = 4 * numBits * (2^lambda - 1);
            selectCost + swapCost + eraseCost
        }
    }

    /// Best dirty swap width for the 2D lookup that fits in `availableDirty` and `maxSwapBits` (-1 for any); 0 means plain `Select`.
    function ComputeOptimalDirtySwapBits2D(
        numOuterData : Int,
        numInnerData : Int,
        numBits : Int,
        outerAddressAlwaysValid : Bool,
        availableDirty : Int,
        maxSwapBits : Int,
    ) : Int {
        let maxLambda = MaxSelectableSwapBits(Ceiling(Lg(IntAsDouble(numInnerData))), maxSwapBits);
        mutable best = SelectSwapDirtyCost2D(0, numOuterData, numInnerData, numBits, outerAddressAlwaysValid);
        mutable bestLambda = 0;
        for lambda in 1..maxLambda {
            if SelectSwapDirtyBorrowedQubits(lambda, numBits) <= availableDirty {
                let cost = SelectSwapDirtyCost2D(
                    lambda,
                    numOuterData,
                    numInnerData,
                    numBits,
                    outerAddressAlwaysValid
                );
                if cost < best {
                    set best = cost;
                    set bestLambda = lambda;
                }
            }
        }
        bestLambda
    }

    /// Qubits a `SelectSwapDirty` load borrows, `numBits * 2^numSwapBits`, or none at width 0.
    function SelectSwapDirtyBorrowedQubits(numSwapBits : Int, numBits : Int) : Int {
        if numSwapBits <= 0 { 0 } else { numBits * (1 <<< numSwapBits) }
    }

    /// Toffoli cost of the forward `SelectSwapDirty` load: `numData - 1` at width 0, otherwise
    /// two `Select` passes over `ceil(d/K)` rows plus four `numBits * (K - 1)` butterflies.
    internal function SelectSwapDirtyCost1D(numSwapBits : Int, numData : Int, numBits : Int) : Int {
        if numSwapBits == 0 {
            numData - 1
        } else {
            let blockSize = 1 <<< numSwapBits;
            let numRows = (numData + blockSize - 1) / blockSize;
            let selectCost = 2 * (numRows - 1);
            let swapCost = 4 * numBits * (blockSize - 1);
            selectCost + swapCost
        }
    }

    /// Best dirty swap width that fits in `availableDirty` and `maxSwapBits` (-1 for any); 0 (plain `Select`) when none pays.
    function ComputeOptimalDirtySwapBits(numData : Int, numBits : Int, availableDirty : Int, maxSwapBits : Int) : Int {
        let maxK = MaxSelectableSwapBits(Ceiling(Lg(IntAsDouble(numData))), maxSwapBits);
        mutable best = SelectSwapDirtyCost1D(0, numData, numBits);
        mutable bestBits = 0;
        for swapBits in 1..maxK {
            if SelectSwapDirtyBorrowedQubits(swapBits, numBits) <= availableDirty {
                let cost = SelectSwapDirtyCost1D(swapBits, numData, numBits);
                if cost < best {
                    set best = cost;
                    set bestBits = swapBits;
                }
            }
        }
        bestBits
    }

    /// XORs `data[address]` into `output` on borrowed `dirty` qubits, returned exactly as they
    /// arrived: two `Select` passes cancel the unknown chunk between four butterflies.
    operation SelectSwapDirty(
        numSwapBits : Int,
        data : Bool[][],
        address : Qubit[],
        dirty : Qubit[],
        output : Qubit[],
    ) : Unit is Adj + Ctl {
        body (...) {
            Controlled SelectSwapDirty([], (numSwapBits, data, address, dirty, output));
        }
        controlled (controls, ...) {
            Fact(not IsEmpty(data), "data cannot be empty");
            let nRequired = DimensionsForSelect(data, address);
            let m = Length(data[0]);
            Fact(Length(output) == m, $"output holds one {m}-bit word, got {Length(output)} qubits");
            Fact(numSwapBits <= nRequired, "Too many bits for SWAP network");

            if numSwapBits == 0 {
                Controlled Select(controls, (data, address[...nRequired - 1], output));
            } else {
                Fact(
                    Length(dirty) >= SelectSwapDirtyBorrowedQubits(numSwapBits, m),
                    $"dirty register needs {SelectSwapDirtyBorrowedQubits(numSwapBits, m)} qubits, got {Length(dirty)}"
                );
                let borrowed = dirty[...SelectSwapDirtyBorrowedQubits(numSwapBits, m) - 1];
                let chunks = Chunks(m, borrowed);
                let numSelectBits = nRequired - numSwapBits;
                // Swap bits low, so the select field is `a / K` and the table stops at ceil(d/K).
                let addressParts = Partitioned([numSwapBits, numSelectBits], address[...nRequired - 1]);
                let swapAddress = addressParts[0];
                let selectAddress = addressParts[1];
                // Contiguous `Select`-aliased blocks, each a unary-iteration subtree, so the table stops at ceil(d/K).
                let aliased = AliasToAddressSpace(data, nRequired);
                let blockSize = 1 <<< numSwapBits;
                let numRows = (Length(data) + blockSize - 1) / blockSize;
                let dataArray = MappedOverRange(
                    s -> Flattened(aliased[s * blockSize..(s + 1) * blockSize - 1]),
                    0..numRows - 1
                );

                within {
                    SwapDataOutputs(swapAddress, chunks);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(chunks[0], output));
                }

                Controlled Select(controls, (dataArray, selectAddress, borrowed));

                within {
                    SwapDataOutputs(swapAddress, chunks);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(chunks[0], output));
                }

                Controlled Select(controls, (dataArray, selectAddress, borrowed));
            }
        }
        adjoint self;
    }
}
