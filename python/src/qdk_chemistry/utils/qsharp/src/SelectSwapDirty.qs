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
    import Std.Canon.ApplyToEachA;
    import Std.Canon.ApplyToEachCA;
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Math.MaxI;
    import Std.Measurement.MResetEachZ;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.SelectSwap.AliasToAddressSpace;
    import QDKChemistry.Utils.SelectSwap.DimensionsForSelect;
    import QDKChemistry.Utils.SelectSwap.SwapDataOutputs;
    import QDKChemistry.Utils.SelectSwap.MeasurementUnlookupCost;
    import QDKChemistry.Utils.SelectSwap.EraseSwappedLoad;
    import QDKChemistry.Utils.SelectSwap.SelectSwap2D;
    import QDKChemistry.Utils.SelectSwap.SelectSwapCost2D;

    /// Toffoli cost of one borrowed `SelectSwap2D` load and its uncompute: Berry et al.'s
    /// `2*ceil(d/K) + 4*b*(K-1)` forward pass, plus one measurement erasure of the target.
    internal function DirtyQROAMCost2D(
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
            outerBlocks * 2^innerAddressBits - 2 + eraseCost
        } else {
            let numEntries = outerBlocks * 2^(innerAddressBits - lambda);
            let selectCost = 2 * (numEntries - 1);
            let swapCost = 4 * numBits * (2^lambda - 1);
            selectCost + swapCost + eraseCost
        }
    }

    /// Best dirty swap width for the 2D lookup that fits in `availableDirty`; 0 means plain `Select`.
    function ComputeOptimalDirtySwapBits2D(
        numOuterData : Int,
        numInnerData : Int,
        numBits : Int,
        outerAddressAlwaysValid : Bool,
        availableDirty : Int,
    ) : Int {
        let innerAddressBits = Ceiling(Lg(IntAsDouble(numInnerData)));
        mutable best = DirtyQROAMCost2D(0, numOuterData, numInnerData, numBits, outerAddressAlwaysValid);
        mutable bestLambda = 0;
        for lambda in 1..innerAddressBits {
            if DirtyQROAMBorrowedQubits(lambda, numBits) <= availableDirty {
                let cost = DirtyQROAMCost2D(
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

    /// Qubits a `SelectSwapDirty` load borrows, `numBits * 2^numSwapBits`, returned unchanged.
    function DirtyQROAMBorrowedQubits(numSwapBits : Int, numBits : Int) : Int {
        numBits * (1 <<< numSwapBits)
    }

    /// Toffoli cost of the forward `SelectSwapDirty` load: `numData - 1` at width 0, otherwise
    /// two `Select` passes over `ceil(d/K)` rows plus four `numBits * (K - 1)` butterflies.
    internal function DirtyQROAMCost(numSwapBits : Int, numData : Int, numBits : Int) : Int {
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

    /// Best dirty swap width that fits in `availableDirty`; 0 (plain `Select`) when none pays.
    function ComputeOptimalDirtySwapBits(numData : Int, numBits : Int, availableDirty : Int) : Int {
        let addressBits = Ceiling(Lg(IntAsDouble(numData)));
        mutable best = DirtyQROAMCost(0, numData, numBits);
        mutable bestBits = 0;
        for swapBits in 1..addressBits {
            if DirtyQROAMBorrowedQubits(swapBits, numBits) <= availableDirty {
                let cost = DirtyQROAMCost(swapBits, numData, numBits);
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
                    Length(dirty) >= DirtyQROAMBorrowedQubits(numSwapBits, m),
                    $"dirty register needs {DirtyQROAMBorrowedQubits(numSwapBits, m)} qubits, got {Length(dirty)}"
                );
                let borrowed = dirty[...DirtyQROAMBorrowedQubits(numSwapBits, m) - 1];
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

    // ═══ Test wrappers ══════════════════════════════════════════════════════════

    /// `SelectSwapDirty` matches a plain `Select` on every address and restores a seeded,
    /// non-trivial lender.
    internal operation TestSelectSwapDirtyCorrectness(
        data : Bool[][],
        numSwapBits : Int,
        dirtySeed : Int,
    ) : Bool {
        let m = Length(data[0]);
        let nAddr = Ceiling(Lg(IntAsDouble(Length(data))));
        let numDirty = DirtyQROAMBorrowedQubits(numSwapBits, m);

        use address = Qubit[nAddr];
        use dirty = Qubit[MaxI(1, numDirty)];
        use output = Qubit[m];
        use copy = Qubit[m];
        use reference = Qubit[m];

        mutable allCorrect = true;

        for addr in 0..2^nAddr - 1 {
            ApplyXorInPlace(addr, address);
            if numDirty > 0 {
                ApplyXorInPlace(dirtySeed % 2^numDirty, dirty[...numDirty - 1]);
            }

            within {
                SelectSwapDirty(numSwapBits, data, address, dirty, output);
            } apply {
                ApplyToEachCA(CNOT, Zipped(output, copy));
            }

            // Forward `Select` twice XORs to zero, so the reference avoids the measurement-based unlookup.
            Select(data, address, reference);
            ApplyToEachCA(CNOT, Zipped(reference, copy));
            Select(data, address, reference);

            let residue = MResetEachZ(copy);
            if not All(r -> r == Zero, residue) {
                Message($"FAIL word: addr={addr}, differs from Select in bits {residue}");
                set allCorrect = false;
            }

            // The lender must come back exactly as it went in.
            if numDirty > 0 {
                ApplyXorInPlace(dirtySeed % 2^numDirty, dirty[...numDirty - 1]);
                let dirtyResidue = MResetEachZ(dirty);
                if not All(r -> r == Zero, dirtyResidue) {
                    Message($"FAIL dirty disturbed: addr={addr}, residue={dirtyResidue}");
                    set allCorrect = false;
                }
            }
            ApplyXorInPlace(addr, address);
        }

        allCorrect
    }

    /// Dirty and width-0 loads agree as phase oracles over superposed address and lender,
    /// catching residual lender entanglement the basis-state check cannot see.
    internal operation TestSelectSwapDirtyPhaseAgreement(
        data : Bool[][],
        numSwapBits : Int
    ) : Bool {
        let m = Length(data[0]);
        let nAddr = Ceiling(Lg(IntAsDouble(Length(data))));
        let numDirty = MaxI(1, DirtyQROAMBorrowedQubits(numSwapBits, m));

        use address = Qubit[nAddr];
        use dirty = Qubit[numDirty];

        ApplyToEachA(H, address);
        ApplyToEachA(H, dirty);

        for swapBits in [numSwapBits, 0] {
            use output = Qubit[m];
            within {
                SelectSwapDirty(swapBits, data, address, dirty, output);
            } apply {
                Z(output[0]);
            }
        }

        Adjoint ApplyToEachA(H, dirty);
        Adjoint ApplyToEachA(H, address);

        All(r -> r == Zero, MResetEachZ(address + dirty))
    }

    /// A borrowed `SelectSwap2D` loads what the clean one loads on every address, including
    /// aliased surplus rows, and restores a seeded, non-trivial lender.
    internal operation TestSelectSwap2DDirtyMatchesClean(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
        dirtySeed : Int,
    ) : Bool {
        let m = Length(data[0][0]);
        let nOuterAddr = Ceiling(Lg(IntAsDouble(Length(data))));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(Length(data[0]))));
        let numDirty = DirtyQROAMBorrowedQubits(numSwapBits, m);

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        use dirty = Qubit[MaxI(1, numDirty)];
        use target = Qubit[m];
        use copy = Qubit[m];

        mutable allCorrect = true;

        for i in 0..2^nOuterAddr - 1 {
            for j in 0..2^nInnerAddr - 1 {
                ApplyXorInPlace(i, outerAddr);
                ApplyXorInPlace(j, innerAddr);
                if numDirty > 0 {
                    ApplyXorInPlace(dirtySeed % 2^numDirty, dirty[...numDirty - 1]);
                }

                // The erasure's phase on a basis-state address is global, so it cannot affect the comparison.
                within {
                    SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, dirty, target);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(target, copy));
                }
                within {
                    SelectSwap2D(data, 0, outerAddressAlwaysValid, outerAddr, innerAddr, [], target);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(target, copy));
                }

                let residue = MResetEachZ(copy);
                if not All(r -> r == Zero, residue) {
                    Message($"FAIL word: outer={i}, inner={j}, differs from clean in {residue}");
                    set allCorrect = false;
                }

                // The lender must come back exactly as it went in.
                if numDirty > 0 {
                    ApplyXorInPlace(dirtySeed % 2^numDirty, dirty[...numDirty - 1]);
                    let dirtyResidue = MResetEachZ(dirty);
                    if not All(r -> r == Zero, dirtyResidue) {
                        Message($"FAIL dirty disturbed: outer={i}, inner={j}, residue={dirtyResidue}");
                        set allCorrect = false;
                    }
                }

                ApplyXorInPlace(i, outerAddr);
                ApplyXorInPlace(j, innerAddr);
            }
        }

        allCorrect
    }

}
