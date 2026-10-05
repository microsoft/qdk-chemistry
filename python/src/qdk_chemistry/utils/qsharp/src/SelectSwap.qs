// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SELECT-SWAP network for efficient QROM data loading (1D and 2D).
///
/// This file holds the *clean* strategy: the swap block is allocated as scratch, so the
/// network costs width the caller did not already have. Its siblings are
/// `SelectSwapDirty` (borrows live caller qubits instead) and `SelectSwapUtils` (the table
/// and address-space helpers both strategies share). The lookup-method tags below route
/// between them and stay here, with the loaders they dispatch to.
///
/// 1D operations:
///   SelectSwap — loads data[address] into output.
///   ApplyBranchPhaseFixup — repairs one branch's phase after a measurement-based erasure.
///
/// 2D operations:
///   SelectSwap2D — loads data[outer][inner] with one select-swap over the combined address,
///     on allocated scratch or, given a lender, on borrowed qubits
///   ComputeOptimalLambda2D — optimal SWAP bits for 2D case
///
/// References:
///   Low, Kliuchnikov, Schaeffer (arXiv:1812.00954)
namespace QDKChemistry.Utils.SelectSwap {

    import Std.Arrays.All;
    import Std.Arrays.Chunks;
    import Std.Arrays.Mapped;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Partitioned;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyToEachA;
    import Std.Canon.ApplyToEachCA;
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsDouble;
    import Std.Convert.ResultAsBool;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Floor;
    import Std.Math.Lg;
    import Std.Math.MaxI;
    import Std.Math.MinI;
    import Std.Measurement.MResetEachZ;
    import Std.StatePreparation.PrepareUniformSuperposition;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.UnaryIteration.UnaryIteration;
    import QDKChemistry.Utils.UnaryIteration.UnaryIterationActionIndex;
    import QDKChemistry.Utils.SelectSwapUtils.PadToAddressSpace;
    import QDKChemistry.Utils.SelectSwapUtils.AliasToAddressSpace;
    import QDKChemistry.Utils.SelectSwapUtils.SwappedLoadShape;
    import QDKChemistry.Utils.SelectSwapUtils.SwapPermutedTable;
    import QDKChemistry.Utils.SelectSwapUtils.DimensionsForSelect;
    import QDKChemistry.Utils.SelectSwapUtils.CreatePaddedData;
    import QDKChemistry.Utils.SelectSwapUtils.SwapDataOutputs;
    import QDKChemistry.Utils.SelectSwapUtils.MeasurementUnlookupCost;

    /// Narrowest swap width within 20% of the cheapest one's Toffolis.
    ///
    /// Returns the *narrowest* qualifying width rather than the cheapest, so a width is taken
    /// only when the extra scratch is paying for itself. Widths are scanned in order, so the
    /// first qualifier is the narrowest by construction.
    function ComputeOptimalLambda2D(
        numOuterData : Int,
        numInnerData : Int,
        numBits : Int,
        outerAddressAlwaysValid : Bool,
    ) : Int {
        let addressBits = Ceiling(Lg(IntAsDouble(numInnerData)));

        mutable best = SelectSwapCost2D(0, numOuterData, numInnerData, numBits, outerAddressAlwaysValid);
        for lambda in 1..addressBits - 1 {
            let cost = SelectSwapCost2D(lambda, numOuterData, numInnerData, numBits, outerAddressAlwaysValid);
            if cost < best {
                set best = cost;
            }
        }

        // Largest relative Toffoli premium worth paying for a narrower swap network, since scoring
        // widths by Toffolis alone never sees the scratch each extra swap bit doubles.
        // At the inner-PREPARE shape `d = 90`, `m = 16`, `b = 21` it takes `k = 2` at 587 Toffolis
        // over the minimum `k = 3` at 491: 19.6% more Toffolis for 30% less scratch.
        // The margin is narrow -- below about 0.195 the rule takes `k = 3` -- so 0.2 is load-bearing.
        let maxToffoliPremium = 0.2;
        let threshold = IntAsDouble(best) * (1.0 + maxToffoliPremium);
        for lambda in 0..addressBits - 1 {
            let cost = SelectSwapCost2D(lambda, numOuterData, numInnerData, numBits, outerAddressAlwaysValid);
            if IntAsDouble(cost) <= threshold {
                return lambda;
            }
        }

        return 0;
    }

    internal function ComputeOptimalLambda1D(numData : Int, numBits : Int) : Int {
        mutable best = 2^32;
        mutable bestLambda = 0;

        let addressBits = Ceiling(Lg(IntAsDouble(numData)));
        for lambda in 0..addressBits - 1 {
            let cost = SelectSwapCost1D(lambda, numData, numBits);
            if cost < best {
                set bestLambda = lambda;
                set best = cost;
            }
        }

        return bestLambda;
    }

    internal function SelectSwapCost1D(lambda : Int, numData : Int, numBits : Int) : Int {
        if lambda == 0 {
            return numData - 2;
        } else {
            let addressBits = Ceiling(Lg(IntAsDouble(numData)));
            let split = MinI(Floor(Lg(IntAsDouble(2^lambda * numBits))), addressBits - 1);

            let select_cost = 2^(addressBits - lambda) - 2;
            let unselect_cost = MaxI(0, 2^split - 2) + 2^(addressBits - split) - 2;
            let swap_cost = (2^lambda - 1) * numBits;

            return select_cost + unselect_cost + swap_cost;
        }
    }

    /// Toffoli cost of one `SelectSwap2D` *and its uncompute*, for a given swap width.
    internal function SelectSwapCost2D(
        lambda : Int,
        numOuterData : Int,
        numInnerData : Int,
        numBits : Int,
        outerAddressAlwaysValid : Bool,
    ) : Int {
        let outerAddressBits = Ceiling(Lg(IntAsDouble(numOuterData)));
        let innerAddressBits = Ceiling(Lg(IntAsDouble(numInnerData)));

        let outerBlocks = if outerAddressAlwaysValid { numOuterData } else { 2^outerAddressBits };
        let numEntries = outerBlocks * 2^(innerAddressBits - lambda);
        let selectCost = numEntries - 2;

        let eraseCost = MeasurementUnlookupCost(outerAddressBits + innerAddressBits);
        let swapCost = (2^lambda - 1) * numBits;
        let numErasures = if lambda == 0 { 1 } else { 2 };

        return selectCost + swapCost + numErasures * eraseCost;
    }

    /// Toffoli cost of the *forward* clean select-swap load only.
    ///
    /// `SelectSwapCost1D` prices a compute/uncompute pair, which is the right model when the
    /// swap network erases itself. A streamed rotation batch is erased by measurement instead,
    /// so only the forward pass is paid for and the optimal width is wider than that model
    /// would choose.
    internal function SelectSwapForwardCost(numSwapBits : Int, numData : Int, numBits : Int) : Int {
        if numSwapBits <= 0 {
            return numData - 1;
        }
        let addressBits = Ceiling(Lg(IntAsDouble(numData)));
        2^(addressBits - numSwapBits) - 2 + (2^numSwapBits - 1) * numBits
    }

    /// Best clean swap width when only the forward load is paid for, and the ancilla it costs.
    ///
    /// Returns 0 when no network beats the plain lookup, in which case the caller should stay on
    /// `Select` rather than allocate scratch for nothing.
    function ComputeOptimalSwapBits(numData : Int, numBits : Int) : Int {
        let addressBits = Ceiling(Lg(IntAsDouble(numData)));
        mutable bestBits = 0;
        mutable best = SelectSwapForwardCost(0, numData, numBits);
        for k in 1..addressBits {
            let cost = SelectSwapForwardCost(k, numData, numBits);
            if cost < best {
                set best = cost;
                set bestBits = k;
            }
        }
        bestBits
    }

    /// Clean scratch qubits a `SelectSwapAliased` load allocates at a given swap width.
    function SelectSwapScratchQubits(numSwapBits : Int, numBits : Int) : Int {
        if numSwapBits <= 0 { 0 } else { numBits * (2^numSwapBits - 1) }
    }

    operation SelectSwap(numSwapBits : Int, data : Bool[][], address : Qubit[], output : Qubit[]) : Unit is Adj + Ctl {
        let nRequired = DimensionsForSelect(data, address);
        let addressFitted = address[...nRequired - 1];

        let swapBits = numSwapBits == -1 ? ComputeOptimalLambda1D(Length(data), Length(data[0])) | numSwapBits;

        Fact(swapBits <= nRequired, "Too many bits for SWAP network");

        let padded = PadToAddressSpace(data, nRequired);
        if swapBits == 0 {
            Select(padded, addressFitted, output);
        } else {
            WithSelectSwap(swapBits, padded, address, intermediate => ApplyToEachCA(CNOT, Zipped(intermediate, output)));
        }
    }

    /// Clean select-swap whose surplus addresses alias the way a bare `Select` reads them.
    ///
    /// `SelectSwap` zero-pads, which is right when its own adjoint erases the load. Here the
    /// erasure is a shared measurement-based unlookup with a phase fixup written against
    /// `Select`'s routing, so the forward load has to agree with that routing instead.
    operation SelectSwapAliased(
        numSwapBits : Int,
        data : Bool[][],
        address : Qubit[],
        output : Qubit[],
    ) : Unit is Adj + Ctl {
        let nRequired = DimensionsForSelect(data, address);
        SelectSwap(numSwapBits, AliasToAddressSpace(data, nRequired), address, output);
    }

    //  2D SELECT-SWAP (single select-swap over the combined outer×inner address)

    /// Loads the single `m`-bit word `data[outer][inner]` into an `m`-bit `target`.
    ///
    /// At `numSwapBits > 0` the swap network needs an `m * 2^numSwapBits` block. With `dirty`
    /// empty that block is allocated and erased by measurement. Otherwise it is borrowed from
    /// `dirty`, which may be entangled with anything and comes back exactly as it arrived, at
    /// the price of running `Select` twice and the butterfly four times. Either way the adjoint
    /// erases `target` with a phase fixup over the combined `(outer, inner)` address, so it is
    /// indifferent to how the forward pass was routed.
    operation SelectSwap2D(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
        outerAddress : Qubit[],
        innerAddress : Qubit[],
        dirty : Qubit[],
        target : Qubit[],
    ) : Unit is Adj {
        body (...) {
            Fact(not IsEmpty(data), "data cannot be empty");
            let m = Length(data[0][0]);
            Fact(
                Length(target) == m,
                $"target holds one {m}-bit word, got {Length(target)} qubits"
            );
            let (flatData, selectAddress, swapAddress) = SwappedLoadShape(
                data,
                outerAddress,
                innerAddress,
                numSwapBits,
                outerAddressAlwaysValid
            );
            let blockSize = m * (1 <<< numSwapBits);
            if numSwapBits == 0 {
                Select(flatData, selectAddress, target);
            } elif IsEmpty(dirty) {
                use swapTarget = Qubit[blockSize];
                Select(flatData, selectAddress, swapTarget);
                SwapDataOutputs(swapAddress, Chunks(m, swapTarget));
                ApplyToEachCA(CNOT, Zipped(swapTarget[0..m - 1], target));
                EraseSwappedLoad(
                    data,
                    outerAddress,
                    innerAddress,
                    numSwapBits,
                    outerAddressAlwaysValid,
                    swapTarget
                );
            } else {
                Fact(
                    Length(dirty) >= blockSize,
                    $"dirty register needs {blockSize} qubits, got {Length(dirty)}"
                );
                let borrowed = dirty[...blockSize - 1];
                let chunks = Chunks(m, borrowed);
                // Twice, because XOR is an involution: the second `Select` restores the lender
                // and the two copies of its addressed chunk cancel in `target`. Re-running
                // `Select` forward, not its measurement-based adjoint, keeps the lender intact.
                for _ in 1..2 {
                    within {
                        SwapDataOutputs(swapAddress, chunks);
                    } apply {
                        ApplyToEachCA(CNOT, Zipped(chunks[0], target));
                    }
                    Select(flatData, selectAddress, borrowed);
                }
            }
        }
        adjoint (...) {
            EraseSwappedLoad(data, outerAddress, innerAddress, 0, outerAddressAlwaysValid, target);
        }
    }

    /// Repairs the phase one branch left on `address` after a shared load was erased by measurement.
    /// Row `2k + flag` carries the parity of `measured` against row `k`, routed through
    /// `UnaryIterationActionIndex` because `Select` aliases surplus addresses onto real rows.
    internal operation ApplyBranchPhaseFixup(
        measured : Bool[],
        data : Bool[][],
        activeOnLowBit : Bool,
        address : Qubit[],
    ) : Unit {
        let phases = MappedOverRange(
            index -> {
                if ((index % 2 == 1) == activeOnLowBit) {
                    let word = data[UnaryIterationActionIndex(Length(data), index / 2)];
                    mutable parity = false;
                    // `Zipped` stops at the shorter of the two, which is what lets a narrower
                    // word share a measurement taken over the full target.
                    for (measuredBit, wordBit) in Zipped(measured, word) {
                        set parity = parity != (measuredBit and wordBit);
                    }
                    parity
                } else {
                    false
                }
            },
            0..(1 <<< Length(address)) - 1
        );
        use marker = Qubit();
        X(marker);
        H(marker);
        // `Select` XORs into its target, so a lookup into |-> kicks the data back as a phase;
        // the adjoint keeps those semantics and picks up the O(sqrt) measurement-based form.
        Adjoint Select(Mapped(phase -> [phase], phases), address, [marker]);
        // The adjoint may leave the marker measured out or still in |->, so the release
        // condition is restored explicitly rather than by undoing the preparation.
        Reset(marker);
    }

    /// Erases a post-butterfly 2D load by measurement instead of running it backwards.
    internal operation EraseSwappedLoad(
        data : Bool[][][],
        outerAddress : Qubit[],
        innerAddress : Qubit[],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
        target : Qubit[],
    ) : Unit {
        Fact(not IsEmpty(data), "data cannot be empty");
        let (flatData, selectAddress, swapAddress) = SwappedLoadShape(
            data,
            outerAddress,
            innerAddress,
            numSwapBits,
            outerAddressAlwaysValid
        );
        // `Adjoint Select` is the measurement-based unlookup; `Unlookup` itself is not exported.
        Adjoint Select(
            SwapPermutedTable(flatData, Length(data[0][0]), numSwapBits, 1 <<< Length(selectAddress)),
            selectAddress + swapAddress,
            target
        );
    }

    /// Runs `action` on the data word addressed by `address`, then uncomputes the lookup.
    internal operation WithSelectSwap(numSwapBits : Int, data : Bool[][], address : Qubit[], action : (Qubit[] => Unit is Adj + Ctl)) : Unit is Adj + Ctl {
        let nRequired = DimensionsForSelect(data, address);
        let addressFitted = address[...nRequired - 1];

        Fact(numSwapBits <= nRequired, "Too many bits for SWAP network");
        Fact(not IsEmpty(data), "data cannot be empty");
        let m = Length(data[0]);

        if numSwapBits == 0 {
            use output = Qubit[m];
            within {
                Select(data, addressFitted, output);
            } apply {
                action(output);
            }
        } else {
            let numSelectBits = nRequired - numSwapBits;
            let addressParts = Partitioned([numSelectBits, numSwapBits], addressFitted);

            use dataRegister = Qubit[m * 2^numSwapBits];

            let dataArray = CreatePaddedData(data, nRequired, m, numSelectBits);
            let chunkedDataRegister = Chunks(m, dataRegister);

            within {
                Select(dataArray, addressParts[0], dataRegister);
                SwapDataOutputs(addressParts[1], chunkedDataRegister);
            } apply {
                action(chunkedDataRegister[0]);
            }
        }
    }

    /// 1D SelectSwap correctness: set address to |addr⟩, apply SelectSwap in within/apply,
    /// CNOT result to persistent copy register, then verify copy matches expected data.
    internal operation TestSelectSwap1DCorrectness(
        data : Bool[][],
        numSwapBits : Int
    ) : Bool {
        let nData = Length(data);
        let m = Length(data[0]);
        let nAddr = Ceiling(Lg(IntAsDouble(nData)));

        use address = Qubit[nAddr];
        use output = Qubit[m];
        use copy = Qubit[m];

        mutable allCorrect = true;

        for addr in 0..nData - 1 {
            ApplyXorInPlace(addr, address);

            within {
                SelectSwap(numSwapBits, data, address, output);
            } apply {
                ApplyToEachCA(CNOT, Zipped(output, copy));
            }

            ApplyXorInPlace(addr, address);

            let actual = Mapped(ResultAsBool, MResetEachZ(copy));
            if actual != data[addr] {
                Message($"FAIL: addr={addr}, actual={actual}, expected={data[addr]}");
                set allCorrect = false;
            }
        }

        allCorrect
    }

    /// Cross-checks the select-swap path against the plain-select path as *phase* oracles.
    internal operation TestSelectSwap1DPhaseAgreement(data : Bool[][], numSwapBits : Int) : Bool {
        let m = Length(data[0]);
        let nAddr = Ceiling(Lg(IntAsDouble(Length(data))));

        use address = Qubit[nAddr];
        ApplyToEachA(H, address);

        {
            use output = Qubit[m];
            within {
                SelectSwap(numSwapBits, data, address, output);
            } apply {
                Z(output[0]);
            }
        }
        {
            use output = Qubit[m];
            within {
                SelectSwap(0, data, address, output);
            } apply {
                Z(output[0]);
            }
        }

        Adjoint ApplyToEachA(H, address);

        All(r -> r == Zero, MResetEachZ(address))
    }

    /// Cross-checks `SelectSwapAliased` against a bare `Select` on every address, by value.
    ///
    /// The comparison is deliberately against `Select` and not `SelectSwap(0, ...)`: the zero-pad
    /// and the alias differ precisely at the surplus addresses, which is the disagreement this
    /// operation exists to catch.
    ///
    /// It compares loaded values rather than phases because `Select` erases a ragged table by
    /// measurement, so `within { Select(...) } apply { Z(...) }` is not a phase oracle there --
    /// it disagrees even with itself. The forward load is the only thing the streamed rotation
    /// path uses `Select` for, and the forward load is what this checks.
    internal operation TestSelectSwapAliasedMatchesSelect1D(data : Bool[][], numSwapBits : Int) : Bool {
        let m = Length(data[0]);
        let nAddr = Ceiling(Lg(IntAsDouble(Length(data))));

        mutable allCorrect = true;
        for addr in 0..2^nAddr - 1 {
            use address = Qubit[nAddr];
            use swapped = Qubit[m];
            use plain = Qubit[m];

            ApplyXorInPlace(addr, address);
            SelectSwapAliased(numSwapBits, data, address, swapped);
            Select(data, address, plain);
            ApplyXorInPlace(addr, address);

            let swappedWord = Mapped(ResultAsBool, MResetEachZ(swapped));
            let plainWord = Mapped(ResultAsBool, MResetEachZ(plain));
            if swappedWord != plainWord {
                Message($"FAIL: addr={addr}, select-swap={swappedWord}, select={plainWord}");
                set allCorrect = false;
            }
            if not All(r -> r == Zero, MResetEachZ(address)) {
                Message($"FAIL: addr={addr} left the address register disturbed");
                set allCorrect = false;
            }
        }

        allCorrect
    }

    /// `SelectSwap2D` loads the addressed word into a one-word target, at every split.
    internal operation TestSelectSwap2DCorrectness(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool
    ) : Bool {
        let nOuter = Length(data);
        let nInner = Length(data[0]);
        let m = Length(data[0][0]);
        let nOuterAddr = Ceiling(Lg(IntAsDouble(nOuter)));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(nInner)));

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        use target = Qubit[m];
        use copy = Qubit[m];

        mutable allCorrect = true;

        for i in 0..nOuter - 1 {
            for j in 0..nInner - 1 {
                ApplyXorInPlace(i, outerAddr);
                ApplyXorInPlace(j, innerAddr);

                // The `within` uncompute is the measurement erasure, so this also checks that
                // the erasure returns `target` to |0> and not merely that the load was right.
                within {
                    SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, [], target);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(target, copy));
                }

                ApplyXorInPlace(i, outerAddr);
                ApplyXorInPlace(j, innerAddr);

                let actual = Mapped(ResultAsBool, MResetEachZ(copy));
                if actual != data[i][j] {
                    Message($"FAIL: (i={i},j={j}), actual={actual}, expected={data[i][j]}");
                    set allCorrect = false;
                }
            }
        }

        allCorrect
    }

    /// Phase-oracle agreement between `SelectSwap2D` and unary iteration.
    internal operation TestSelectSwap2DPhaseAgreement(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool
    ) : Bool {
        let m = Length(data[0][0]);
        let nOuter = Length(data);
        let nOuterAddr = Ceiling(Lg(IntAsDouble(nOuter)));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(Length(data[0]))));

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        within {
            if outerAddressAlwaysValid {
                PrepareUniformSuperposition(nOuter, outerAddr);
            } else {
                ApplyToEachA(H, outerAddr);
            }
            ApplyToEachA(H, innerAddr);
        } apply {
            {
                use target = Qubit[m];
                within {
                    SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, [], target);
                } apply {
                    Z(target[0]);
                }
            }
            {
                use target = Qubit[m];
                within {
                    UnaryIteration(outerAddr, Length(data), (index) => {
                        Select(PadToAddressSpace(data[index], nInnerAddr), innerAddr, target);
                    });
                } apply {
                    Z(target[0]);
                }
            }
        }

        All(r -> r == Zero, MResetEachZ(outerAddr + innerAddr))
    }

    /// Traces `SelectSwap2D` in one or both directions for a costing regression.
    internal operation TestSelectSwap2DResourceProbe(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
        applyForward : Bool,
        applyAdjoint : Bool
    ) : Unit {
        let nOuterAddr = Ceiling(Lg(IntAsDouble(Length(data))));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(Length(data[0]))));

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        use target = Qubit[Length(data[0][0])];

        if applyForward {
            SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, [], target);
        }
        if applyAdjoint {
            Adjoint SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, [], target);
        }
    }

}
