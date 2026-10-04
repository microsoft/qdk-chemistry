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
///   SelectSwap2D — loads data[outer][inner] with one select-swap over the combined address
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

    //  LOOKUP METHOD TAGS
    //
    //  The three ways any table in this library can be loaded. Shared by every loader --
    //  the streamed rotation batches and the inner alias-sampling tables both select from
    //  this one set -- so a caller names a routing strategy once and every lookup honours it.
    //
    //  Each tag is a strategy, not a guarantee. Every method falls back to `LookupSelect` when
    //  its own cost model says no network beats the plain load at that shape, so naming a
    //  method can never cost Toffolis relative to `LookupSelect`; it only grants permission to
    //  spend qubits if doing so pays.
    //
    //  Three, where the reference library (microsoft/qdk `library/table_lookup`) has four: it
    //  also loads via power products -- `LookupViaPP` and `LookupViaSplitPP`, after Gidney
    //  (arXiv:2505.15917, section A.4) -- which we deliberately do not implement. Power products
    //  put the address into a product basis, Mobius-transform the data, and emit one CCNOT per
    //  set bit of the transformed mask *per output bit*, so the Toffoli cost scales with the word
    //  width where a `Select` address iteration does not. That is least attractive at exactly our
    //  shapes, which are narrow and deep -- order a hundred rows of a few tens of bits. The
    //  argument is structural rather than benchmarked, so measure before concluding the gap is
    //  real.

    /// Plain unary-iteration `Select`: no scratch, no borrowing, `numData - 1` Toffolis.
    ///
    /// The widest-register, cheapest-space option, and the floor every other method falls back
    /// to. Choose it when neither allocatable nor borrowable space exists.
    function LookupSelect() : Int { 0 }

    /// Clean select-swap (QROAM): allocates scratch to cut Toffolis.
    ///
    /// Trades `numBits * (2^k - 1)` fresh qubits for a shallower address iteration. The scratch
    /// adds to peak width, so this pays when Toffolis bind and width does not.
    function LookupSelectSwap() : Int { 1 }

    /// Select-swap that borrows live caller qubits instead of allocating scratch.
    ///
    /// Costs no width at all, but runs `Select` twice and the butterfly four times, so it pays
    /// roughly two to three times the Toffolis of the clean network at equal width. It only
    /// undercuts a plain `Select` on tables large relative to the word -- roughly
    /// `numData > 32 * numBits` -- and is additionally capped by how much the caller lent.
    /// At shapes where neither holds it declines and falls back to `LookupSelect`.
    function LookupDirtySelectSwap() : Int { 2 }

    /// Largest relative Toffoli premium worth paying for a narrower swap network.
    ///
    /// Each extra swap bit doubles the scratch block, and for a caller not already wider
    /// elsewhere that block sets the peak width of the whole algorithm. Scoring widths by
    /// Toffoli count alone never sees that, so it keeps widening while the gains flatten.
    ///
    /// Measured at the inner-PREPARE shape this rule's test exercises (`d = 90`, `m = 16`,
    /// `b = 21`): `k = 3` is the Toffoli minimum at 491, and this rule instead takes `k = 2`
    /// at 587 -- 19.6% more Toffolis for 84 of 283 scratch qubits (30%) less width.
    ///
    /// That decision is narrow: 587 clears the `491 * 1.2 = 589.2` threshold by 2.2 Toffolis,
    /// so a tolerance under roughly 0.195 would take `k = 3` and widen the algorithm instead.
    /// The margin lives in this constant rather than in the shape, so treat it as load-bearing
    /// rather than as a round number.
    internal function MaxToffoliPremiumForNarrowing() : Double {
        0.2
    }

    /// Narrowest swap width within `MaxToffoliPremiumForNarrowing()` of the cheapest.
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

        let threshold = IntAsDouble(best) * (1.0 + MaxToffoliPremiumForNarrowing());
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

    //  2D SELECT-SWAP (single select-swap over the combined outer×inner address)

    /// Loads the single `m`-bit word `data[outer][inner]` into an `m`-bit `target`.
    ///
    /// At `numSwapBits > 0`, the lookup writes `m * 2^numSwapBits` scratch bits, moves the
    /// addressed word to the first chunk, copies it to `target`, and erases the scratch by
    /// measurement. Its custom adjoint erases `target` with a phase fixup over the combined
    /// `(outer, inner)` address, independent of the swap width used by the forward pass.
    operation SelectSwap2D(
        data : Bool[][][],
        outerAddress : Qubit[],
        innerAddress : Qubit[],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
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
            if numSwapBits == 0 {
                Select(flatData, selectAddress, target);
            } else {
                use swapTarget = Qubit[m * (1 <<< numSwapBits)];
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
                    SelectSwap2D(data, outerAddr, innerAddr, numSwapBits, outerAddressAlwaysValid, target);
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
                    SelectSwap2D(data, outerAddr, innerAddr, numSwapBits, outerAddressAlwaysValid, target);
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
            SelectSwap2D(data, outerAddr, innerAddr, numSwapBits, outerAddressAlwaysValid, target);
        }
        if applyAdjoint {
            Adjoint SelectSwap2D(data, outerAddr, innerAddr, numSwapBits, outerAddressAlwaysValid, target);
        }
    }
}
