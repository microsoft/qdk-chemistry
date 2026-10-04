// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SELECT-SWAP network for efficient QROM data loading (1D and 2D).
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
    import Std.Arrays.Enumerated;
    import Std.Arrays.Flattened;
    import Std.Arrays.Mapped;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Padded;
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
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;
    import QDKChemistry.Utils.UnaryIteration.UnaryIteration;
    import QDKChemistry.Utils.UnaryIteration.UnaryIterationActionIndex;

    /// Zero-pads a lookup table out to the full `2^nRequired` address space.
    internal function PadToAddressSpace(data : Bool[][], nRequired : Int) : Bool[][] {
        Padded(-2^nRequired, [false, size = Length(data[0])], data)
    }

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

    /// Expands a lookup table to the full `2^nRequired` address space the way `Select` reads it.
    ///
    /// `Select` never reads a surplus address as zero; `UnaryIteration` routes anything at or
    /// above `Length(data)` back onto a real row. A loader that zero-pads instead disagrees with
    /// `Select` -- and with any phase fixup written against `Select` -- on exactly those
    /// addresses. Building the table this way makes the agreement unconditional rather than a
    /// property of which addresses happen to carry amplitude.
    internal function AliasToAddressSpace(data : Bool[][], nRequired : Int) : Bool[][] {
        MappedOverRange(
            i -> data[UnaryIterationActionIndex(Length(data), i)],
            0..2^nRequired - 1
        )
    }

    /// Register slices and the flattened table shared by the swap path and its erasure.
    internal function SwappedLoadShape(
        data : Bool[][][],
        outerAddress : Qubit[],
        innerAddress : Qubit[],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
    ) : (Bool[][], Qubit[], Qubit[]) {
        let nRequired = DimensionsForSelect(data[0], innerAddress);
        Fact(numSwapBits <= nRequired, "Too many bits for SWAP network");
        let m = Length(data[0][0]);
        let k = nRequired - numSwapBits;
        let innerAddressParts = Partitioned([k, numSwapBits], innerAddress[...nRequired - 1]);
        let flatData = FlattenPaddedData(data, nRequired, m, k, outerAddressAlwaysValid);
        let selectAddress = innerAddressParts[0] + outerAddress[...AddressQubits(Length(data)) - 1];
        (flatData, selectAddress, innerAddressParts[1])
    }

    /// Where each chunk ends up after `SwapDataOutputs` runs for a given swap value.
    ///
    /// `result[position]` is the chunk index the butterfly leaves at `position`. Only
    /// `result[0] == swap` is the point of the network; the rest are a permutation that is
    /// *not* `position XOR swap` beyond one swap bit, so it is replayed here rather than
    /// assumed.
    internal function SwapNetworkPermutation(numSwapBits : Int, swap : Int) : Int[] {
        let numChunks = 1 <<< numSwapBits;
        mutable permutation = MappedOverRange(chunk -> chunk, 0..numChunks - 1);
        for bit in 0..numSwapBits - 1 {
            if (swap >>> bit) &&& 1 == 1 {
                let innerStep = 1 <<< bit;
                let outerStep = 1 <<< (bit + 1);
                for pair in 0..numChunks / outerStep - 1 {
                    let low = pair * outerStep;
                    let high = low + innerStep;
                    let held = permutation[low];
                    set permutation w/= low <- permutation[high];
                    set permutation w/= high <- held;
                }
            }
        }
        permutation
    }

    /// The post-butterfly target contents indexed by `select + swap * numSelectStates`.
    internal function SwapPermutedTable(
        flatData : Bool[][],
        m : Int,
        numSwapBits : Int,
        numSelectStates : Int,
    ) : Bool[][] {
        let numChunks = 1 <<< numSwapBits;
        let unreachable = [false, size = m * numChunks];
        mutable table : Bool[][] = [];
        for swap in 0..numChunks - 1 {
            let permutation = SwapNetworkPermutation(numSwapBits, swap);
            for index in 0..numSelectStates - 1 {
                if index < Length(flatData) {
                    let chunks = Chunks(m, flatData[index]);
                    mutable permuted : Bool[] = [];
                    for position in 0..numChunks - 1 {
                        set permuted += chunks[permutation[position]];
                    }
                    set table += [permuted];
                } else {
                    set table += [unreachable];
                }
            }
        }
        table
    }

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

    /// Toffoli cost of erasing an `addressBits`-wide load by measurement and phase fixup.
    ///
    /// The `O(sqrt(N))` unlookup: measure the loaded word out in the X basis and repair the
    /// phase it leaves on the address with a lookup over a table of parities.
    internal function MeasurementUnlookupCost(addressBits : Int) : Int {
        2^((addressBits + 1) / 2) + 2^(addressBits / 2) - (addressBits + 2)
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

    /// Toffoli cost of one `SelectSwap2DDirty` and its uncompute, for a given swap width.
    ///
    /// Two differences from `SelectSwapCost2D`, both from borrowing rather than allocating.
    /// The forward pass runs `Select` twice and the butterfly four times instead of once each,
    /// which is the price of restoring the lender by XOR involution rather than by measuring
    /// it -- `2*ceil(d/K) + 4*b*(K-1)` in Berry et al. (arXiv:1902.02134, Appendix A,
    /// Theorem 1). Against that, only the target is ever erased: there is no clean scratch to
    /// unlook, so a wide load pays one measurement erasure where the clean path pays two.
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
            // Width 0 is a single plain `Select`, not the `K = 1` limit of the swap formula:
            // that limit charges two passes for a load that only makes one.
            outerBlocks * 2^innerAddressBits - 2 + eraseCost
        } else {
            let numEntries = outerBlocks * 2^(innerAddressBits - lambda);
            let selectCost = 2 * (numEntries - 1);
            let swapCost = 4 * numBits * (2^lambda - 1);
            selectCost + swapCost + eraseCost
        }
    }

    /// Best dirty swap width for the 2D lookup, given how many qubits the caller can lend.
    ///
    /// Returns 0 when no wider network fits in `availableDirty` or when none beats the plain
    /// load, in which case the caller should stay on `Select` and borrow nothing.
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

    internal function DimensionsForSelect(data : Bool[][], address : Qubit[]) : Int {
        let N = Length(data);
        Fact(N > 0, "data cannot be empty");

        let n = Ceiling(Lg(IntAsDouble(N)));
        Fact(Length(address) >= n, $"address register is too small, requires at least {n} qubits");

        return n;
    }

    internal function CreatePaddedData(data : Bool[][], nRequired : Int, m : Int, k : Int) : Bool[][] {
        let dataPadded = Padded(-2^nRequired, [false, size = m], data);

        MappedOverRange(i -> Flattened(dataPadded[i..2^k..2^nRequired - 1]), 0..2^k - 1)
    }

    /// Concatenates the per-outer-index padded lookup tables into one row-major table, so a
    /// combined `(innerSelect, outer)` address indexes it directly.
    internal function FlattenPaddedData(
        data : Bool[][][],
        nRequired : Int,
        m : Int,
        k : Int,
        trimToValidOuter : Bool,
    ) : Bool[][] {
        let numOuterStates = if trimToValidOuter { Length(data) } else { 1 <<< AddressQubits(Length(data)) };
        Flattened(
            MappedOverRange(
                state -> CreatePaddedData(data[UnaryIterationActionIndex(Length(data), state)], nRequired, m, k),
                0..numOuterStates - 1
            )
        )
    }

    //  1D SELECT-SWAP
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

    /// `SelectSwap2D` that borrows the swap block instead of allocating it.
    ///
    /// Same contract as `SelectSwap2D` — XORs `data[outer][inner]` into `target`, and its
    /// adjoint is the same measurement-based erasure — but the `m * 2^numSwapBits` swap block
    /// is lent by the caller, so the load costs no width at all. `dirty` may be entangled with
    /// anything and is handed back exactly as it arrived; nothing there is measured or reset.
    ///
    /// The forward pass goes through `SwappedLoadShape`, so the flattened table, the select
    /// address and the butterfly address are the ones the clean path builds. That is what keeps
    /// the shared erasure valid: the adjoint reads the combined `(outer, inner)` address and is
    /// indifferent to how the forward pass was routed, but only if both agree on which row the
    /// address names.
    ///
    /// Cancellation, with the borrowed block as chunks `psi_0..psi_{K-1}` and `s` the swap
    /// address, exactly as in `SelectSwapDirty`:
    ///
    ///   1. butterfly, `target ^= psi_s`, unbutterfly
    ///   2. `Select` — chunk `p` becomes `psi_p ^ data[select + p*2^k]`
    ///   3. butterfly, `target ^= psi_s ^ data[...]`, unbutterfly — the two `psi_s` cancel
    ///   4. `Select` again — XOR is an involution, so the lender is restored
    ///
    /// Step 4 re-runs `Select` forward rather than taking its adjoint: the library adjoint is a
    /// measurement-based unlookup, which would destroy the lender's state.
    operation SelectSwap2DDirty(
        data : Bool[][][],
        outerAddress : Qubit[],
        innerAddress : Qubit[],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
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
            if numSwapBits == 0 {
                Select(flatData, selectAddress, target);
            } else {
                let needed = DirtyQROAMBorrowedQubits(numSwapBits, m);
                Fact(
                    Length(dirty) >= needed,
                    $"dirty register needs {needed} qubits, got {Length(dirty)}"
                );
                let borrowed = dirty[...needed - 1];
                let chunks = Chunks(m, borrowed);

                within {
                    SwapDataOutputs(swapAddress, chunks);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(chunks[0], target));
                }

                Select(flatData, selectAddress, borrowed);

                within {
                    SwapDataOutputs(swapAddress, chunks);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(chunks[0], target));
                }

                Select(flatData, selectAddress, borrowed);
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

    internal operation SwapDataOutputs(address : Qubit[], outputs : Qubit[][]) : Unit is Adj {
        let l = Length(address);
        for (i, control) in Enumerated(address) {
            let innerStepSize = 2^i;
            let outerStepSize = 2^(i + 1);
            let numSwaps = 2^l / 2^(i + 1);
            for j in 0..numSwaps - 1 {
                let targets1 = outputs[j * outerStepSize];
                let targets2 = outputs[j * outerStepSize + innerStepSize];
                ApplyToEachA(ts => Controlled SWAP([control], ts), Zipped(targets1, targets2));
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

    /// `SelectSwap2DDirty` loads exactly what `SelectSwap2D` loads, and returns the lender.
    ///
    /// The two loads are run back to back into the same `copy` register, so `copy` cancels to
    /// zero precisely when they agree. Comparing against the clean path rather than against
    /// `data[i][j]` is what lets the sweep cover the *whole* address space: surplus outer
    /// addresses are aliased onto real rows by `Select`, and the point at issue is that the
    /// borrowed path aliases them the same way, not what the alias happens to be.
    ///
    /// The borrowed register is seeded into a non-trivial product state, so a construction that
    /// silently assumed `|0>` scratch shows up as either a wrong word or a disturbed lender.
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

                // Each `within` uncompute is the measurement erasure, which leaves a phase on
                // the address but returns `target` to |0>. The address is a basis state here,
                // so that phase is global and cannot affect the comparison.
                within {
                    SelectSwap2DDirty(
                        data,
                        outerAddr,
                        innerAddr,
                        numSwapBits,
                        outerAddressAlwaysValid,
                        dirty,
                        target
                    );
                } apply {
                    ApplyToEachCA(CNOT, Zipped(target, copy));
                }
                within {
                    SelectSwap2D(data, outerAddr, innerAddr, 0, outerAddressAlwaysValid, target);
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

    /// Traces `SelectSwap2DDirty` for a costing regression.
    ///
    /// The lender is allocated here because a standalone probe has nothing to borrow from, so
    /// this measures Toffolis rather than width; the width saving only exists where the lender
    /// is a register the caller already owns.
    internal operation TestSelectSwap2DDirtyResourceProbe(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
        applyForward : Bool,
        applyAdjoint : Bool
    ) : Unit {
        let m = Length(data[0][0]);
        let nOuterAddr = Ceiling(Lg(IntAsDouble(Length(data))));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(Length(data[0]))));

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        use target = Qubit[m];
        use dirty = Qubit[MaxI(1, DirtyQROAMBorrowedQubits(numSwapBits, m))];

        if applyForward {
            SelectSwap2DDirty(
                data,
                outerAddr,
                innerAddr,
                numSwapBits,
                outerAddressAlwaysValid,
                dirty,
                target
            );
        }
        if applyAdjoint {
            Adjoint SelectSwap2DDirty(
                data,
                outerAddr,
                innerAddr,
                numSwapBits,
                outerAddressAlwaysValid,
                dirty,
                target
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Dirty-qubit QROAM
    // ═══════════════════════════════════════════════════════════════════════════

    /// Qubits a `SelectSwapDirty` load has to borrow for a given swap width.
    ///
    /// The butterfly needs the full `2^numSwapBits` chunks resident, so the borrowed block is
    /// `numBits * 2^numSwapBits` wide. Unlike `SelectSwap` these are never allocated: the caller
    /// lends registers that are already live, and gets them back unchanged.
    function DirtyQROAMBorrowedQubits(numSwapBits : Int, numBits : Int) : Int {
        numBits * (1 <<< numSwapBits)
    }

    /// Toffoli cost of the forward `SelectSwapDirty` load (its measurement-based adjoint is
    /// charged separately, as for `SelectSwap2D`).
    ///
    /// At width 0 the load is a single plain `Select`, whose unary iteration over `numData`
    /// rows costs `numData - 1`. Widths above that are two `Select` passes over `2^(n-k)`
    /// aliased rows plus four butterflies of `numBits * (2^k - 1)` controlled swaps, which is
    /// the structure of `2*ceil(d/K) + 4*b*(K-1)` in :cite:`Berry2019` (Appendix A, Theorem 1).
    ///
    /// The butterfly term is theirs exactly. The `Select` term is deliberately not: it is
    /// `2*(2^(n-k) - 1)` rather than `2*ceil(d/K)`, which differs in two ways that pull in
    /// opposite directions. The `-1` per pass is the exact unary-iteration cost rather than
    /// their bound, so a power-of-two table costs 2 Toffolis less here than their formula
    /// quotes. Against that, the table is padded to `2^ceil(lg d)` rows, so a table far from a
    /// power of two is charged for the padding: at the Fe2S2 inner shape (`d = 90`, `K = 4`)
    /// this is 62 against their 46. The padded form is what the implementation below actually
    /// addresses, so costing the unpadded table would under-report a circuit nobody builds --
    /// but it does mean this model is conservative about dirty loads at awkward table sizes,
    /// which is the direction that makes borrowing look worse than it is.
    ///
    /// The width-0 case has to be the plain cost and not the `K = 1` limit of the swap formula:
    /// that limit charges two passes for a load that only makes one, and the doubled baseline
    /// would make a swap network look profitable at table shapes where it is not.
    internal function DirtyQROAMCost(numSwapBits : Int, numData : Int, numBits : Int) : Int {
        if numSwapBits == 0 {
            numData - 1
        } else {
            let addressBits = Ceiling(Lg(IntAsDouble(numData)));
            let selectCost = 2 * (2^(addressBits - numSwapBits) - 1);
            let swapCost = 4 * numBits * (2^numSwapBits - 1);
            selectCost + swapCost
        }
    }

    /// Best swap width for a dirty load, given how many qubits the caller can actually lend.
    ///
    /// Returns 0 (plain `Select`, nothing borrowed) when no wider network fits or when none
    /// beats the plain load. Borrowing only pays off once the table is tall relative to the
    /// word it returns -- roughly `numData > 32 * numBits` -- so a narrow table correctly
    /// declines the swap network rather than paying for one.
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

    /// Chunked table for a dirty load, with surplus addresses aliased the way `Select` does.
    ///
    /// `CreatePaddedData` fills the surplus rows with zeros, which is what the clean swap path
    /// wants. A dirty load is a drop-in for a bare `Select`, and `Select` instead *aliases* the
    /// addresses at or above `Length(data)` onto real rows -- the same routing
    /// `ApplyBranchPhaseFixup` compensates for. Zero padding here would leave the forward load
    /// and that fixup disagreeing on exactly those addresses.
    ///
    /// Row `i` holds chunk `p` = `data[i + p * 2^k]`, matching the little-endian split of the
    /// address into `k` select bits and `numSwapBits` swap bits.
    internal function CreateAliasedData(data : Bool[][], nRequired : Int, k : Int) : Bool[][] {
        let aliased = AliasToAddressSpace(data, nRequired);
        MappedOverRange(i -> Flattened(aliased[i..2^k..2^nRequired - 1]), 0..2^k - 1)
    }

    /// QROAM that borrows already-live qubits instead of allocating clean scratch.
    ///
    /// XORs `data[address]` into `output` and returns `dirty` to whatever state it was in.
    /// `dirty` may be entangled with anything; nothing is ever measured or reset there.
    ///
    /// How the unknown contents cancel: write the borrowed block as chunks `psi_0..psi_{K-1}`
    /// and let `s` be the value of the swap-address bits. The butterfly leaves `psi_s` at
    /// position 0, so copying position 0 out *before* and *after* the data XOR contributes
    /// `psi_s` twice, which cancels, while the data word contributes once:
    ///
    ///   1. butterfly, `output ^= psi_s`, unbutterfly
    ///   2. `Select`  — chunk `p` becomes `psi_p ^ data[select + p*2^k]`
    ///   3. butterfly, `output ^= psi_s ^ data[select + s*2^k]`, unbutterfly
    ///   4. `Select` again — XOR is an involution, so the borrowed block is restored
    ///
    /// Step 4 re-runs `Select` forward rather than taking its adjoint: the library adjoint is a
    /// measurement-based unlookup, which would destroy the lender's state.
    ///
    /// Only the two `Select` passes carry the control. On the inactive branch the butterflies
    /// still run and the two `psi_s` copies cancel on their own, so `output` is untouched —
    /// controlling them as well would be redundant.
    ///
    /// The net action is `output ^= data[address]`, which is its own inverse, hence
    /// `adjoint self`. Callers that are *releasing* `output` should prefer the measurement-based
    /// erasure (`Adjoint Select` over the full address, as `EraseSwappedLoad` does) instead of
    /// the adjoint here: the forward pass already handed the borrowed qubits back, so the
    /// cheap unlookup applies verbatim and costs `O(sqrt(2^n))` rather than a second load.
    ///
    /// Reference: Berry, Gidney, Motta, McClean, Babbush (arXiv:1902.02134), Appendix A,
    /// Theorem 1; Low, Kliuchnikov, Schaeffer (arXiv:1812.00954).
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
                let addressParts = Partitioned([numSelectBits, numSwapBits], address[...nRequired - 1]);
                let dataArray = CreateAliasedData(data, nRequired, numSelectBits);

                within {
                    SwapDataOutputs(addressParts[1], chunks);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(chunks[0], output));
                }

                Controlled Select(controls, (dataArray, addressParts[0], borrowed));

                within {
                    SwapDataOutputs(addressParts[1], chunks);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(chunks[0], output));
                }

                Controlled Select(controls, (dataArray, addressParts[0], borrowed));
            }
        }
        adjoint self;
    }

    /// `SelectSwapDirty` loads the addressed word and hands the borrowed qubits back untouched.
    ///
    /// The borrowed register is seeded into a non-trivial product state so a construction that
    /// silently assumed `|0>` scratch would show up as either a wrong word or a disturbed lender.
    ///
    /// The sweep covers the whole address space, not just the `Length(data)` real rows: the
    /// expected word is taken from a plain `Select` rather than from `data[addr]`, so the
    /// surplus addresses assert that a dirty load aliases them exactly the way `Select` does.
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

            // `Select` is its own reference here: running it forward twice XORs to zero, so the
            // expected word is read off without invoking the measurement-based unlookup.
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

    /// Cross-checks the dirty load against a plain-`Select` load as a *phase* oracle, over a
    /// superposed address and a superposed borrowed register.
    ///
    /// This is the test that would catch a residual dirty-state entanglement that the
    /// computational-basis check in `TestSelectSwapDirtyCorrectness` cannot see: if the
    /// borrowed qubits stayed correlated with the address, the interference below would not
    /// return the address register to `|0>`.
    ///
    /// Both arms go through `SelectSwapDirty`, the reference one at `numSwapBits = 0` where it
    /// is a bare `Select`. Conjugating by `Select` directly would instead pull in its
    /// measurement-based unlookup, which for a table whose row count is not a power of two does
    /// not return a superposed address to `|0>` on its own -- an unrelated property that would
    /// make this test read as a failure here.
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
}
