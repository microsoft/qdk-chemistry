// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SELECT-SWAP network for efficient QROM data loading (1D and 2D).
///
/// Implements the SELECT-SWAP technique that trades ancilla qubits for
/// reduced T-gate count when loading classical data into quantum registers.
/// Uses measurement-based uncomputation for the adjoint.
///
/// 1D operations:
///   SelectSwap — loads data[address] into output.
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

    // ═══════════════════════════════════════════════════════════════════════════
    // Classical helpers
    // ═══════════════════════════════════════════════════════════════════════════

    /// Zero-pads a lookup table out to the full `2^nRequired` address space.
    internal function PadToAddressSpace(data : Bool[][], nRequired : Int) : Bool[][] {
        Padded(-2^nRequired, [false, size = Length(data[0])], data)
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
    ///
    /// `numSelectStates` must be the full `2^Length(selectAddress)`, not `Length(flatData)`:
    /// the address register puts the swap bits above the select bits, so a shorter stride
    /// would misalign every row with a nonzero swap value. Rows past the end of `flatData` are
    /// unreachable and carry no phase.
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

    function ComputeOptimalLambda2D(
        numOuterData : Int,
        numInnerData : Int,
        numBits : Int,
        outerAddressAlwaysValid : Bool,
    ) : Int {
        mutable best = 2^32;
        mutable bestLambda = 0;

        let addressBits = Ceiling(Lg(IntAsDouble(numInnerData)));
        for lambda in 0..addressBits - 1 {
            let cost = SelectSwapCost2D(lambda, numOuterData, numInnerData, numBits, outerAddressAlwaysValid);
            if cost < best {
                set bestLambda = lambda;
                set best = cost;
            }
        }

        return bestLambda;
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
    ///
    /// This mirrors the implementation term by term: a `Select` over the combined
    /// `(innerSelect, outer)` address, the swap network once, and a measurement erasure. The
    /// swap network is charged once rather than twice because both erasures measure their
    /// target out instead of running the butterfly backwards, and an erasure addresses the
    /// full `(outer, inner)` address whatever `lambda` is.
    ///
    /// The round trip pays that erasure twice at `lambda > 0` -- once inside the load to clear
    /// the wide register the swap network wrote, once on the way back to clear the word -- but
    /// only once at `lambda == 0`, where there is no wide register and the load is a bare
    /// `Select`. That discount is the whole reason `lambda` selection cannot just drop the
    /// term as a shared constant.
    ///
    /// `outerAddressAlwaysValid` must match what the caller passes to `SelectSwap2D`: it
    /// decides whether the lookup covers `numOuterData` outer blocks or the full
    /// `2^ceil(lg numOuterData)`, which is a factor of up to two on the `Select` term and can
    /// move the optimum by a whole swap bit.
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

        let eraseBits = outerAddressBits + innerAddressBits;
        let eraseCost = 2^((eraseBits + 1) / 2) + 2^(eraseBits / 2) - (eraseBits + 2);
        let swapCost = (2^lambda - 1) * numBits;
        let numErasures = if lambda == 0 { 1 } else { 2 };

        return selectCost + swapCost + numErasures * eraseCost;
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
    ///
    /// By default the table covers all `2^AddressQubits(Length(data))` outer address states,
    /// not just the `Length(data)` valid ones, with each unused state routed to the same entry
    /// `UnaryIteration` would have selected. That keeps the lookup identical to the
    /// `numSwapBits == 0` path on every address state and makes the combined table a power of
    /// two, so `Select` never applies aliasing of its own.
    ///
    /// `trimToValidOuter` drops the unused states, shrinking the `Select` by the ratio
    /// `2^AddressQubits(Length(data)) / Length(data)`. It is only sound when the caller
    /// guarantees the outer register never holds an out-of-range value, because the two paths
    /// then disagree there -- `Select` aliases the trimmed states onto real entries.
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

    // ═══════════════════════════════════════════════════════════════════════════
    // Quantum operations
    // ═══════════════════════════════════════════════════════════════════════════

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

    //  2D SELECT-SWAP (single select-swap over the combined outer×inner address)
    /// Loads the single `m`-bit word `data[outer][inner]` into an `m`-bit `target`.
    ///
    /// At `numSwapBits > 0`, the lookup writes `m * 2^numSwapBits` scratch bits, moves the
    /// addressed word to the first chunk, copies it to `target`, and erases the scratch by
    /// measurement. Its custom adjoint erases `target` with a phase fixup over the combined
    /// `(outer, inner)` address, independent of the swap width used by the forward pass.
    ///
    /// The round trip is `select + swap + 2 erase` instead of `2(select + swap + erase)`. At
    /// the SOSSA inner PREPARE (90 conditions x 16 slots, 21-bit words) that is 491 Toffolis
    /// against 816 at the swap width both pick, and against 2,876 for the unary-iteration
    /// load this replaced.
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
            // Erase against the flat (outer, inner) table: `target` holds `data[outer][inner]`
            // however the forward pass got it there, so the fixup does not depend on the swap
            // width the load used.
            EraseSwappedLoad(data, outerAddress, innerAddress, 0, outerAddressAlwaysValid, target);
        }
    }

    /// Erases a post-butterfly 2D load by measurement instead of running it backwards.
    ///
    /// Reversing the load costs `(2^lambda - 1) * m` Toffolis for the swap network plus the
    /// unlookup, and the swap network is the larger of the two. The whole target is instead a
    /// known function of the address -- the butterfly applies a known permutation to the
    /// selected chunks -- so it can be measured out in the X basis and the kickback repaired
    /// with one phase lookup. That lookup is over the combined
    /// select-and-swap address, which is the full `(outer, inner)` address and therefore the
    /// same width whatever `lambda` is, so the swap network is paid once rather than twice.
    ///
    /// This is the erasure of arXiv:2502.15882v1 Appendix B step 2: "(2^k-1)b qubits are
    /// erased (via an X measurement and a later phase fixup)".
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
    ///
    /// The adjoint is compiler-generated. Uncomputing a select-swap lookup by hand is
    /// error-prone: it must undo the swap network as well as the lookup, and `Adjoint
    /// Select` measures the target and repairs the phase kickback that leaves on the
    /// address register. A wrong uncompute still loads the right values and corrupts only
    /// the address-register phase, which for a state-preparation caller is the state.
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

    // ═══════════════════════════════════════════════════════════════════════════
    // Test wrappers
    // ═══════════════════════════════════════════════════════════════════════════

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
    ///
    /// A lookup with a wrong uncompute still loads the right bits, so the correctness
    /// wrappers above cannot see it: the damage lands on the phase of the address register.
    /// Both `within { lookup } apply { Z }` blocks are diagonal ±1 oracles and therefore
    /// self-inverse, so composing the swap path with the no-swap path is the identity
    /// exactly when the two agree, and the address register returns to |0...0>.
    ///
    /// The verdict is a single measurement, so it is a *probabilistic* detector: when the two
    /// paths disagree the address is left in a superposition that still collapses to |0...0>
    /// a large fraction of the time (~35-50% for the shapes tested here). A `false` is
    /// therefore conclusive but a `true` is not, and callers must repeat the trial -- see
    /// `_assert_phase_agreement` in `python/tests/test_utils_select_swap.py`. A deterministic
    /// `CheckAllZero` verdict is not available: it is rejected under the Adaptive_RIF profile
    /// these tests run on.
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
    ///
    /// `SelectSwap2D` is the one lookup here whose adjoint is not the reverse of its body:
    /// the body may run a swap network, while the erasure ignores it and fixes up against the
    /// flat `(outer, inner)` table. An erasure that got that wrong would still leave the right
    /// bits in `target` and pass every value test, showing up only as a phase on the address
    /// register -- which is the state, for the state-preparation caller.
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
    ///
    /// The load and its erasure are selectable separately because the erasure is written by
    /// hand rather than derived from the body, so the two costs move independently.
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
