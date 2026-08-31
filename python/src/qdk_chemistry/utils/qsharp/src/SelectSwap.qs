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
///   Select2DLoad — loads data[outer][inner] into target via unary iteration + SWAP
///   ComputeOptimalLambda2D — optimal SWAP bits for 2D case
///
/// References:
///   Low, Kliuchnikov, Schaeffer (arXiv:1812.00954)
namespace QDKChemistry.Utils.SelectSwap {

    import Std.Arrays.All;
    import Std.Arrays.Chunks;
    import Std.Arrays.Enumerated;
    import Std.Arrays.Flattened;
    import Std.Arrays.IndexRange;
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
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;
    import QDKChemistry.Utils.UnaryIteration.UnaryIteration;
    import QDKChemistry.Utils.UnaryIteration.UnaryIterationActionIndex;

    /// Zero-pads a lookup table out to the full `2^nRequired` address space.
    ///
    /// An unused address then deterministically loads the all-zero word instead of whatever
    /// entry `Select` would alias it onto. Without this the word loaded for an unused address
    /// depends on the select/swap split -- the swap path zero-pads through `CreatePaddedData`
    /// while a plain `Select` aliases -- and since `ComputeOptimalLambda*` derives that split
    /// from the table shape, the operation's action on those addresses would silently change
    /// with the data dimensions. Padding is a no-op when the table already fills its address
    /// space, which is the case for every production caller.
    internal function PadToAddressSpace(data : Bool[][], nRequired : Int) : Bool[][] {
        Padded(-2^nRequired, [false, size = Length(data[0])], data)
    }

    //  1D SELECT-SWAP
    operation SelectSwap(numSwapBits : Int, data : Bool[][], address : Qubit[], output : Qubit[]) : Unit is Adj + Ctl {
        let (n, nRequired) = DimensionsForSelect(data, address);
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

    /// Historical lookup model retained only for comparison with pre-cleanup estimates.
    internal operation LegacySelectSwapResourceEstimate(
        numSwapBits : Int,
        data : Bool[][],
        address : Qubit[],
        output : Qubit[],
    ) : Unit is Adj + Ctl {
        Fact(IsResourceEstimating(), "LegacySelectSwapResourceEstimate is not an executable circuit");
        let (n, nRequired) = DimensionsForSelect(data, address);
        let addressFitted = address[...nRequired - 1];
        let swapBits = numSwapBits == -1 ? ComputeOptimalLambda1D(Length(data), Length(data[0])) | numSwapBits;

        Fact(swapBits <= nRequired, "Too many bits for SWAP network");
        if swapBits == 0 {
            Select(data, addressFitted, output);
        } else {
            WithSelectSwap(swapBits, data, address, intermediate => ApplyToEachCA(CNOT, Zipped(intermediate, output)));
        }
    }

    //  2D SELECT-SWAP (single select-swap over the combined outer×inner address)
    /// Loads `data[outer][inner]` into `target` with one select-swap lookup whose address is
    /// the outer index concatenated with the select part of the inner one. The adjoint is
    /// compiler-generated: it has to undo the swap network as well as the lookup, which is
    /// easy to get wrong by hand.
    ///
    /// The outer index is folded into the `Select` address rather than driven by an enclosing
    /// `UnaryIteration`. Both cost the same forwards, but `Adjoint Select` is a
    /// measurement-based unlookup over the combined address, whereas `Adjoint UnaryIteration`
    /// re-runs its AND ladder at full price -- so the uncompute would otherwise cost as much
    /// as the compute. `TestSelect2DLoadPhaseAgreement` cross-checks this path against the
    /// unary-iteration path in `numSwapBits == 0` over every address state, padded ones
    /// included, because a mismatched routing shows up only as an address-register phase.
    operation Select2DLoad(data : Bool[][][], outerAddress : Qubit[], innerAddress : Qubit[], numSwapBits : Int, target : Qubit[]) : Unit is Adj {
        Fact(not IsEmpty(data), "data cannot be empty");
        let (n, nRequired) = DimensionsForSelect(data[0], innerAddress);
        if numSwapBits == 0 {
            UnaryIteration(outerAddress, Length(data), (index) => {
                Select(PadToAddressSpace(data[index], nRequired), innerAddress[...nRequired - 1], target);
            });
        } else {
            Fact(numSwapBits <= nRequired, "Too many bits for SWAP network");
            let innerAddressFitted = innerAddress[...nRequired - 1];

            let m = Length(data[0][0]);
            let l = numSwapBits;
            let k = nRequired - numSwapBits;

            let innerAddressParts = Partitioned([k, l], innerAddressFitted);
            let chunkedDataRegister = Chunks(m, target);

            // Row-major over the outer index, so the combined address value is
            // innerSelect + outer * 2^k with the inner select bits least significant.
            let flatData = FlattenPaddedData(data, nRequired, m, k);
            let outerAddressFitted = outerAddress[...AddressQubits(Length(data)) - 1];
            Select(flatData, innerAddressParts[0] + outerAddressFitted, target);

            SwapDataOutputs(innerAddressParts[1], chunkedDataRegister);
        }
    }

    /// Historical 2D lookup model retained only for comparison with pre-cleanup estimates.
    ///
    /// TODO: Remove this compatibility model after a phase-correct measurement unlookup
    /// reaches the same cost. Its adjoint does not reconstruct the post-SWAP target over
    /// the complete padded address space and therefore is not a valid executable circuit.
    internal operation LegacySelect2DLoadResourceEstimate(
        data : Bool[][][],
        outerAddress : Qubit[],
        innerAddress : Qubit[],
        numSwapBits : Int,
        target : Qubit[],
    ) : Unit is Adj {
        body (...) {
            Fact(IsResourceEstimating(), "LegacySelect2DLoadResourceEstimate is not an executable circuit");
            if numSwapBits == 0 {
                UnaryIteration(outerAddress, Length(data), (index) => {
                    Select(data[index], innerAddress, target);
                });
            } else {
                let (n, nRequired) = DimensionsForSelect(data[0], innerAddress);
                let innerAddressFitted = innerAddress[...nRequired - 1];
                let m = Length(data[0][0]);
                let k = nRequired - numSwapBits;
                let innerAddressParts = Partitioned([k, numSwapBits], innerAddressFitted);
                let chunkedDataRegister = Chunks(m, target);

                UnaryIteration(outerAddress, Length(data), (index) => {
                    let dataArray = CreatePaddedData(data[index], nRequired, m, k);
                    Select(dataArray, innerAddressParts[0], target);
                });
                SwapDataOutputs(innerAddressParts[1], chunkedDataRegister);
            }
        }
        adjoint (...) {
            Fact(IsResourceEstimating(), "LegacySelect2DLoadResourceEstimate is not an executable circuit");
            let (n, nRequired) = DimensionsForSelect(data[0], innerAddress);
            let mapOne : (Int -> Bool[][]) = index -> CreatePaddedData(data[index], nRequired, Length(data[index][0]), nRequired - numSwapBits);
            let flattenedData = Flattened(MappedOverRange(mapOne, IndexRange(data)));
            Adjoint Select(flattenedData, innerAddress + outerAddress, target);
        }
    }

    function ComputeOptimalLambda2D(numOuterData : Int, numInnerData : Int, numBits : Int) : Int {
        mutable best = 2^32;
        mutable bestLambda = 0;

        let addressBits = Ceiling(Lg(IntAsDouble(numInnerData)));
        for lambda in 0..addressBits - 1 {
            let cost = SelectSwapCost2D(lambda, numOuterData, numInnerData, numBits);
            if cost < best {
                set bestLambda = lambda;
                set best = cost;
            }
        }

        return bestLambda;
    }

    /// Runs `action` on the data word addressed by `address`, then uncomputes the lookup.
    ///
    /// The adjoint is compiler-generated. Uncomputing a select-swap lookup by hand is
    /// error-prone: it must undo the swap network as well as the lookup, and `Adjoint
    /// Select` measures the target and repairs the phase kickback that leaves on the
    /// address register. A wrong uncompute still loads the right values and corrupts only
    /// the address-register phase, which for a state-preparation caller is the state.
    internal operation WithSelectSwap(numSwapBits : Int, data : Bool[][], address : Qubit[], action : (Qubit[] => Unit is Adj + Ctl)) : Unit is Adj + Ctl {
        let (n, nRequired) = DimensionsForSelect(data, address);
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

    /// Toffoli cost of one `Select2DLoad` *and its uncompute*, for a given swap width.
    ///
    /// This mirrors the implementation term by term: a single `Select` over the combined
    /// `(innerSelect, outer)` address of `combinedBits` qubits, the swap network applied once
    /// in each direction, and a measurement-based unlookup on the way back. Costing the round
    /// trip rather than the forward pass is what makes the comparison across `lambda`
    /// meaningful, because the unlookup and the swap network scale in opposite directions.
    ///
    /// `lambda == 0` is the unary-iteration path, whose adjoint re-runs the whole AND ladder
    /// at full price and so costs the same again.
    internal function SelectSwapCost2D(lambda : Int, numOuterData : Int, numInnerData : Int, numBits : Int) : Int {
        let outerAddressBits = Ceiling(Lg(IntAsDouble(numOuterData)));
        let innerAddressBits = Ceiling(Lg(IntAsDouble(numInnerData)));

        if lambda == 0 {
            return 2 * ((numOuterData - 2) + numOuterData * (numInnerData - 2));
        }

        let combinedBits = (outerAddressBits + innerAddressBits) - lambda;
        let selectCost = 2^combinedBits - 2;
        let unselectCost = 2^((combinedBits + 1) / 2) + 2^(combinedBits / 2) - (combinedBits + 2);
        let swapCost = (2^lambda - 1) * numBits;

        return selectCost + unselectCost + 2 * swapCost;
    }

    internal function DimensionsForSelect(data : Bool[][], address : Qubit[]) : (Int, Int) {
        let N = Length(data);
        Fact(N > 0, "data cannot be empty");

        let n = Ceiling(Lg(IntAsDouble(N)));
        Fact(Length(address) >= n, $"address register is too small, requires at least {n} qubits");

        return (N, n);
    }

    internal function CreatePaddedData(data : Bool[][], nRequired : Int, m : Int, k : Int) : Bool[][] {
        let dataPadded = Padded(-2^nRequired, [false, size = m], data);

        MappedOverRange(i -> Flattened(dataPadded[i..2^k..2^nRequired - 1]), 0..2^k - 1)
    }

    /// Concatenates the per-outer-index padded lookup tables into one row-major table, so a
    /// combined `(innerSelect, outer)` address indexes it directly.
    ///
    /// The table covers all `2^AddressQubits(Length(data))` outer address states, not just the
    /// `Length(data)` valid ones, with each unused state routed to the same entry
    /// `UnaryIteration` would have selected. That keeps the lookup identical to the
    /// `numSwapBits == 0` path on every address state and makes the combined table a power of
    /// two, so `Select` never applies aliasing of its own.
    internal function FlattenPaddedData(data : Bool[][][], nRequired : Int, m : Int, k : Int) : Bool[][] {
        let numOuterStates = 1 <<< AddressQubits(Length(data));
        Flattened(
            MappedOverRange(
                state -> CreatePaddedData(data[UnaryIterationActionIndex(Length(data), state)], nRequired, m, k),
                0..numOuterStates - 1
            )
        )
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

    /// 2D Select2DLoad correctness: for each (i,j), load data[i][j] into target,
    /// CNOT to copy, verify.
    internal operation TestSelect2DLoadCorrectness(
        data : Bool[][][],
        numSwapBits : Int
    ) : Bool {
        let nOuter = Length(data);
        let nInner = Length(data[0]);
        let m = Length(data[0][0]);
        let nOuterAddr = Ceiling(Lg(IntAsDouble(nOuter)));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(nInner)));
        let nTarget = if numSwapBits > 0 { m * (1 <<< numSwapBits) } else { m };

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        use target = Qubit[nTarget];
        use copy = Qubit[m];

        mutable allCorrect = true;

        for i in 0..nOuter - 1 {
            for j in 0..nInner - 1 {
                ApplyXorInPlace(i, outerAddr);
                ApplyXorInPlace(j, innerAddr);

                within {
                    Select2DLoad(
                        data,
                        outerAddr,
                        innerAddr,
                        numSwapBits,
                        target
                    );
                } apply {
                    ApplyToEachCA(CNOT, Zipped(target[0..m - 1], copy));
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

    /// Phase-oracle agreement between the 2D swap path and the 2D no-swap path.
    /// See `TestSelectSwap1DPhaseAgreement` for why a value test cannot replace this, and why
    /// a single `true` verdict proves nothing on its own.
    internal operation TestSelect2DLoadPhaseAgreement(data : Bool[][][], numSwapBits : Int) : Bool {
        let m = Length(data[0][0]);
        let nOuterAddr = Ceiling(Lg(IntAsDouble(Length(data))));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(Length(data[0]))));

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        ApplyToEachA(H, outerAddr);
        ApplyToEachA(H, innerAddr);

        {
            use target = Qubit[m * 2^numSwapBits];
            within {
                Select2DLoad(data, outerAddr, innerAddr, numSwapBits, target);
            } apply {
                Z(target[0]);
            }
        }
        {
            use target = Qubit[m];
            within {
                Select2DLoad(data, outerAddr, innerAddr, 0, target);
            } apply {
                Z(target[0]);
            }
        }

        Adjoint ApplyToEachA(H, outerAddr);
        Adjoint ApplyToEachA(H, innerAddr);

        All(r -> r == Zero, MResetEachZ(outerAddr + innerAddr))
    }
}
