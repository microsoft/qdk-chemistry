// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SELECT-SWAP loaders that borrow live caller qubits instead of allocating scratch.
///
/// The dirty-QROAM strategy: the swap block is lent by the caller rather than allocated,
/// so the lookup costs no marginal width, paid for by running `Select` twice and the
/// butterfly four times. See `DirtyQROAMCost` for where that does and does not pay.
///
/// References:
///   Berry et al. (arXiv:1902.02134) App. A Thm 1
///   Low, Kliuchnikov, Schaeffer (arXiv:1812.00954)
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
    import QDKChemistry.Utils.SelectSwapUtils.AliasToAddressSpace;
    import QDKChemistry.Utils.SelectSwapUtils.SwappedLoadShape;
    import QDKChemistry.Utils.SelectSwapUtils.DimensionsForSelect;
    import QDKChemistry.Utils.SelectSwapUtils.CreatePaddedData;
    import QDKChemistry.Utils.SelectSwapUtils.SwapDataOutputs;
    import QDKChemistry.Utils.SelectSwapUtils.MeasurementUnlookupCost;
    import QDKChemistry.Utils.SelectSwap.EraseSwappedLoad;
    import QDKChemistry.Utils.SelectSwap.SelectSwap2D;
    import QDKChemistry.Utils.SelectSwap.SelectSwapCost2D;

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
