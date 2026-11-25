// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Circuits for real unitaries given as Givens rotation layers and sign corrections.
namespace QDKChemistry.Utils.UnitarySynthesis {

    import Std.Arithmetic.ApplyIfGreaterL;
    import Std.Arrays.All;
    import Std.Arrays.Chunks;
    import Std.Arrays.Flattened;
    import Std.Arrays.ForEach;
    import Std.Arrays.IndexRange;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.Mapped;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Most;
    import Std.Arrays.Padded;
    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Arrays.Tail;
    import Std.Arrays.Where;
    import Std.Arrays.Zipped;
    import Std.Convert.IntAsBigInt;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.ResultArrayAsBoolArray;
    import Std.Diagnostics.Fact;
    import Std.Math.MaxI;
    import Std.Math.MinI;
    import Std.Math.PI;
    import Std.Measurement.MResetX;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.Arithmetic.AddConstant;
    import QDKChemistry.Utils.PhaseGradient.QuantizeRyAngle;
    import QDKChemistry.Utils.PhaseGradient.QuantizeRyAngles;
    import QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;
    import QDKChemistry.Utils.SelectSwap.ApplyBranchPhaseFixup;
    import QDKChemistry.Utils.SelectSwap.ComputeOptimalLambda2D;
    import QDKChemistry.Utils.SelectSwap.ComputeOptimalSwapBits;
    import QDKChemistry.Utils.SelectSwap.CreatePaddedData;
    import QDKChemistry.Utils.SelectSwap.MeasurementUnlookupCost;
    import QDKChemistry.Utils.SelectSwap.SelectSwap2D;
    import QDKChemistry.Utils.SelectSwap.SelectSwapAliased;
    import QDKChemistry.Utils.SelectSwap.SelectSwapForwardCost;
    import QDKChemistry.Utils.SelectSwap.SwapDataOutputs;
    import QDKChemistry.Utils.SelectSwap.SwapNetworkPermutation;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;

    /// Givens representation U = D · L_{m-1} ⋯ L_0 of a real orthogonal matrix, mirroring the
    /// C++ `qdk::chemistry::utils::GivensDecomposition`, where layer L_j applies
    /// G(θ) = [[cos θ, -sin θ], [sin θ, cos θ]] to the pairs (0, 1), (2, 3), … or, when shifted,
    /// (1, 2), (3, 4), … and D = diag(±1).
    struct GivensDecomposition {
        /// Rotation angles θ of each layer, by increasing pair index.
        layerAngles : Double[][],
        /// Whether each layer acts on the odd-starting pairs.
        layerShifted : Bool[],
        /// Whether each basis state receives a minus sign from D.
        phases : Bool[],
    }

    /// `GivensDecomposition` with each layer quantized into a Select table.
    struct QuantizedGivensDecomposition {
        /// `QuantizeRyAngles` words of Ry(2θ) per layer, without trailing zero words.
        layers : Bool[][][],
        layerShifted : Bool[],
        /// Empty applies no signs.
        phases : Bool[],
        /// Widest swap network of the angle lookups, -1 to let each loader's cost model choose.
        maxSwapBits : Int,
    }

    /// `QuantizeRyAngles` without the trailing angles that quantize to the zero word, which are identity rotations.
    function QuantizeTrimmedRyAngles(angles : Double[], bits : Int) : Bool[][] {
        mutable length = Length(angles);
        while length > 0 and QuantizeRyAngle(angles[length - 1], bits) == 0 {
            set length -= 1;
        }
        QuantizeRyAngles(angles[...length - 1], bits)
    }

    /// Quantizes every layer of `givens` into words of `rotationBits` bits, at most
    /// `numAddresses` per layer.
    function QuantizeGivensDecomposition(
        givens : GivensDecomposition,
        numAddresses : Int,
        rotationBits : Int,
        maxSwapBits : Int
    ) : QuantizedGivensDecomposition {
        Fact(All(angles -> Length(angles) <= numAddresses, givens.layerAngles), "A Givens layer has more angles than addresses.");
        // G(θ) is Ry(2θ) on the active qubit.
        let quantizeLayer = angles -> QuantizeTrimmedRyAngles(Mapped(angle -> 2.0 * angle, angles), rotationBits);
        new QuantizedGivensDecomposition {
            layers = Mapped(quantizeLayer, givens.layerAngles),
            layerShifted = givens.layerShifted,
            phases = givens.phases,
            maxSwapBits = maxSwapBits,
        }
    }

    /// Whether a table of `numData` words is cheaper to load behind an `address < numData`
    /// flag than zero-padded to all `2^addressBits` addresses.
    function UseAddressGuard(numData : Int, addressBits : Int, numBits : Int, maxSwapBits : Int) : Bool {
        let full = 1 <<< addressBits;
        if numData >= full {
            return false;
        }
        let paddedCost = SelectSwapForwardCost(ComputeOptimalSwapBits(full, numBits, maxSwapBits), full, numBits);
        let trimmedCost = SelectSwapForwardCost(ComputeOptimalSwapBits(numData, numBits, maxSwapBits), numData, numBits);
        // The comparator is computed and uncomputed.
        trimmedCost + 2 * addressBits < paddedCost
    }

    /// Loads `data[address]` into the clean `target` when `flag` is |1⟩, with an adjoint that
    /// erases it by measurement.
    operation LoadAngleWord(
        data : Bool[][],
        maxSwapBits : Int,
        address : Qubit[],
        flag : Qubit,
        target : Qubit[]
    ) : Unit is Adj {
        body (...) {
            let swapBits = ComputeOptimalSwapBits(Length(data), Length(target), maxSwapBits);
            if swapBits == 0 {
                Controlled Select([flag], (data, address, target));
            } else {
                // Only the copy out is controlled; the scratch load is erased by measurement.
                Controlled SelectSwapAliased([flag], (swapBits, data, address, target));
            }
        }
        adjoint (...) {
            let measured = ResultArrayAsBoolArray(ForEach(MResetX, target));
            // The load reads only the address prefix that enumerates `data`.
            ApplyBranchPhaseFixup(measured, data, true, [flag] + address[...AddressQubits(Length(data)) - 1]);
        }
    }

    /// Sets `flag` to the AND of `controls` and, when `guarded`, of `address < bound`.
    operation ComputeLoadFlag(controls : Qubit[], guarded : Bool, bound : Int, address : Qubit[], flag : Qubit) : Unit is Adj {
        if guarded {
            if IsEmpty(controls) {
                ApplyIfGreaterL(X, IntAsBigInt(bound), address, flag);
            } else {
                ApplyIfGreaterL(Controlled X(controls, _), IntAsBigInt(bound), address, flag);
            }
        } else {
            Controlled X(controls, flag);
        }
    }

    /// Applies Ry(4π·data[a]/2^b) to `targetQubit` for each state |a⟩ of the little-endian
    /// `address`, with the identity past `Length(data)` and a clean b-qubit `angleReg`.
    operation ApplyMultiplexedRy(
        data : Bool[][],
        maxSwapBits : Int,
        address : Qubit[],
        targetQubit : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj + Ctl {
        ApplyControlledMultiplexedRy(data, maxSwapBits, address, [], targetQubit, phaseGradient, angleReg);
    }

    /// `ApplyMultiplexedRy` that rotates only when every qubit of `controls` is |1⟩, controlling
    /// only the lookup, which is cheaper than `Controlled ApplyMultiplexedRy`.
    operation ApplyControlledMultiplexedRy(
        data : Bool[][],
        maxSwapBits : Int,
        address : Qubit[],
        controls : Qubit[],
        targetQubit : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj + Ctl {
        body (...) {
            if not IsEmpty(data) {
                let numBits = Length(angleReg);
                let guarded = UseAddressGuard(Length(data), Length(address), numBits, maxSwapBits);
                let table = guarded ? data | Padded(-(1 <<< Length(address)), [false, size = numBits], data);
                if not guarded and IsEmpty(controls) {
                    let swapBits = ComputeOptimalLambda2D(1, Length(table), numBits, true, maxSwapBits);
                    within {
                        SelectSwap2D([table], swapBits, true, [], address, [], angleReg);
                    } apply {
                        RyViaPhaseGradient(targetQubit, angleReg, phaseGradient);
                    }
                } elif not guarded and Length(controls) == 1 {
                    within {
                        LoadAngleWord(table, maxSwapBits, address, controls[0], angleReg);
                    } apply {
                        RyViaPhaseGradient(targetQubit, angleReg, phaseGradient);
                    }
                } else {
                    use flag = Qubit();
                    within {
                        ComputeLoadFlag(controls, guarded, Length(table), address, flag);
                        LoadAngleWord(table, maxSwapBits, address, flag, angleReg);
                    } apply {
                        RyViaPhaseGradient(targetQubit, angleReg, phaseGradient);
                    }
                }
            }
        }
        adjoint auto;
        // External controls join the lookup controls, as the phase-gradient adder takes exactly one control.
        controlled (ctls, ...) {
            ApplyControlledMultiplexedRy(data, maxSwapBits, address, ctls + controls, targetQubit, phaseGradient, angleReg);
        }
        controlled adjoint (ctls, ...) {
            Adjoint ApplyControlledMultiplexedRy(data, maxSwapBits, address, ctls + controls, targetQubit, phaseGradient, angleReg);
        }
    }

    /// Applies D = diag((-1)^{phases[x]}) on a little-endian register by multi-controlled Z gates
    /// from the Möbius transform of `phases`, or by the measurement-based sign fixup of Berry et
    /// al. (Quantum 3, 208 (2019)) when that needs fewer Toffolis.
    operation ApplyPhasePolynomial(phases : Bool[], register : Qubit[]) : Unit is Adj + Ctl {
        body (...) {
            Controlled ApplyPhasePolynomial([], (phases, register));
        }
        adjoint self;
        controlled (controls, ...) {
            let n = Length(register);
            let numControls = Length(controls);
            Fact(IsEmpty(phases) or Length(phases) == 1 <<< n, "Phase data needs one entry per basis state of the register.");
            // In-place Möbius butterfly; bit i of index s marks qubit i of the monomial S.
            mutable coeffs = phases;
            for i in 0..n - 1 {
                for s in IndexRange(coeffs) {
                    if (s >>> i) &&& 1 == 1 {
                        set coeffs w/= s <- coeffs[s] != coeffs[s - (1 <<< i)];
                    }
                }
            }
            let monomials = Mapped(s -> Subarray(Where(b -> b, IntAsBoolArray(s, n)), register), Where(c -> c, coeffs));
            // Toffolis of an X or Z on a target with m controls, as Q# decomposes it.
            let gateCost = m -> MaxI(0, 2 * m - 3);
            mutable polynomialCost = 0;
            for qubits in monomials {
                set polynomialCost += gateCost(Length(qubits) + numControls - 1);
            }
            // The lookup reads AND(controls) from one control or from a computed flag.
            let lookupCost = MeasurementUnlookupCost(n + MinI(numControls, 1)) + (numControls > 1 ? 2 * gateCost(numControls) | 0);
            if lookupCost < polynomialCost {
                if numControls == 0 {
                    ApplySignTable(phases, register);
                } elif numControls == 1 {
                    ApplyBranchPhaseFixup([true], Mapped(phase -> [phase], phases), true, controls + register);
                } else {
                    use flag = Qubit();
                    within {
                        Controlled X(controls, flag);
                    } apply {
                        ApplyBranchPhaseFixup([true], Mapped(phase -> [phase], phases), true, [flag] + register);
                    }
                }
            } else {
                for qubits in monomials {
                    let gateQubits = qubits + controls;
                    if IsEmpty(gateQubits) {
                        Fact(n > 0, "A constant sign needs a nonempty register to carry it.");
                        R(PauliI, 2.0 * PI(), register[0]);
                    } else {
                        Controlled Z(Most(gateQubits), Tail(gateQubits));
                    }
                }
            }
        }
        controlled adjoint self;
    }

    /// Returns the parity of `measured` against `word`, the sign an X-basis erasure leaves on it.
    internal function MeasuredParity(measured : Bool[], word : Bool[]) : Bool {
        mutable parity = false;
        for (measuredBit, wordBit) in Zipped(measured, word) {
            set parity = parity != (measuredBit and wordBit);
        }
        parity
    }

    /// Returns the entrywise XOR of two sign tables.
    internal function XorSigns(left : Bool[], right : Bool[]) : Bool[] {
        Fact(Length(left) == Length(right), "Sign tables need equal lengths.");
        Mapped((leftSign, rightSign) -> leftSign != rightSign, Zipped(left, right))
    }

    /// Applies diag((-1)^{phases[x]}) on the little-endian `register` by one measurement-based lookup.
    internal operation ApplySignTable(phases : Bool[], register : Qubit[]) : Unit {
        Fact(Length(phases) == 1 <<< Length(register), "Sign data needs one entry per basis state of the register.");
        // The lookup fixes signs relative to phases[0], which is applied as -I.
        if not All(phase -> phase == phases[0], phases) {
            use marker = Qubit();
            X(marker);
            H(marker);
            Adjoint Select(Mapped(phase -> [phase != phases[0]], phases), register, [marker]);
            Reset(marker);
        }
        if phases[0] {
            Fact(not IsEmpty(register), "A constant sign needs a nonempty register to carry it.");
            R(PauliI, 2.0 * PI(), register[0]);
        }
    }

    /// Loads `table[address]` into the clean `target` when every qubit of `controls` is |1⟩ and measures out
    /// the swap scratch, returning its sign per address prefix (one entry without a swap network).
    internal operation LoadWordDeferred(
        table : Bool[][],
        swapBits : Int,
        address : Qubit[],
        controls : Qubit[],
        target : Qubit[]
    ) : Bool[] {
        let addressBits = AddressQubits(Length(table));
        if swapBits == 0 {
            Controlled Select(controls, (table, address[...addressBits - 1], target));
            [false]
        } else {
            let selectBits = addressBits - swapBits;
            let wordBits = Length(target);
            let flat = CreatePaddedData(table, addressBits, wordBits, selectBits);
            use scratch = Qubit[wordBits <<< swapBits];
            Select(flat, address[...selectBits - 1], scratch);
            SwapDataOutputs(address[selectBits..addressBits - 1], Chunks(wordBits, scratch));
            for (source, copy) in Zipped(scratch, target) {
                Controlled CNOT(controls, (source, copy));
            }
            let measured = ResultArrayAsBoolArray(ForEach(MResetX, scratch));
            // Each parity pairs the measured chunks, moved back through the swap network, with the unswapped row.
            let measuredChunks = Chunks(wordBits, measured);
            let unswapped = MappedOverRange(
                swap -> {
                    let permutation = SwapNetworkPermutation(swapBits, swap);
                    mutable chunks = measuredChunks;
                    for position in IndexRange(permutation) {
                        set chunks w/= permutation[position] <- measuredChunks[position];
                    }
                    Flattened(chunks)
                },
                0..(1 <<< swapBits) - 1
            );
            MappedOverRange(
                state -> MeasuredParity(unswapped[state >>> selectBits], flat[state % (1 <<< selectBits)]),
                0..(1 <<< addressBits) - 1
            )
        }
    }

    /// `ApplyControlledMultiplexedRy` that measures out its loads without phase fixups and returns their signs,
    /// indexed by a + 2^{Length(address)}·AND(controls) for address state a, with no AND term without controls.
    internal operation ApplyMultiplexedRyDeferred(
        data : Bool[][],
        maxSwapBits : Int,
        address : Qubit[],
        controls : Qubit[],
        targetQubit : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Bool[] {
        let addressBits = Length(address);
        let numSlices = IsEmpty(controls) ? 1 | 2;
        if IsEmpty(data) {
            [false, size = numSlices <<< addressBits]
        } else {
            let numBits = Length(angleReg);
            let guarded = UseAddressGuard(Length(data), addressBits, numBits, maxSwapBits);
            let table = guarded ? data | Padded(-(1 <<< addressBits), [false, size = numBits], data);
            let swapBits = not guarded and IsEmpty(controls) ? ComputeOptimalLambda2D(1, Length(table), numBits, true, maxSwapBits) | ComputeOptimalSwapBits(Length(table), numBits, maxSwapBits);
            // A flag holds the load condition unless that is a single control or always true.
            let needsFlag = guarded or Length(controls) > 1;
            use flag = Qubit[needsFlag ? 1 | 0];
            if needsFlag {
                ComputeLoadFlag(controls, guarded, Length(table), address, flag[0]);
            }
            let scratchSigns = LoadWordDeferred(table, swapBits, address, needsFlag ? flag | controls, angleReg);
            RyViaPhaseGradient(targetQubit, angleReg, phaseGradient);
            let measured = ResultArrayAsBoolArray(ForEach(MResetX, angleReg));
            if needsFlag {
                Adjoint ComputeLoadFlag(controls, guarded, Length(table), address, flag[0]);
            }
            MappedOverRange(
                state -> {
                    let addressValue = state % (1 <<< addressBits);
                    let loaded = state >>> addressBits == numSlices - 1 and addressValue < Length(table);
                    (loaded and MeasuredParity(measured, table[addressValue])) != scratchSigns[addressValue % Length(scratchSigns)]
                },
                0..(numSlices <<< addressBits) - 1
            )
        }
    }

    /// Applies the layers of `givens`, or their adjoints in reverse order when `reverse`, to a register whose
    /// state carries the pending `signs`, and returns them with the measured-out loads and D added.
    /// Signs are indexed by t + 2^n·AND(controls) for the value t of `Reversed(target)`, with no AND term
    /// without controls.
    internal operation ApplyGivensLayersDeferred(
        givens : QuantizedGivensDecomposition,
        reverse : Bool,
        controls : Qubit[],
        target : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[],
        signs : Bool[]
    ) : Bool[] {
        let n = Length(target);
        let numStates = 1 <<< n;
        let numLayers = Length(givens.layers);
        Fact(Length(givens.layerShifted) == numLayers, "Givens data needs one shift flag per layer.");
        Fact(IsEmpty(givens.phases) or Length(givens.phases) == numStates, "Givens phase data needs one entry per basis state of the target register.");
        Fact(n > 0 or All(layer -> IsEmpty(layer), givens.layers), "Givens rotation layers need a nonempty target register.");
        Fact(Length(signs) == (IsEmpty(controls) ? numStates | 2 * numStates), "Pending signs need one entry per target state and AND(controls).");
        // The layers and D act on the slice where every control is set, the top `numStates` entries.
        let allOn = Length(signs) - numStates;
        let phaseSigns = IsEmpty(givens.phases) ? [false, size = Length(signs)] | MappedOverRange(state -> state >= allOn and givens.phases[state - allOn], IndexRange(signs));
        // D is diagonal, so it is added to the pending signs instead of applied.
        mutable updated = reverse ? XorSigns(signs, phaseSigns) | signs;
        for step in 0..numLayers - 1 {
            let layerIndex = reverse ? numLayers - 1 - step | step;
            let layer = givens.layers[layerIndex];
            if not IsEmpty(layer) {
                let shift = givens.layerShifted[layerIndex] ? 1 | 0;
                // Layers differ only in data, so the cache key holds every width that sets a layer's cost.
                let cacheName = $"GivensLayer_c{Length(controls)}_g{Length(phaseGradient)}_a{Length(angleReg)}_d{Length(layer)}_s{givens.maxSwapBits}";
                if BeginEstimateCaching(cacheName, 2 * n + shift) {
                    let pending = updated;
                    // Z·Ry(φ)·Z = Ry(-φ), so a pair with unequal pending signs rotates by the negated word.
                    let words = MappedOverRange(
                        pair -> {
                            let lowSign = pending[allOn + (2 * pair + shift) % numStates];
                            let highSign = pending[allOn + (2 * pair + 1 + shift) % numStates];
                            let negate = reverse != (lowSign != highSign);
                            mutable word : Bool[] = [];
                            mutable lowerBitSet = false;
                            for bit in layer[pair] {
                                set word += [bit != (negate and lowerBitSet)];
                                set lowerBitSet = lowerBitSet or bit;
                            }
                            word
                        },
                        IndexRange(layer)
                    );
                    // Decrementing maps the shifted pairs (1, 2), (3, 4), … onto (0, 1), (2, 3), ….
                    if shift == 1 {
                        AddConstant(-1, Reversed(target));
                    }
                    let erased = ApplyMultiplexedRyDeferred(words, givens.maxSwapBits, Reversed(Most(target)), controls, Tail(target), phaseGradient, angleReg);
                    if shift == 1 {
                        Adjoint AddConstant(-1, Reversed(target));
                    }
                    // The erasure signs are indexed in the decremented frame.
                    set updated = MappedOverRange(
                        state -> pending[state] != erased[((state % numStates - shift + numStates) % numStates) / 2 + ((state / numStates) <<< (n - 1))],
                        IndexRange(pending)
                    );
                    EndEstimateCaching();
                }
            }
        }
        reverse ? updated | XorSigns(updated, phaseSigns)
    }

    /// Applies the real unitary of `givens`, controlled on `controls`, to the most-significant-first
    /// `target` with phase-gradient Givens layers (Berry et al., PRX Quantum 6, 020327; Clements et
    /// al., arXiv:1603.08788), where leading block-selecting qubits give a block-diagonal unitary.
    /// The layers' measured-out loads are fixed together with D by one sign lookup.
    operation ApplyRealUnitaryViaGivens(
        givens : QuantizedGivensDecomposition,
        controls : Qubit[],
        target : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        body (...) {
            ApplyRealUnitaryViaGivensInDirection(givens, false, controls, target, phaseGradient, angleReg);
        }
        adjoint (...) {
            ApplyRealUnitaryViaGivensInDirection(givens, true, controls, target, phaseGradient, angleReg);
        }
    }

    /// Applies `ApplyRealUnitaryViaGivens`, or its adjoint when `reverse`.
    internal operation ApplyRealUnitaryViaGivensInDirection(
        givens : QuantizedGivensDecomposition,
        reverse : Bool,
        controls : Qubit[],
        target : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit {
        if All(layer -> IsEmpty(layer), givens.layers) {
            Controlled ApplyPhasePolynomial(controls, (givens.phases, Reversed(target)));
        } else {
            let numSigns = (IsEmpty(controls) ? 1 | 2) <<< Length(target);
            let signs = ApplyGivensLayersDeferred(givens, reverse, controls, target, phaseGradient, angleReg, [false, size = numSigns]);
            if Length(controls) > 1 {
                use flag = Qubit();
                within {
                    Controlled X(controls, flag);
                } apply {
                    ApplySignTable(signs, Reversed(target) + [flag]);
                }
            } else {
                ApplySignTable(signs, Reversed(target) + controls);
            }
        }
    }
}
