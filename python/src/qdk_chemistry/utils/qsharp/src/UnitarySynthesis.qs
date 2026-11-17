// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Circuits for real unitaries given as Givens rotation layers and sign corrections.
namespace QDKChemistry.Utils.UnitarySynthesis {

    import Std.Arithmetic.ApplyIfGreaterL;
    import Std.Arrays.All;
    import Std.Arrays.Any;
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
    import Std.Math.PI;
    import Std.Measurement.MResetX;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.Arithmetic.AddConstant;
    import QDKChemistry.Utils.PhaseGradient.QuantizeRyAngles;
    import QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;
    import QDKChemistry.Utils.SelectSwap.ApplyBranchPhaseFixup;
    import QDKChemistry.Utils.SelectSwap.ComputeOptimalSwapBits;
    import QDKChemistry.Utils.SelectSwap.SelectSwap2D;
    import QDKChemistry.Utils.SelectSwap.SelectSwapAliased;
    import QDKChemistry.Utils.SelectSwap.SelectSwapForwardCost;

    /// Givens representation U = D · L_{m-1} ⋯ L_0 of a real orthogonal matrix, mirroring the
    /// C++ `qdk::chemistry::utils::GivensDecomposition`.
    ///
    /// Layer L_j rotates adjacent basis states by G(θ) = [[cos θ, -sin θ], [sin θ, cos θ]] on
    /// the pairs (0, 1), (2, 3), …, or (1, 2), (3, 4), … when shifted. D = diag(±1).
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
        /// `layers[i][k]` is the `QuantizeRyAngles` word of Ry(2θ) for the k-th angle of layer
        /// i, without trailing zero words; addresses past the end get the identity.
        layers : Bool[][][],
        layerShifted : Bool[],
        /// Empty applies no signs.
        phases : Bool[],
        /// Widest swap network of the angle lookups, -1 for the Toffoli-optimal width.
        maxSwapBits : Int,
    }

    /// Drops the trailing all-zero words of a quantized angle table, whose rotations are the identity.
    function TrimZeroWords(words : Bool[][]) : Bool[][] {
        mutable length = Length(words);
        while length > 0 and not Any(bit -> bit, words[length - 1]) {
            set length -= 1;
        }
        words[...length - 1]
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
        let quantizeLayer = angles -> TrimZeroWords(QuantizeRyAngles(Mapped(angle -> 2.0 * angle, angles), rotationBits));
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

    /// Loads `data[address]` into the clean `target` when `flag` is |1⟩ and leaves it zero
    /// otherwise. The adjoint erases `target` by X-basis measurement and a phase fixup on
    /// `flag` and `address`, so the lookup is never run backwards.
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
            ApplyBranchPhaseFixup(measured, data, true, [flag] + address);
        }
    }

    /// Sets `flag` to the AND of `controls` and, when `guarded`, of `address < bound`.
    operation ComputeLoadFlag(controls : Qubit[], guarded : Bool, bound : Int, address : Qubit[], flag : Qubit) : Unit is Adj {
        if guarded {
            // Fires when bound > address.
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
    /// `address`, loading the word from `QuantizeRyAngles` into the clean b-qubit `angleReg`.
    /// Addresses at or past `Length(data)` get the identity.
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

    /// `ApplyMultiplexedRy` that rotates only when every qubit of `controls` is |1⟩.
    ///
    /// Only the lookup is controlled, since a zero angle register already gives the identity.
    /// This is cheaper than `Controlled ApplyMultiplexedRy`, which also controls the adder.
    /// The lookup is erased by measurement for every swap width (`LoadAngleWord`, or the
    /// adjoint of `SelectSwap2D` for an uncontrolled full table). A table shorter than the
    /// address space is either zero-padded or, when cheaper (`UseAddressGuard`), loaded only
    /// under an `address < Length(data)` flag so that no other word reaches `angleReg`.
    /// `maxSwapBits` caps the swap network width, trading Toffolis for scratch qubits.
    operation ApplyControlledMultiplexedRy(
        data : Bool[][],
        maxSwapBits : Int,
        address : Qubit[],
        controls : Qubit[],
        targetQubit : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj + Ctl {
        if not IsEmpty(data) {
            let numBits = Length(angleReg);
            let guarded = UseAddressGuard(Length(data), Length(address), numBits, maxSwapBits);
            let table = guarded ? data | Padded(-(1 <<< Length(address)), [false, size = numBits], data);
            if not guarded and IsEmpty(controls) {
                let swapBits = ComputeOptimalSwapBits(Length(table), numBits, maxSwapBits);
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

    /// Applies D = diag((-1)^{f(x)}) on a little-endian register, with `phases[x]` = f(x).
    ///
    /// The Möbius transform over GF(2) gives f(x) = ⊕_S c_S ∏_{i∈S} x_i, and each nonzero c_S
    /// becomes a Z, CZ, CCZ, … gate on the qubits in S. The constant c_∅ is the global sign
    /// -I = R(PauliI, 2π). D is its own inverse.
    ///
    /// Controlled on the qubits C, the phase is AND(C) · f(x) = ⊕_S c_S ∏_{i∈S∪C} x_i, so each
    /// gate also acts on C and c_∅ becomes a Z, CZ, … on C alone. Only the 2^n table of the
    /// register is transformed, so the classical cost does not grow with the number of controls.
    operation ApplyPhasePolynomial(phases : Bool[], register : Qubit[]) : Unit is Adj + Ctl {
        body (...) {
            Controlled ApplyPhasePolynomial([], (phases, register));
        }
        adjoint self;
        controlled (controls, ...) {
            let n = Length(register);
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
            for s in Where(c -> c, coeffs) {
                let qubits = Subarray(Where(b -> b, IntAsBoolArray(s, n)), register) + controls;
                if IsEmpty(qubits) {
                    Fact(n > 0, "A constant sign needs a nonempty register to carry it.");
                    R(PauliI, 2.0 * PI(), register[0]);
                } else {
                    Controlled Z(Most(qubits), Tail(qubits));
                }
            }
        }
        controlled adjoint self;
    }

    /// Applies the real unitary U = D · L_{m-1} ⋯ L_0 of `givens` to `target`, controlled on
    /// every qubit of `controls` (none applies it unconditionally).
    ///
    /// `target` is most significant qubit first. Its last qubit is the active qubit of every
    /// layer and the others address the angle lookup, so a block-diagonal unitary is applied
    /// by passing the block-selecting qubits first. Each layer loads its angles with
    /// `ApplyControlledMultiplexedRy` and rotates through the phase gradient (Berry et al., PRX Quantum 6,
    /// 020327, https://doi.org/10.1103/PRXQuantum.6.020327; Clements et al.,
    /// arXiv:1603.08788), and D is applied by `ApplyPhasePolynomial`.
    ///
    /// Only the angle lookups and D are controlled: with a control in |0⟩ the angle register
    /// stays zero, Ry(0) = I, and the shift/unshift pair of each shifted layer cancels.
    operation ApplyRealUnitaryViaGivens(
        givens : QuantizedGivensDecomposition,
        controls : Qubit[],
        target : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        let n = Length(target);
        Fact(Length(givens.layerShifted) == Length(givens.layers), "Givens data needs one shift flag per layer.");
        Fact(IsEmpty(givens.phases) or Length(givens.phases) == 1 <<< n, "Givens phase data needs one entry per basis state of the target register.");
        Fact(n > 0 or IsEmpty(givens.layers), "Givens rotation layers need a nonempty target register.");

        // Layers differ only in data, so cache the shifted and unshifted circuits. The key holds
        // every width that sets a layer's cost, so compositions at other precisions or table
        // sizes are counted separately.
        for (layer, shifted) in Zipped(givens.layers, givens.layerShifted) {
            // A layer without nonzero words is the identity.
            if not IsEmpty(layer) {
                let cacheName = $"GivensLayer_c{Length(controls)}_g{Length(phaseGradient)}_a{Length(angleReg)}_d{Length(layer)}_s{givens.maxSwapBits}";
                if BeginEstimateCaching(cacheName, 2 * n + (shifted ? 1 | 0)) {
                    // Decrementing maps the shifted pairs (1, 2), (3, 4), … onto (0, 1), (2, 3), ….
                    within {
                        if shifted { AddConstant(-1, Reversed(target)); }
                    } apply {
                        ApplyControlledMultiplexedRy(layer, givens.maxSwapBits, Reversed(Most(target)), controls, Tail(target), phaseGradient, angleReg);
                    }
                    EndEstimateCaching();
                }
            }
        }
        Controlled ApplyPhasePolynomial(controls, (givens.phases, Reversed(target)));
    }
}
