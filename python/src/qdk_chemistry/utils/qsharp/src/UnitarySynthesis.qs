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
    import Std.Math.MaxI;
    import Std.Math.MinI;
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
    import QDKChemistry.Utils.SelectSwap.MeasurementUnlookupCost;
    import QDKChemistry.Utils.SelectSwap.SelectSwap2D;
    import QDKChemistry.Utils.SelectSwap.SelectSwapAliased;
    import QDKChemistry.Utils.SelectSwap.SelectSwapForwardCost;
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
        /// Widest swap network of the angle lookups, -1 for the Toffoli-optimal width.
        maxSwapBits : Int,
    }

    /// Drops the trailing all-zero words, which are identity rotations, of a quantized angle table.
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
                    // The erasure fixes signs relative to phases[0], which is applied as -I.
                    use marker = Qubit();
                    X(marker);
                    H(marker);
                    Adjoint Select(Mapped(phase -> [phase != phases[0]], phases), register, [marker]);
                    Reset(marker);
                    if phases[0] {
                        R(PauliI, 2.0 * PI(), register[0]);
                    }
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

    /// Applies the real unitary of `givens`, controlled on `controls`, to the most-significant-first
    /// `target` with phase-gradient Givens layers (Berry et al., PRX Quantum 6, 020327; Clements et
    /// al., arXiv:1603.08788), where leading block-selecting qubits give a block-diagonal unitary.
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

        // Layers differ only in data, so the cache key holds every width that sets a layer's cost.
        for (layer, shifted) in Zipped(givens.layers, givens.layerShifted) {
            if not IsEmpty(layer) {
                let cacheName = $"GivensLayer_c{Length(controls)}_g{Length(phaseGradient)}_a{Length(angleReg)}_d{Length(layer)}_s{givens.maxSwapBits}";
                ApplyGivensLayer(cacheName, layer, shifted, givens.maxSwapBits, controls, target, phaseGradient, angleReg);
            }
        }
        Controlled ApplyPhasePolynomial(controls, (givens.phases, Reversed(target)));
    }

    /// Applies one layer of `ApplyRealUnitaryViaGivens`, estimated once per `cacheName` and shift
    /// with an explicit adjoint so that `EndEstimateCaching` follows the layer.
    operation ApplyGivensLayer(
        cacheName : String,
        layer : Bool[][],
        shifted : Bool,
        maxSwapBits : Int,
        controls : Qubit[],
        target : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        body ... {
            if BeginEstimateCaching(cacheName, 2 * Length(target) + (shifted ? 1 | 0)) {
                // Decrementing maps the shifted pairs (1, 2), (3, 4), … onto (0, 1), (2, 3), ….
                within {
                    if shifted { AddConstant(-1, Reversed(target)); }
                } apply {
                    ApplyControlledMultiplexedRy(layer, maxSwapBits, Reversed(Most(target)), controls, Tail(target), phaseGradient, angleReg);
                }
                EndEstimateCaching();
            }
        }
        adjoint ... {
            if BeginEstimateCaching(cacheName, 2 * Length(target) + (shifted ? 1 | 0)) {
                within {
                    if shifted { AddConstant(-1, Reversed(target)); }
                } apply {
                    Adjoint ApplyControlledMultiplexedRy(layer, maxSwapBits, Reversed(Most(target)), controls, Tail(target), phaseGradient, angleReg);
                }
                EndEstimateCaching();
            }
        }
    }
}
