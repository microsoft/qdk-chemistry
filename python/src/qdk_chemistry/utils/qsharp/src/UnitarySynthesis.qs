// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
//
// Portions of this file are adapted from code by Felix Rupprecht published at
// https://zenodo.org/records/20393500, Copyright 2026 German Aerospace Center
// (DLR), licensed under the Apache License, Version 2.0, and modified for QDK
// Chemistry.

/// Circuits for real unitaries given as Givens rotation layers and sign corrections.
namespace QDKChemistry.Utils.UnitarySynthesis {

    import Std.Arrays.IndexRange;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.Mapped;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Most;
    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Arrays.Tail;
    import Std.Arrays.Where;
    import Std.Arrays.Zipped;
    import Std.Convert.IntAsBoolArray;
    import Std.Diagnostics.Fact;
    import Std.Math.PI;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import QDKChemistry.Utils.Arithmetic.AddConstant;
    import QDKChemistry.Utils.PhaseGradient.QuantizeRyAngles;
    import QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;
    import QDKChemistry.Utils.SelectSwap.SelectSwap;

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
        /// `layers[i][k]` is the `QuantizeRyAngles` word of Ry(2θ) for the k-th angle of layer i.
        layers : Bool[][][],
        layerShifted : Bool[],
        /// Empty applies no signs.
        phases : Bool[],
    }

    /// Quantizes every layer of `givens` into `numAddresses` words of `rotationBits` bits.
    function QuantizeGivensDecomposition(
        givens : GivensDecomposition,
        numAddresses : Int,
        rotationBits : Int
    ) : QuantizedGivensDecomposition {
        // G(θ) is Ry(2θ) on the active qubit; addresses past the last pair get the identity.
        let quantizeLayer = angles -> QuantizeRyAngles(
            MappedOverRange(k -> k < Length(angles) ? 2.0 * angles[k] | 0.0, 0..numAddresses - 1),
            rotationBits
        );
        new QuantizedGivensDecomposition {
            layers = Mapped(quantizeLayer, givens.layerAngles),
            layerShifted = givens.layerShifted,
            phases = givens.phases,
        }
    }

    /// Applies Ry(4π·data[a]/2^b) to `targetQubit` for each state |a⟩ of the little-endian
    /// `address`, loading the word from `QuantizeRyAngles` into the clean b-qubit `angleReg`
    /// with `SelectSwap`.
    operation ApplyMultiplexedRy(
        data : Bool[][],
        address : Qubit[],
        targetQubit : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj + Ctl {
        ApplyControlledMultiplexedRy(data, address, [], targetQubit, phaseGradient, angleReg);
    }

    /// `ApplyMultiplexedRy` that rotates only when every qubit of `controls` is |1⟩.
    ///
    /// Only the lookup is controlled, since a zero angle register already gives the identity.
    /// This is cheaper than `Controlled ApplyMultiplexedRy`, which also controls the adder.
    /// Without controls the lookup is erased by measurement; `Controlled Adjoint SelectSwap`
    /// would erase it coherently even for an empty control register.
    operation ApplyControlledMultiplexedRy(
        data : Bool[][],
        address : Qubit[],
        controls : Qubit[],
        targetQubit : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj + Ctl {
        within {
            if IsEmpty(controls) {
                SelectSwap(-1, data, address, angleReg);
            } else {
                Controlled SelectSwap(controls, (-1, data, address, angleReg));
            }
        } apply {
            RyViaPhaseGradient(targetQubit, angleReg, phaseGradient);
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
    /// `SelectSwap` and rotates through the phase gradient (Berry et al., PRX Quantum 6,
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
            let cacheName = $"GivensLayer_c{Length(controls)}_g{Length(phaseGradient)}_a{Length(angleReg)}_d{Length(layer)}";
            if BeginEstimateCaching(cacheName, 2 * n + (shifted ? 1 | 0)) {
                // Decrementing maps the shifted pairs (1, 2), (3, 4), … onto (0, 1), (2, 3), ….
                within {
                    if shifted { AddConstant(-1, Reversed(target)); }
                } apply {
                    ApplyControlledMultiplexedRy(layer, Reversed(Most(target)), controls, Tail(target), phaseGradient, angleReg);
                }
                EndEstimateCaching();
            }
        }
        Controlled ApplyPhasePolynomial(controls, (givens.phases, Reversed(target)));
    }
}
