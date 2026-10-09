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

    import Std.Arrays.Any;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.Mapped;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Reversed;
    import Std.Diagnostics.Fact;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import QDKChemistry.Utils.Arithmetic.AddConstant;
    import QDKChemistry.Utils.PhaseGradient.ApplyControlledMultiplexedRy;
    import QDKChemistry.Utils.PhaseGradient.ApplyMultiplexedRy;
    import QDKChemistry.Utils.PhaseGradient.QuantizeRyAngles;

    export GivensDecomposition, QuantizedGivensDecomposition, QuantizeGivensDecomposition, ApplyRealUnitaryViaGivens;

    // =============================================================================
    // Givens decomposition data
    // =============================================================================

    /// # Summary
    /// Givens representation U = D · L_{m-1} ⋯ L_0 of a real orthogonal matrix, mirroring the
    /// C++ `qdk::chemistry::utils::detail::GivensDecomposition`.
    ///
    /// # Description
    /// Layer L_j rotates adjacent basis states by G(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]. It
    /// acts on the pairs (0, 1), (2, 3), …, or on (1, 2), (3, 4), … when it is shifted. D is a
    /// diagonal sign matrix.
    ///
    /// # Input
    /// ## layerAngles
    /// Double[numLayers][numPairs]: rotation angles θ of each layer, by increasing pair index.
    /// ## layerShifted
    /// Bool[numLayers]: whether each layer acts on the odd-starting pairs.
    /// ## phases
    /// Bool[dim]: whether each basis state receives a minus sign from D.
    struct GivensDecomposition {
        layerAngles : Double[][],
        layerShifted : Bool[],
        phases : Bool[],
    }

    /// # Summary
    /// `GivensDecomposition` with the angles of each layer quantized into a Select table.
    ///
    /// # Input
    /// ## layers
    /// Bool[numLayers][numAddresses][rotationBits]: `layers[i][k]` holds the k-th angle θ of
    /// layer i as the rotationBits-bit integer x with θ = 2π·x/2^rotationBits.
    /// ## layerShifted
    /// Bool[numLayers]: whether each layer acts on the odd-starting pairs.
    /// ## phases
    /// Bool[dim]: whether each basis state receives a minus sign. An empty array applies no
    /// signs.
    struct QuantizedGivensDecomposition {
        layers : Bool[][][],
        layerShifted : Bool[],
        phases : Bool[],
    }

    /// # Summary
    /// Quantizes the angles of every layer of a `GivensDecomposition` into `numAddresses` words.
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

    // =============================================================================
    // Phase polynomial correction (Reed-Muller decomposition)
    // =============================================================================

    /// # Summary
    /// Applies a phase correction D = diag(±1) using the Reed-Muller polynomial decomposition.
    ///
    /// # Description
    /// For an n-qubit diagonal D = diag((-1)^{f(x)}) where f: {0,1}^n → {0,1}, the Möbius
    /// transform over GF(2) gives the multilinear polynomial
    ///   f(x) = ⊕_S c_S · ∏_{i∈S} x_i   with   c_S = ⊕_{T⊆S} f(T),
    /// and each active monomial c_S becomes a Z, CZ (Controlled-Z), CCZ (doubly-Controlled-Z),
    /// etc. gate on the qubits in S.
    /// For n≤2 qubits: all Clifford (0 CCZ). For n=3: at most 1 CCZ. For n=4: at most 5 CCZ.
    /// This is more efficient than the Select-based approach for small registers.
    ///
    /// # Input
    /// ## phases
    /// Bool[2^n]: phases[i] = true if |i⟩ gets Z flip.
    /// ## register
    /// Qubit[n]: the register in LE (little-endian) order
    /// (register[0] = LSB (Least Significant Bit)).
    operation ApplyPhasePolynomial(phases : Bool[], register : Qubit[]) : Unit {
        let n = Length(register);
        let dim = Length(phases);

        // Möbius transform (in-place butterfly): coeffs[S] = true if monomial ∏_{i∈S} x_i is
        // active, with the subset S encoded as a bitmask.
        mutable coeffs = phases;
        mutable step = 1;
        while step < dim {
            for j in 0..dim - 1 {
                if (j / step) % 2 == 1 {
                    set coeffs w/= j <- coeffs[j] != coeffs[j - step];
                }
            }
            set step = step * 2;
        }

        // Apply multi-controlled Z for each nonzero coefficient of degree ≥ 1
        for s in 1..dim - 1 {
            if coeffs[s] {
                // Extract qubit indices where s has bit 1
                mutable qubits : Qubit[] = [];
                for bit in 0..n - 1 {
                    if (s >>> bit) &&& 1 == 1 {
                        set qubits += [register[bit]];
                    }
                }
                // Apply multi-controlled Z: degree 1 = Z, degree 2 = CZ (Controlled-Z),
                // degree 3 = CCZ (doubly-Controlled-Z), etc.
                if Length(qubits) == 1 {
                    Z(qubits[0]);
                } elif Length(qubits) == 2 {
                    Controlled Z([qubits[0]], qubits[1]);
                } else {
                    // degree ≥ 3: use Controlled Z
                    Controlled Z(qubits[0..Length(qubits) - 2], qubits[Length(qubits) - 1]);
                }
            }
        }
    }

    // =============================================================================
    // Full unitary via Givens decomposition
    // =============================================================================

    /// # Summary
    /// Applies a real unitary matrix via its Givens rotation decomposition, controlled on
    /// every qubit of `controls`.
    ///
    /// # Description
    /// A real unitary U is decomposed as: U = D · R_{k-1} · ... · R_1 · R_0
    /// where each R_i is a Givens rotation layer and D = diag(±1) is a phase correction.
    /// A block-diagonal unitary block_diag(U_0, U_1, ...) is applied by passing the
    /// block-selecting qubits as the most significant qubits of `target`.
    ///
    /// The layers are applied in order (R_0 first), followed by the phase correction.
    /// Each layer uses QROAM (Quantum Read-Only Access Memory) loaded angles and phase
    /// gradient rotations:
    ///   1. SelectSwap loads the quantized angle for the current address state
    ///   2. Ry via phase gradient applies the rotation to the active qubit
    ///   3. Adjoint SelectSwap uncomputes the angle register
    /// The phase correction is applied as a Reed-Muller phase polynomial.
    ///
    /// With controls, only the angle lookups and the phase correction are controlled: when a
    /// control is |0⟩ the angle register stays zero, Ry(0) = I, and the unconditional
    /// shift/unshift pair of each shifted layer cancels.
    ///
    /// # References
    /// - Berry et al. (PRX Quantum 6, 020327): https://doi.org/10.1103/PRXQuantum.6.020327
    /// - Clements et al. (arXiv:1603.08788): https://arxiv.org/abs/1603.08788
    ///
    /// # Input
    /// ## givens
    /// Quantized Givens layers and sign corrections of the unitary. Empty `phases` apply no
    /// signs.
    /// ## controls
    /// Control qubits; an empty array applies the unitary unconditionally.
    /// ## target
    /// Target register in MSB (Most Significant Bit)-first format
    /// (target[0] = MSB, target[n-1] = LSB (Least Significant Bit)).
    /// State value = target[0]*2^(n-1) + ... + target[n-1]*2^0.
    /// ## phaseGradient
    /// Phase gradient register.
    /// ## angleReg
    /// Clean register that receives each loaded angle.
    operation ApplyRealUnitaryViaGivens(
        givens : QuantizedGivensDecomposition,
        controls : Qubit[],
        target : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit {
        let n = Length(target);
        let layers = givens.layers;
        let layerIsShifted = givens.layerShifted;
        let phaseFlips = givens.phases;
        Fact(Length(layerIsShifted) == Length(layers), "Givens data needs one shift flag per layer.");
        Fact(IsEmpty(phaseFlips) or Length(phaseFlips) == 1 <<< n, "Givens phase data needs one entry per basis state of the target register.");
        // Active qubit = LSB of state = target[n-1].
        // Address = higher bits = target[0..n-2], reversed for Select (LSB-first).
        let activeQubit = target[n - 1];
        let address = Reversed(target[0..n - 2]);

        // All layers have the same circuit structure (same register sizes), only data differs.
        // Cache by shifted/non-shifted variant to avoid re-tracing ~1000 identical layers.
        for i in 0..Length(layers) - 1 {
            let variant = n * 2 + (if layerIsShifted[i] { 1 } else { 0 });
            if BeginEstimateCaching($"GivensLayer{Length(controls)}", variant) {
                // Reversed(target) gives LE (little-endian) view: index 0 = LSB of state value
                if layerIsShifted[i] {
                    AddConstant(-1, Reversed(target));
                }
                if IsEmpty(controls) {
                    ApplyMultiplexedRy(layers[i], address, activeQubit, phaseGradient, angleReg);
                } else {
                    ApplyControlledMultiplexedRy(layers[i], address, controls, activeQubit, phaseGradient, angleReg);
                }
                if layerIsShifted[i] {
                    AddConstant(1, Reversed(target));
                }
                EndEstimateCaching();
            }
        }

        // Phase correction D = diag(±1) on the LE register Reversed(target) + controls. The
        // controls are its most significant qubits, so only the all-ones control block flips.
        if Any(flip -> flip, phaseFlips) {
            let uncontrolledFlips = Repeated(false, ((1 <<< Length(controls)) - 1) * Length(phaseFlips));
            ApplyPhasePolynomial(uncontrolledFlips + phaseFlips, Reversed(target) + controls);
        }
    }
}
