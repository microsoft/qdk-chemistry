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

    /// Givens representation U = D · L_{m-1} ⋯ L_0 of a real orthogonal matrix, mirroring the
    /// C++ `qdk::chemistry::utils::detail::GivensDecomposition`.
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

    /// Applies D = diag((-1)^{f(x)}) on a little-endian register, with `phases[x]` = f(x).
    ///
    /// The Möbius transform over GF(2) gives f(x) = ⊕_S c_S ∏_{i∈S} x_i, and each nonzero c_S
    /// becomes a Z, CZ, CCZ, … gate on the qubits in S.
    operation ApplyPhasePolynomial(phases : Bool[], register : Qubit[]) : Unit {
        let n = Length(register);
        let dim = Length(phases);

        // In-place Möbius butterfly; bitmask s encodes the subset S.
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

        for s in 1..dim - 1 {
            if coeffs[s] {
                mutable qubits : Qubit[] = [];
                for bit in 0..n - 1 {
                    if (s >>> bit) &&& 1 == 1 {
                        set qubits += [register[bit]];
                    }
                }
                if Length(qubits) == 1 {
                    Z(qubits[0]);
                } elif Length(qubits) == 2 {
                    Controlled Z([qubits[0]], qubits[1]);
                } else {
                    Controlled Z(qubits[0..Length(qubits) - 2], qubits[Length(qubits) - 1]);
                }
            }
        }
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
    ) : Unit {
        let n = Length(target);
        let layers = givens.layers;
        let layerIsShifted = givens.layerShifted;
        let phaseFlips = givens.phases;
        Fact(Length(layerIsShifted) == Length(layers), "Givens data needs one shift flag per layer.");
        Fact(IsEmpty(phaseFlips) or Length(phaseFlips) == 1 <<< n, "Givens phase data needs one entry per basis state of the target register.");
        let activeQubit = target[n - 1];
        let address = Reversed(target[0..n - 2]);

        // Layers differ only in data, so cache the shifted and unshifted circuits.
        for i in 0..Length(layers) - 1 {
            let variant = n * 2 + (if layerIsShifted[i] { 1 } else { 0 });
            if BeginEstimateCaching($"GivensLayer{Length(controls)}", variant) {
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

        // The controls are the most significant qubits of the register D acts on, so only the
        // all-ones control block flips.
        if Any(flip -> flip, phaseFlips) {
            let uncontrolledFlips = Repeated(false, ((1 <<< Length(controls)) - 1) * Length(phaseFlips));
            ApplyPhasePolynomial(uncontrolledFlips + phaseFlips, Reversed(target) + controls);
        }
    }
}
