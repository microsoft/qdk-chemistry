// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

import Std.Math.*;
import Std.Convert.*;
import Std.Arrays.*;
import Std.Canon.*;
import Std.Diagnostics.*;
import Std.ResourceEstimation.*;
import QDKChemistry.Utils.Arithmetic.AddConstant;
import QDKChemistry.Utils.SelectSwap.SelectSwap;
import QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;

export GivensDecomposition, QuantizedGivensDecomposition, QuantizeGivensDecomposition, ApplyRealUnitaryViaGivens, ApplyControlledRealUnitaryViaGivens, QuantizeGivensAngles, QuantizeRyAngles, ApplyMultiplexedRy, ApplyControlledMultiplexedRy;

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
/// Quantizes the angles of every layer of a `GivensDecomposition` with `QuantizeGivensAngles`.
function QuantizeGivensDecomposition(
    givens : GivensDecomposition,
    numAddresses : Int,
    rotationBits : Int
) : QuantizedGivensDecomposition {
    new QuantizedGivensDecomposition {
        layers = Mapped(angles -> QuantizeGivensAngles(angles, numAddresses, rotationBits), givens.layerAngles),
        layerShifted = givens.layerShifted,
        phases = givens.phases,
    }
}

// =============================================================================
// Multiplexed Ry rotations (QROM-loaded angles + phase gradient)
// =============================================================================

/// # Summary
/// Applies Ry(4π·data[a]/2^b) to `activeQubit` for each address state |a⟩.
///
/// # Description
/// Loads the addressed angle with `SelectSwap`, rotates via the phase gradient register,
/// and unloads the angle, matching the pattern used by `QROMStatePrepare`.
///
/// # Input
/// ## data
/// Bool[N][b]: quantized rotation angles.
/// ## address
/// Little-endian address register with at least ⌈log₂ N⌉ qubits.
/// ## activeQubit
/// Rotation target.
/// ## phaseGradient
/// Phase gradient register (b qubits), pre-initialized.
/// ## angleReg
/// Clean b-qubit register that receives the loaded angle.
operation ApplyMultiplexedRy(
    data : Bool[][],
    address : Qubit[],
    activeQubit : Qubit,
    phaseGradient : Qubit[],
    angleReg : Qubit[]
) : Unit {
    within {
        SelectSwap(-1, data, address, angleReg);
    } apply {
        RyViaPhaseGradient(activeQubit, angleReg, phaseGradient);
    }
}

/// # Summary
/// Same as `ApplyMultiplexedRy`, but the angle is loaded only when `control` is |1⟩.
///
/// # Description
/// When `control` is |0⟩ the angle register stays zero, so the rotation is the identity.
operation ApplyControlledMultiplexedRy(
    data : Bool[][],
    address : Qubit[],
    control : Qubit,
    activeQubit : Qubit,
    phaseGradient : Qubit[],
    angleReg : Qubit[]
) : Unit {
    within {
        Controlled SelectSwap([control], (-1, data, address, angleReg));
    } apply {
        RyViaPhaseGradient(activeQubit, angleReg, phaseGradient);
    }
}

// =============================================================================
// Angle quantization helpers (moved from Python preprocessing)
// =============================================================================

/// # Summary
/// Quantize Givens rotation angles to Bool[][] format for Select/QROAM
/// (Quantum Read-Only Access Memory).
///
/// # Description
/// For a Givens rotation with angle θ (where the 2×2 matrix is [[cos θ, -sin θ], [sin θ, cos θ]]):
/// RyViaPhaseGradient applies Ry(4π·x/2^b). We need Ry(2θ), so x = θ·2^b/(2π).
///
/// # Input
/// ## angles
/// Double[numAngles]: raw Givens rotation angles in radians.
/// ## numAddresses
/// The padded address space dimension (= dim/2 for a dim-qubit register).
/// ## rotationBits
/// Phase gradient precision bits.
///
/// # Output
/// Bool[numAddresses][rotationBits]: quantized angle data for Select.
function QuantizeGivensAngles(angles : Double[], numAddresses : Int, rotationBits : Int) : Bool[][] {
    let scale = 1 <<< rotationBits;
    let scaleF = IntAsDouble(scale);
    mutable data : Bool[][] = [];
    for k in 0..numAddresses - 1 {
        let angle = k < Length(angles) ? angles[k] | 0.0;
        mutable xInt = Round(scaleF * angle / (2.0 * PI()));
        set xInt = ((xInt % scale) + scale) % scale;
        set data += [IntAsBoolArray(xInt, rotationBits)];
    }
    return data;
}

/// # Summary
/// Quantize standard Ry angles to Bool[][] format for Select/QROAM
/// (Quantum Read-Only Access Memory).
///
/// # Description
/// For a standard Ry(α) rotation: RyViaPhaseGradient applies Ry(4π·x/2^b).
/// We need Ry(α), so x = α·2^b/(4π).
///
/// # Input
/// ## angles
/// Double[dim]: standard Ry rotation angles in radians.
/// ## rotationBits
/// Phase gradient precision bits.
///
/// # Output
/// Bool[dim][rotationBits]: quantized angle data for Select.
function QuantizeRyAngles(angles : Double[], rotationBits : Int) : Bool[][] {
    let scale = 1 <<< rotationBits;
    let scaleF = IntAsDouble(scale);
    mutable data : Bool[][] = [];
    for k in 0..Length(angles) - 1 {
        mutable xInt = Round(scaleF * angles[k] / (4.0 * PI()));
        set xInt = ((xInt % scale) + scale) % scale;
        set data += [IntAsBoolArray(xInt, rotationBits)];
    }
    return data;
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
/// Applies a real unitary matrix via its Givens rotation decomposition.
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
/// # References
/// - Berry et al. (PRX Quantum 6, 020327): https://doi.org/10.1103/PRXQuantum.6.020327
/// - Clements et al. (arXiv:1603.08788): https://arxiv.org/abs/1603.08788
///
/// # Input
/// ## givens
/// Quantized Givens layers and sign corrections of the unitary. Empty `phases` apply no
/// signs.
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
        if BeginEstimateCaching("GivensLayer", variant) {
            // Reversed(target) gives LE (little-endian) view: index 0 = LSB of state value
            if layerIsShifted[i] {
                AddConstant(-1, Reversed(target));
            }
            ApplyMultiplexedRy(layers[i], address, activeQubit, phaseGradient, angleReg);
            if layerIsShifted[i] {
                AddConstant(1, Reversed(target));
            }
            EndEstimateCaching();
        }
    }

    // Phase correction: D = diag(±1) via Reed-Muller polynomial
    if Any(flip -> flip, phaseFlips) {
        ApplyPhasePolynomial(phaseFlips, Reversed(target));
    }
}

/// # Summary
/// Applies a controlled real unitary via Givens decomposition.
///
/// # Description
/// When control = |1⟩, applies the unitary. When control = |0⟩, identity.
/// The angle lookup of each Givens layer is controlled, so when control = |0⟩ the angle
/// register stays zero and Ry(0) = I. The shifts of shifted layers are therefore applied
/// unconditionally: when control = |0⟩ the shift/unshift pair cancels.
/// The phase correction is controlled as well.
///
/// # Input
/// ## givens
/// Quantized Givens layers and sign corrections of the unitary. Empty `phases` apply no
/// signs.
/// ## target
/// Target register.
/// ## phaseGradient
/// Phase gradient register.
/// ## control
/// Control qubit.
/// ## angleReg
/// Clean register that receives each loaded angle.
operation ApplyControlledRealUnitaryViaGivens(
    givens : QuantizedGivensDecomposition,
    target : Qubit[],
    phaseGradient : Qubit[],
    control : Qubit,
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

    for i in 0..Length(layers) - 1 {
        let variant = n * 2 + (if layerIsShifted[i] { 1 } else { 0 });
        if BeginEstimateCaching("ControlledGivensLayer", variant) {
            if layerIsShifted[i] {
                AddConstant(-1, Reversed(target));
            }
            ApplyControlledMultiplexedRy(layers[i], address, control, activeQubit, phaseGradient, angleReg);
            if layerIsShifted[i] {
                AddConstant(1, Reversed(target));
            }
            EndEstimateCaching();
        }
    }

    // Controlled phase correction: a diagonal on Reversed(target) + [control], where the
    // control is the MSB. Extended phases: ctrl=0 → no flip, ctrl=1 → phaseFlips.
    if Any(flip -> flip, phaseFlips) {
        ApplyPhasePolynomial(Repeated(false, Length(phaseFlips)) + phaseFlips, Reversed(target) + [control]);
    }
}
