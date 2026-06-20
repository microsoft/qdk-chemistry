// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// --------------------------------------------------------------------------------------------
// Based on code originally published by Felix Rupprecht (DLR) on Zenodo:
//   https://zenodo.org/records/15587498
// Rewritten and adapted for integration into the QDK Chemistry library.
// --------------------------------------------------------------------------------------------

/// # Summary
/// Gate-based unitary synthesis via Givens rotation layers.
///
/// # Description
/// Applies arbitrary unitaries using QROAM-loaded angles + phase gradient rotations.
/// Each unitary is pre-decomposed (classically) into layers of 2×2 Givens rotations.
/// Each layer is applied as:
///   1. QROAM loads the quantized angle for the current address state
///   2. Ry via phase gradient applies the rotation to the active qubit
///   3. Adjoint QROAM uncomputes the angle register
///
/// This replaces the simulation-only `ApplyUnitary` with actual gate decompositions.
///
/// # References
/// - Berry et al. (arXiv:2409.11748): MPS state preparation
/// - Clements et al. (arXiv:1603.08788): Optimal multiport interferometer design

import Std.Math.*;
import Std.Convert.*;
import Std.Arrays.*;
import Std.Canon.*;
import Std.Diagnostics.*;
import Std.Arithmetic.RippleCarryCGIncByLE;
import Std.TableLookup.Select;
import PhaseGradient.RyViaPhaseGradient;
import PhaseGradient.RzViaPhaseGradient;

export ApplyGivensLayer, ApplyRealUnitaryViaGivens, ApplyControlledRealUnitaryViaGivens, ApplyBlockDiagUnitaryViaGivens, IncrementByOne, QuantizeGivensAngles, QuantizeRyAngles, PhaseFlipsAsSelectData, ApplyPhasePolynomial, ComputePhasePolynomialCoeffs;

// =============================================================================
// Increment/Decrement helper
// =============================================================================

/// # Summary
/// Increments a little-endian register by 1 (mod 2^n).
///
/// # Input
/// ## target
/// Register in little-endian format (target[0] = LSB).
operation IncrementByOne(target : Qubit[]) : Unit is Adj + Ctl {
    let n = Length(target);
    if n == 1 {
        X(target[0]);
    } elif n == 2 {
        // For 2-bit: |ab⟩ → |(a⊕b)(b⊕1)⟩. All Cliffords (0 Toffoli).
        CNOT(target[0], target[1]);
        X(target[0]);
    } else {
        use increment = Qubit[n];
        within {
            X(increment[0]);
        } apply {
            RippleCarryCGIncByLE(increment, target);
        }
    }
}

// =============================================================================
// Angle quantization helpers (moved from Python preprocessing)
// =============================================================================

/// # Summary
/// Quantize Givens rotation angles to Bool[][] format for Select/QROAM.
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
/// ## bRot
/// Phase gradient precision bits.
///
/// # Output
/// Bool[numAddresses][bRot]: quantized angle data for Select.
function QuantizeGivensAngles(angles : Double[], numAddresses : Int, bRot : Int) : Bool[][] {
    let scale = 1 <<< bRot;
    let scaleF = IntAsDouble(scale);
    mutable data : Bool[][] = [];
    for k in 0..numAddresses - 1 {
        let angle = k < Length(angles) ? angles[k] | 0.0;
        mutable xInt = Round(scaleF * angle / (2.0 * PI()));
        set xInt = ((xInt % scale) + scale) % scale;
        set data += [IntAsBoolArray(xInt, bRot)];
    }
    return data;
}

/// # Summary
/// Quantize standard Ry angles to Bool[][] format for Select/QROAM.
///
/// # Description
/// For a standard Ry(α) rotation: RyViaPhaseGradient applies Ry(4π·x/2^b).
/// We need Ry(α), so x = α·2^b/(4π).
///
/// # Input
/// ## angles
/// Double[dim]: standard Ry rotation angles in radians.
/// ## bRot
/// Phase gradient precision bits.
///
/// # Output
/// Bool[dim][bRot]: quantized angle data for Select.
function QuantizeRyAngles(angles : Double[], bRot : Int) : Bool[][] {
    let scale = 1 <<< bRot;
    let scaleF = IntAsDouble(scale);
    mutable data : Bool[][] = [];
    for k in 0..Length(angles) - 1 {
        mutable xInt = Round(scaleF * angles[k] / (4.0 * PI()));
        set xInt = ((xInt % scale) + scale) % scale;
        set data += [IntAsBoolArray(xInt, bRot)];
    }
    return data;
}

/// # Summary
/// Convert a Bool[] phase flip array to Bool[][1] format for Select.
///
/// # Input
/// ## phases
/// Bool[dim]: true if state |i⟩ needs a Z flip.
///
/// # Output
/// Bool[dim][1]: Select-compatible format.
function PhaseFlipsAsSelectData(phases : Bool[]) : Bool[][] {
    mutable data : Bool[][] = [];
    for p in phases {
        set data += [[p]];
    }
    return data;
}

// =============================================================================
// Phase polynomial correction (Reed-Muller decomposition)
// =============================================================================

/// # Summary
/// Computes Reed-Muller (Möbius) coefficients for a phase diagonal.
///
/// # Description
/// For an n-qubit diagonal D = diag((-1)^{f(x)}) where f: {0,1}^n → {0,1},
/// computes the multilinear polynomial representation:
///   f(x) = ⊕_S c_S · ∏_{i∈S} x_i
/// where c_S = ⊕_{T⊆S} f(T) (Möbius inversion over GF(2)).
///
/// # Input
/// ## phases
/// Bool[2^n]: phases[i] = true if state |i⟩ gets (-1) phase.
///
/// # Output
/// Bool[2^n]: coefficients. coeffs[S] = true if monomial ∏_{i∈S} x_i is active.
/// Index S encodes the subset as a bitmask.
function ComputePhasePolynomialCoeffs(phases : Bool[]) : Bool[] {
    let dim = Length(phases);
    mutable coeffs = phases;
    // Möbius transform (in-place butterfly)
    mutable step = 1;
    while step < dim {
        for j in 0..dim - 1 {
            if (j / step) % 2 == 1 {
                set coeffs w/= j <- coeffs[j] != coeffs[j - step];
            }
        }
        set step = step * 2;
    }
    return coeffs;
}

/// # Summary
/// Applies a phase correction D = diag(±1) using the Reed-Muller polynomial decomposition.
///
/// # Description
/// Decomposes the diagonal into Z, CZ, CCZ, etc. gates via the Möbius transform.
/// For n≤2 qubits: all Clifford (0 CCZ). For n=3: at most 1 CCZ. For n=4: at most 5 CCZ.
/// This is more efficient than the Select-based approach for small registers.
///
/// # Input
/// ## phases
/// Bool[2^n]: phases[i] = true if |i⟩ gets Z flip.
/// ## register
/// Qubit[n]: the register in LE order (register[0] = LSB).
operation ApplyPhasePolynomial(phases : Bool[], register : Qubit[]) : Unit {
    let n = Length(register);
    let dim = Length(phases);
    let coeffs = ComputePhasePolynomialCoeffs(phases);

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
            // Apply multi-controlled Z: degree 1 = Z, degree 2 = CZ, degree 3 = CCZ, etc.
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
// Single Givens rotation layer
// =============================================================================

/// # Summary
/// Applies one Givens rotation layer to a target register.
///
/// # Description
/// A Givens layer applies dim/2 independent Ry rotations on pairs of adjacent
/// basis states. For a non-shifted layer (Berry eq. 23):
///   block_diag(Ry(θ₀), Ry(θ₁), ..., I)
/// Acting on pairs (0,1), (2,3), (4,5), ...
///
/// For a shifted layer (Berry eq. 24):
///   block_diag(I₁, Ry(θ₀), Ry(θ₁), ..., I)
/// Acting on pairs (1,2), (3,4), (5,6), ...
///
/// Implementation: the LSB (target[0]) selects between the two states in each pair.
/// The remaining bits (target[1..n-1]) serve as the address for QROAM angle lookup.
/// For shifted layers, we decrement the register by 1 first to align pairs with LSB.
///
/// # Input
/// ## angleData
/// Bool[numAngles][bRot]: pre-quantized rotation angles. angleData[k] is the angle
/// for the k-th pair, encoded as a b_rot-bit integer x such that θ = 4π·x/2^bRot.
/// ## isShifted
/// If true, applies the shifted version (Berry eq. 24).
/// ## target
/// Target register in MSB-first format (target[0] = MSB, target[n-1] = LSB).
/// State value = target[0]*2^(n-1) + ... + target[n-1]*2^0.
/// ## phaseGradient
/// Phase gradient ancilla register.
operation ApplyGivensLayer(
    angleData : Bool[][],
    isShifted : Bool,
    target : Qubit[],
    phaseGradient : Qubit[],
    angleReg : Qubit[]
) : Unit {
    let n = Length(target);

    // Reversed(target) gives LE view: index 0 = LSB of state value
    if isShifted {
        Adjoint IncrementByOne(Reversed(target));
    }

    // Active qubit = LSB of state = target[n-1].
    // Address = higher bits = target[0..n-2], reversed for Select (LSB-first).
    let activeQubit = target[n - 1];
    let address = target[0..n - 2];

    // QROAM load → Ry → uncompute (deterministic via within/apply)
    within {
        Select(angleData, Reversed(address), angleReg);
    } apply {
        RyViaPhaseGradient(activeQubit, angleReg, phaseGradient);
    }

    if isShifted {
        IncrementByOne(Reversed(target));
    }
}

/// # Summary
/// Same as ApplyGivensLayer but controlled on an additional qubit.
///
/// # Description
/// When control = |1⟩, applies the Givens rotation layer.
/// When control = |0⟩, does nothing (identity).
///
/// Optimization: The control qubit is folded into the QROAM address as MSB.
/// The data table is extended with zeros for the ctrl=0 case, so:
///   - ctrl=0: QROAM loads zeros → Ry(0) = I (identity)
///   - ctrl=1: QROAM loads actual angles → rotation applied
///
/// The IncrementByOne shift is applied UNCONDITIONALLY (not controlled).
/// This is correct because when ctrl=0, Ry(0)=I makes the shift/unshift pair
/// cancel out, producing identity on the target register.
///
/// # Input
/// ## angleData
/// Bool[numAngles][bRot]: pre-quantized rotation angles.
/// ## isShifted
/// If true, applies the shifted version.
/// ## target
/// Target register.
/// ## phaseGradient
/// Phase gradient register.
/// ## control
/// Control qubit.
operation ApplyControlledGivensLayer(
    angleData : Bool[][],
    isShifted : Bool,
    target : Qubit[],
    phaseGradient : Qubit[],
    control : Qubit,
    angleReg : Qubit[]
) : Unit {
    let n = Length(target);
    let numAngles = Length(angleData);
    let bRot = Length(angleReg);

    // Build extended data: zeros (ctrl=0) ++ angleData (ctrl=1)
    // Control qubit is MSB of the address register.
    let zeros = Repeated(Repeated(false, bRot), numAngles);
    let extendedData = zeros + angleData;

    // Shifts are UNCONDITIONAL — saves Controlled IncrementByOne cost.
    // When ctrl=0, QROAM loads zeros so Ry(0)=I, making shift/unshift cancel.
    if isShifted {
        Adjoint IncrementByOne(Reversed(target));
    }

    let activeQubit = target[n - 1];
    let address = target[0..n - 2];

    within {
        Select(extendedData, Reversed(address) + [control], angleReg);
    } apply {
        RyViaPhaseGradient(activeQubit, angleReg, phaseGradient);
    }

    if isShifted {
        IncrementByOne(Reversed(target));
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
///
/// The layers are applied in order (R_0 first), followed by the phase correction.
/// The phase correction is implemented by loading sign bits via QROAM and applying Z.
///
/// # Input
/// ## layerAngleData
/// Bool[numLayers][numAngles][bRot]: angle data for each Givens layer.
/// ## layerIsShifted
/// Bool[numLayers]: whether each layer is shifted.
/// ## phaseFlipData
/// Bool[dim][1]: phase correction. phaseFlipData[i] = [true] if state |i⟩ gets Z.
/// Empty array means no phase correction needed.
/// ## target
/// Target register.
/// ## phaseGradient
/// Phase gradient register.
operation ApplyRealUnitaryViaGivens(
    layerAngleData : Bool[][][],
    layerIsShifted : Bool[],
    phaseFlipData : Bool[][],
    target : Qubit[],
    phaseGradient : Qubit[],
    angleReg : Qubit[]
) : Unit {
    let numLayers = Length(layerAngleData);

    // Apply Givens layers in order
    for i in 0..numLayers - 1 {
        ApplyGivensLayer(layerAngleData[i], layerIsShifted[i], target, phaseGradient, angleReg);
    }

    // Phase correction: D = diag(±1) via Reed-Muller polynomial
    if Length(phaseFlipData) > 0 {
        mutable hasFlips = false;
        for row in phaseFlipData {
            if Length(row) > 0 and row[0] {
                set hasFlips = true;
            }
        }
        if hasFlips {
            mutable phases : Bool[] = [];
            for row in phaseFlipData {
                set phases += [Length(row) > 0 and row[0]];
            }
            ApplyPhasePolynomial(phases, Reversed(target));
        }
    }
}

/// # Summary
/// Applies a controlled real unitary via Givens decomposition.
///
/// # Description
/// When control = |1⟩, applies the unitary. When control = |0⟩, identity.
/// Each Givens layer is controlled, as is the phase correction.
///
/// # Input
/// ## layerAngleData
/// Bool[numLayers][numAngles][bRot]: angle data for each Givens layer.
/// ## layerIsShifted
/// Bool[numLayers]: whether each layer is shifted.
/// ## phaseFlipData
/// Bool[dim][1]: phase correction data. Empty array means no correction.
/// ## target
/// Target register.
/// ## phaseGradient
/// Phase gradient register.
/// ## control
/// Control qubit.
operation ApplyControlledRealUnitaryViaGivens(
    layerAngleData : Bool[][][],
    layerIsShifted : Bool[],
    phaseFlipData : Bool[][],
    target : Qubit[],
    phaseGradient : Qubit[],
    control : Qubit,
    angleReg : Qubit[]
) : Unit {
    let numLayers = Length(layerAngleData);

    for i in 0..numLayers - 1 {
        ApplyControlledGivensLayer(
            layerAngleData[i], layerIsShifted[i], target, phaseGradient, control, angleReg
        );
    }

    // Controlled phase correction: when ctrl=1, apply D on target.
    // This is a diagonal on (ctrl, target) register = [control] + Reversed(target).
    // Extended phases: ctrl=0 → no flip, ctrl=1 → phaseFlipData.
    if Length(phaseFlipData) > 0 {
        mutable hasFlips = false;
        for row in phaseFlipData {
            if Length(row) > 0 and row[0] {
                set hasFlips = true;
            }
        }
        if hasFlips {
            // Build extended phase array: [zeros for ctrl=0] ++ [phases for ctrl=1]
            let dim = Length(phaseFlipData);
            mutable extendedPhases : Bool[] = Repeated(false, dim);
            for row in phaseFlipData {
                set extendedPhases += [Length(row) > 0 and row[0]];
            }
            // Register order: Reversed(target) = LE of target state; control = MSB of extended register
            ApplyPhasePolynomial(extendedPhases, Reversed(target) + [control]);
        }
    }
}

/// # Summary
/// Applies a block-diagonal unitary via Givens decomposition.
///
/// # Description
/// Applies block_diag(U_0, U_1, ..., U_{2^s-1}) where s = Length(subspace).
/// The block-diagonal is decomposed jointly into Givens layers that act on the
/// combined (subspace ⊕ target) register.
///
/// # Input
/// ## layerAngleData
/// Bool[numLayers][numAngles][bRot]: angle data for the joint Givens decomposition.
/// The address space covers both subspace and target upper bits.
/// ## layerIsShifted
/// Bool[numLayers]: whether each layer is shifted.
/// ## phaseFlipData
/// Bool[totalDim][1]: phase correction for the full (subspace ⊕ target) space.
/// ## target
/// Target register (ancilla qubits).
/// ## subspace
/// Subspace selection register (site qubits).
/// ## phaseGradient
/// Phase gradient register.
operation ApplyBlockDiagUnitaryViaGivens(
    layerAngleData : Bool[][][],
    layerIsShifted : Bool[],
    phaseFlipData : Bool[][],
    target : Qubit[],
    subspace : Qubit[],
    phaseGradient : Qubit[],
    angleReg : Qubit[]
) : Unit {
    // The joint register: subspace (MSB) ++ target forms the full state space.
    // This matches ApplyUnitary(M, Reversed(target + subspace)) convention when
    // the caller passes Reversed(ancilla) as target and [q1,q0] as subspace.
    let jointReg = subspace + target;
    let numLayers = Length(layerAngleData);

    for i in 0..numLayers - 1 {
        ApplyGivensLayer(layerAngleData[i], layerIsShifted[i], jointReg, phaseGradient, angleReg);
    }

    // Phase correction on the joint register via Reed-Muller polynomial
    if Length(phaseFlipData) > 0 {
        mutable hasFlips = false;
        for row in phaseFlipData {
            if Length(row) > 0 and row[0] {
                set hasFlips = true;
            }
        }
        if hasFlips {
            mutable phases : Bool[] = [];
            for row in phaseFlipData {
                set phases += [Length(row) > 0 and row[0]];
            }
            ApplyPhasePolynomial(phases, Reversed(jointReg));
        }
    }
}
