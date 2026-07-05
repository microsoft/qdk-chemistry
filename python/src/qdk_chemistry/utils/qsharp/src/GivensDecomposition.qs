// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

import Std.Math.*;
import Std.Convert.*;
import Std.Arrays.*;
import Std.Canon.*;
import Std.Diagnostics.*;
import Std.ResourceEstimation.*;
import QDKChemistry.Utils.SelectSwap.ControlledQroamCleanRotation;
import QDKChemistry.Utils.SelectSwap.QroamCleanRotation;
import QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;

export ApplyGivensLayer, ApplyRealUnitaryViaGivens, ApplyControlledRealUnitaryViaGivens, ApplyBlockDiagUnitaryViaGivens, IncrementByOne, QuantizeGivensAngles, QuantizeRyAngles, PhaseFlipsAsSelectData, ApplyPhasePolynomial, ApplyRealUnitaryViaGivensFast, ApplyControlledRealUnitaryViaGivensFast, ApplyBlockDiagUnitaryViaGivensFast;

/// # Summary
/// Increments a little-endian register by 1 (mod 2^n).
///
/// # Input
/// ## target
/// Register in little-endian (LE) format (target[0] = LSB (Least Significant Bit)).
operation IncrementByOne(target : Qubit[]) : Unit {
    AddConstant(1, target);
}

// =============================================================================
// Classical constant addition (Sanders et al. Fig. 18)
// =============================================================================

/// # Summary
/// Adds a classical constant `c` to a little-endian quantum register in place.
///
/// # Description
/// Implements |x⟩ → |x + c (mod 2^n)⟩ using the ripple-carry structure from
/// Sanders et al. (PRX Quantum 1, 020312, 2020), Fig. 18.
///
/// Resource cost:
///   - n − 2 AND gates  → 4(n − 2) T-gates
///   - n − 2 IAND gates → 0 T-gates (measurement-based)
///   - O(n) Clifford gates
///   - n − 1 ancilla qubits (borrowed, returned to |0⟩)
///
/// # Input
/// ## c
/// The classical integer to add.
/// ## target
/// Register in little-endian format (target[0] = LSB). Length n.
///
/// # References
/// - Sanders, Y.R., et al. "Compilation of Fault-Tolerant Quantum Heuristics
///   for Combinatorial Optimization." PRX Quantum 1, 020312 (2020).
operation AddConstant(c : Int, target : Qubit[]) : Unit {
    let n = Length(target);
    let cMod = ((c % (1 <<< n)) + (1 <<< n)) % (1 <<< n);
    let cBits = IntAsBoolArray(cMod, n); // LE: cBits[0] = LSB

    if n == 1 {
        if cBits[0] {
            X(target[0]);
        }
    } elif n == 2 {
        // 2-bit: no Toffolis needed. Use Clifford-only circuit.
        if cBits[1] {
            if cBits[0] {
                // +3 = -1 mod 4: decrement
                X(target[0]);
                CNOT(target[0], target[1]);
            } else {
                // +2: flip MSB
                X(target[1]);
            }
        } else {
            if cBits[0] {
                // +1: increment
                CNOT(target[0], target[1]);
                X(target[0]);
            }
            // +0: identity
        }
    } else {
        use ancillas = Qubit[n - 1];

        // --- Forward pass: compute carry bits ---
        if cBits[0] {
            CNOT(target[0], ancillas[0]);
        }

        for i in 1..n - 2 {
            let j = i - 1;
            CNOT(ancillas[j], target[i]);
            if cBits[i] {
                X(ancillas[j]);
            }
            AND(ancillas[j], target[i], ancillas[i]);
            if cBits[i] {
                X(ancillas[j]);
            }
            CNOT(ancillas[j], ancillas[i]);
        }

        // --- MSB: XOR final carry ---
        CNOT(ancillas[n - 2], target[n - 1]);

        // --- Reverse pass: uncompute ancillas ---
        for i in n - 2..-1..1 {
            let j = i - 1;
            CNOT(ancillas[j], ancillas[i]);
            if cBits[i] {
                X(ancillas[j]);
            }
            Adjoint AND(ancillas[j], target[i], ancillas[i]);
            if cBits[i] {
                X(ancillas[j]);
            }
        }

        if cBits[0] {
            CNOT(target[0], ancillas[0]);
        }

        // --- Final XOR: add constant bits ---
        for i in 0..n - 1 {
            if cBits[i] {
                X(target[i]);
            }
        }
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
/// Decomposes the diagonal into Z, CZ (Controlled-Z), CCZ (doubly-Controlled-Z),
/// etc. gates via the Möbius transform.
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
// Single Givens rotation layer
// =============================================================================

/// # Summary
/// Gate-based unitary synthesis via Givens rotation layers.
///
/// # Description
/// Applies arbitrary unitaries using QROAM (Quantum Read-Only Access Memory)
/// loaded angles + phase gradient rotations.
/// Each unitary is pre-decomposed (classically) into layers of 2×2 Givens rotations.
/// Each layer is applied as:
///   1. QROAM loads the quantized angle for the current address state
///   2. Ry via phase gradient applies the rotation to the active qubit
///   3. Adjoint QROAM uncomputes the angle register
///
/// # References
/// - Berry et al. (PRX Quantum 6, 020327): https://doi.org/10.1103/PRXQuantum.6.020327
/// - Clements et al. (arXiv:1603.08788): https://arxiv.org/abs/1603.08788
///
/// # Input
/// ## angleData
/// Bool[numAngles][rotationBits]: pre-quantized rotation angles. angleData[k] is the angle
/// for the k-th pair, encoded as a rotationBits-bit integer x such that θ = 4π·x/2^rotationBits.
/// ## isShifted
/// If true, applies the shifted version (Berry eq. 24).
/// ## target
/// Target register in MSB (Most Significant Bit)-first format
/// (target[0] = MSB, target[n-1] = LSB (Least Significant Bit)).
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

    // Reversed(target) gives LE (little-endian) view: index 0 = LSB of state value
    if isShifted {
        AddConstant(-1, Reversed(target));
    }

    // Active qubit = LSB of state = target[n-1].
    // Address = higher bits = target[0..n-2], reversed for Select (LSB-first).
    let activeQubit = target[n - 1];
    let address = target[0..n - 2];

    // QROAM-clean: forward-only Select+Swap + Ry + Unlookup for half the cost
    QroamCleanRotation(angleData, Reversed(address), activeQubit, phaseGradient);

    if isShifted {
        AddConstant(1, Reversed(target));
    }
}

/// # Summary
/// Same as ApplyGivensLayer but controlled on an additional qubit.
///
/// # Description
/// When control = |1⟩, applies the Givens rotation layer.
/// When control = |0⟩, does nothing (identity).
///
/// Uses QROAMClean pattern (Berry et al. arXiv:1902.02134):
///   1. Controlled Select loads N data entries (NOT 2N extended data).
///      Cost: N-1 AND gates (vs 2N-2 for data-doubling approach).
///   2. Ry rotation via phase gradient.
///   3. Measurement-based uncomputation (Unlookup) on 2N extended data
///      to correctly handle the ctrl=0 branch.
///
/// The IncrementByOne shift is applied UNCONDITIONALLY (not controlled).
/// This is correct because when ctrl=0, Select loads nothing, so
/// Ry(0)=I makes the shift/unshift pair cancel out.
///
/// # Input
/// ## angleData
/// Bool[numAngles][rotationBits]: pre-quantized rotation angles.
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
    let rotationBits = Length(angleReg);

    // Shifts are UNCONDITIONAL.
    // When ctrl=0, Select loads nothing so Ry(0)=I, making shift/unshift cancel.
    if isShifted {
        AddConstant(-1, Reversed(target));
    }

    let activeQubit = target[n - 1];
    let address = target[0..n - 2];

    // QROAMClean with SelectSwap blocking: uses forward-only Controlled Select
    // with blocking + measurement-based Unlookup for reduced T-count.
    ControlledQroamCleanRotation(angleData, Reversed(address), control, activeQubit, phaseGradient);

    if isShifted {
        AddConstant(1, Reversed(target));
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
/// Bool[numLayers][numAngles][rotationBits]: angle data for each Givens layer.
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
    // All layers have the same circuit structure (same register sizes), only data differs.
    // Cache by shifted/non-shifted variant to avoid re-tracing ~1000 identical layers.
    for i in 0..numLayers - 1 {
        let variant = Length(target) * 2 + (if layerIsShifted[i] { 1 } else { 0 });
        if BeginEstimateCaching("GivensLayer", variant) {
            ApplyGivensLayer(layerAngleData[i], layerIsShifted[i], target, phaseGradient, angleReg);
            EndEstimateCaching();
        }
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
/// Bool[numLayers][numAngles][rotationBits]: angle data for each Givens layer.
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
        let variant = Length(target) * 2 + (if layerIsShifted[i] { 1 } else { 0 });
        if BeginEstimateCaching("ControlledGivensLayer", variant) {
            ApplyControlledGivensLayer(
                layerAngleData[i],
                layerIsShifted[i],
                target,
                phaseGradient,
                control,
                angleReg
            );
            EndEstimateCaching();
        }
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
/// Bool[numLayers][numAngles][rotationBits]: angle data for the joint Givens decomposition.
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
        let variant = Length(jointReg) * 2 + (if layerIsShifted[i] { 1 } else { 0 });
        if BeginEstimateCaching("BlockDiagGivensLayer", variant) {
            ApplyGivensLayer(layerAngleData[i], layerIsShifted[i], jointReg, phaseGradient, angleReg);
            EndEstimateCaching();
        }
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

// =============================================================================
// Fast resource estimation variants using RepeatEstimates
// =============================================================================
// These accept only 2 representative layers (one non-shifted, one shifted) plus
// the total layer count. They call BeginEstimateCaching only twice and use
// RepeatEstimates to inform the resource estimator of the total repetitions.
// This eliminates both the O(dim^2) data serialization and the O(dim) loop overhead.

/// # Summary
/// Fast resource estimation variant of ApplyRealUnitaryViaGivens.
///
/// # Description
/// Accepts exactly 2 representative layer angle arrays (index 0 = non-shifted,
/// index 1 = shifted) and the total number of layers. Uses RepeatEstimates to
/// multiply the single-layer cost by the appropriate count.
operation ApplyRealUnitaryViaGivensFast(
    repLayerAngles : Double[][],
    numLayers : Int,
    phaseFlipData : Bool[][],
    numAddresses : Int,
    rotationBits : Int,
    cacheId : Int,
    target : Qubit[],
    phaseGradient : Qubit[],
    angleReg : Qubit[]
) : Unit {
    // Iterate over all layers using cache hits.
    // Only 2 representative layers are provided; the cache ensures that
    // each variant (shifted/non-shifted) is traced once and reused.
    for i in 0..numLayers - 1 {
        let shifted = i % 2 == 1;
        let variant = Length(target) * 2 + (if shifted { 1 } else { 0 });
        if BeginEstimateCaching($"GivensLayer_{cacheId}", variant) {
            let layerIdx = if shifted { 1 } else { 0 };
            let quantized = QuantizeGivensAngles(repLayerAngles[layerIdx], numAddresses, rotationBits);
            ApplyGivensLayer(quantized, shifted, target, phaseGradient, angleReg);
            EndEstimateCaching();
        }
    }

    // Phase correction
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
/// Fast resource estimation variant of ApplyControlledRealUnitaryViaGivens.
operation ApplyControlledRealUnitaryViaGivensFast(
    repLayerAngles : Double[][],
    numLayers : Int,
    phaseFlipData : Bool[][],
    numAddresses : Int,
    rotationBits : Int,
    cacheId : Int,
    target : Qubit[],
    phaseGradient : Qubit[],
    control : Qubit,
    angleReg : Qubit[]
) : Unit {
    for i in 0..numLayers - 1 {
        let shifted = i % 2 == 1;
        let variant = Length(target) * 2 + (if shifted { 1 } else { 0 });
        if BeginEstimateCaching($"ControlledGivensLayer_{cacheId}", variant) {
            let layerIdx = if shifted { 1 } else { 0 };
            let quantized = QuantizeGivensAngles(repLayerAngles[layerIdx], numAddresses, rotationBits);
            ApplyControlledGivensLayer(quantized, shifted, target, phaseGradient, control, angleReg);
            EndEstimateCaching();
        }
    }

    // Controlled phase correction
    if Length(phaseFlipData) > 0 {
        mutable hasFlips = false;
        for row in phaseFlipData {
            if Length(row) > 0 and row[0] {
                set hasFlips = true;
            }
        }
        if hasFlips {
            let dim = Length(phaseFlipData);
            mutable extendedPhases : Bool[] = Repeated(false, dim);
            for row in phaseFlipData {
                set extendedPhases += [Length(row) > 0 and row[0]];
            }
            ApplyPhasePolynomial(extendedPhases, Reversed(target) + [control]);
        }
    }
}

/// # Summary
/// Fast resource estimation variant of ApplyBlockDiagUnitaryViaGivens.
operation ApplyBlockDiagUnitaryViaGivensFast(
    repLayerAngles : Double[][],
    numLayers : Int,
    phaseFlipData : Bool[][],
    numAddresses : Int,
    rotationBits : Int,
    cacheId : Int,
    target : Qubit[],
    subspace : Qubit[],
    phaseGradient : Qubit[],
    angleReg : Qubit[]
) : Unit {
    let jointReg = subspace + target;
    for i in 0..numLayers - 1 {
        let shifted = i % 2 == 1;
        let variant = Length(jointReg) * 2 + (if shifted { 1 } else { 0 });
        if BeginEstimateCaching($"BlockDiagGivensLayer_{cacheId}", variant) {
            let layerIdx = if shifted { 1 } else { 0 };
            let quantized = QuantizeGivensAngles(repLayerAngles[layerIdx], numAddresses, rotationBits);
            ApplyGivensLayer(quantized, shifted, jointReg, phaseGradient, angleReg);
            EndEstimateCaching();
        }
    }

    // Phase correction on joint register
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
