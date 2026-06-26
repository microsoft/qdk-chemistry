// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

/// # Summary
/// Dense state preparation using QROM + phase-gradient Ry rotations.
///
/// # Description
/// Replaces the multiplexed-Pauli cascade in PreparePureStateD with:
///   QROM load → Ry via phase-gradient adder → X-measurement uncompute
///
/// # References
/// - Low, Kliuchnikov, Schaeffer (arXiv:1812.00954): QROM state prep
/// - Sanders et al. (arXiv:2007.07391): Phase gradient rotation
/// - Shende, Bullock, Markov (arXiv:quant-ph/0406176): SBM decomposition

import Std.Math.*;
import Std.Convert.*;
import Std.Arrays.*;
import Std.Canon.*;
import Std.Diagnostics.*;
import Std.TableLookup.Select;
import PhaseGradient.RyViaPhaseGradient;
import PhaseGradient.PreparePhaseGradientState;

export QroamStatePrep, ComputeSBMAngles, ComputeSignBits;

/// # Summary
/// Prepares a dense quantum state |ψ⟩ = Σ_j α_j |j⟩ / ‖α‖ using QROM-loaded rotations.
///
/// # Description
/// Given real coefficients α_j and a phase-gradient ancilla, prepares the target state
/// by iterating over qubits MSB to LSB. At each level i, the rotation angle θ(address)
/// is loaded via QROM conditioned on the previously-prepared qubits, applied as Ry via
/// the phase-gradient register, and uncomputed via X-basis measurement.
///
/// The phase-gradient register must be pre-initialized (use PreparePhaseGradientState)
/// and is returned unchanged (up to phases absorbed by MResetX).
///
/// Uses only Std.* (standard library) operations:
///   - Std.TableLookup.Select for QROM (unary iteration)
///   - Std.Arithmetic.RippleCarryCGIncByLE for phase-gradient rotation
///
/// # Input
/// ## coefficients
/// Real amplitudes of the target state. Length must be ≤ 2^Length(target).
/// Padded with zeros if shorter.
/// ## target
/// State register, initialized to |0...0⟩.
/// ## phaseGradient
/// Phase-gradient ancilla register (rotationBits qubits), pre-initialized.
///
/// # Remarks
/// Qubit ordering: target[0] = MSB (first-allocated = highest bit of state index).
/// The operation is not adjointable due to mid-circuit measurements.
operation QroamStatePrep(
    coefficients : Double[],
    target : Qubit[],
    phaseGradient : Qubit[],
    angleReg : Qubit[]
) : Unit {
    let nQubits = Length(target);
    let rotationBits = Length(phaseGradient);
    Fact(nQubits >= 1, "Need at least 1 target qubit.");
    Fact(rotationBits >= 2, "Phase gradient register needs at least 2 qubits.");

    // Classical pre-computation: SBM angle tree quantized to rotationBits bits.
    let angleTree = ComputeSBMAngles(coefficients, nQubits, rotationBits);

    // Iterate from level 0 (MSB = target[0]) to level n-1 (LSB = target[nQubits-1]).
    // At level i, 2^i angles are conditioned on the i previously-prepared qubits.
    for level in 0..nQubits - 1 {
        let targetQubit = target[level];

        // Extract this level's angles from the binary-heap-indexed tree.
        let startIdx = 1 <<< level;
        let numAngles = 1 <<< level;

        if level == 0 {
            // Root level: single unconditional rotation Ry(θ_root).
            let angleBits = IntAsBoolArray(angleTree[1], rotationBits);
            within {
                ApplyPauliFromBitString(PauliX, true, angleBits, angleReg);
            } apply {
                RyViaPhaseGradient(targetQubit, angleReg, phaseGradient);
            }
        } else {
            // Address = previously prepared qubits (target[0..level-1]), reversed
            // so that Select's LE convention yields the correct tree node index.
            let address = Reversed(target[0..level - 1]);

            // Build QROM data table: Bool[2^level][rotationBits].
            let data = ComputeQROMData(angleTree, startIdx, numAngles, rotationBits);

            within {
                Select(data, address, angleReg);
            } apply {
                RyViaPhaseGradient(targetQubit, angleReg, phaseGradient);
            }
        }
    }

    // Sign correction: the Ry rotations only produce |α_j| (positive amplitudes).
    // For real states with negative coefficients, flip the phase of those basis states.
    let signData = ComputeSignBits(coefficients, nQubits);
    if Any(row -> row[0], signData) {
        // At least one coefficient is negative — apply sign correction via QROM.
        // Load 1 sign bit per basis state, apply Z, uncompute.
        // Reversed(target) so Select's LE address matches coefficient index j.
        use signBit = Qubit[1];
        within {
            Select(signData, Reversed(target), signBit);
        } apply {
            Z(signBit[0]);
        }
    }
}

/// # Summary
/// Compute the SBM (Shende-Bullock-Markov) rotation angles for Grover-Rudolph state prep.
///
/// # Description
/// Builds a binary tree of probabilities p_i = Σ_{leaves(i)} |α_j|² and converts
/// each node to a quantized Ry angle:
///   θ_i = arccos(√(p_left / p_i))
///   quantized_i = Round(2^b · θ / (2π)) mod 2^b
///
/// The RyViaPhaseGradient operation applies Ry(4π·x/2^b), so to get Ry(2θ) we need
/// x = 2^b · θ / (2π).
///
/// # Input
/// ## coefficients
/// Real amplitudes (will be padded to length 2^nQubits).
/// ## nQubits
/// Number of target qubits.
/// ## rotationBits
/// Phase-gradient precision (number of bits).
///
/// # Output
/// Int array of length 2^nQubits, indexed as a binary heap (root at index 1).
function ComputeSBMAngles(coefficients : Double[], nQubits : Int, rotationBits : Int) : Int[] {
    let nCoeffs = 1 <<< nQubits;

    // Pad coefficients to 2^n.
    mutable coeffs = Repeated(0.0, nCoeffs);
    for i in 0..MinI(Length(coefficients), nCoeffs) - 1 {
        set coeffs w/= i <- coefficients[i];
    }

    // Build probability tree (binary heap, 1-indexed).
    // Leaves at [nCoeffs..2*nCoeffs-1] store |α_j|².
    mutable tree = Repeated(0.0, 2 * nCoeffs);
    for i in 0..nCoeffs - 1 {
        set tree w/= (nCoeffs + i) <- coeffs[i] * coeffs[i];
    }
    // Internal nodes = sum of children.
    for i in (nCoeffs - 1)..-1..1 {
        set tree w/= i <- tree[2 * i] + tree[2 * i + 1];
    }

    // Compute quantized angles.
    let scale = IntAsDouble(1 <<< rotationBits);
    mutable angles = Repeated(0, nCoeffs);

    for level in 0..nQubits - 1 {
        let startIdx = 1 <<< level;
        let endIdx = (1 <<< (level + 1)) - 1;
        for node in startIdx..endIdx {
            let pTotal = tree[node];
            let pLeft = tree[2 * node];

            mutable angle = 0.0;
            if pTotal > 1e-15 {
                let cosVal = MinD(1.0, Sqrt(pLeft / pTotal));
                set angle = ArcCos(cosVal);
            }

            // Quantize: x = 2^b · θ / (2π), so Ry(4π·x/2^b) = Ry(2θ) as desired.
            let xInt = Round(scale * angle / (2.0 * PI())) % (1 <<< rotationBits);
            set angles w/= node <- xInt;
        }
    }

    return angles;
}

/// # Summary
/// Converts a slice of the angle tree into QROM-compatible Bool[][] format.
function ComputeQROMData(angleTree : Int[], startIdx : Int, count : Int, rotationBits : Int) : Bool[][] {
    mutable data : Bool[][] = [];
    for i in 0..count - 1 {
        set data += [IntAsBoolArray(angleTree[startIdx + i], rotationBits)];
    }
    return data;
}

/// # Summary
/// Computes the sign correction table for real-valued state preparation.
///
/// # Description
/// Returns a Bool[2^n][1] table where entry j is [true] if coefficients[j] < 0,
/// and [false] otherwise. Used with Select to flip the phase of basis states
/// corresponding to negative coefficients.
///
/// # Input
/// ## coefficients
/// Real amplitudes of the target state.
/// ## nQubits
/// Number of target qubits.
///
/// # Output
/// Bool[][] with 2^nQubits rows, each containing a single Bool (the sign bit).
function ComputeSignBits(coefficients : Double[], nQubits : Int) : Bool[][] {
    let nCoeffs = 1 <<< nQubits;
    mutable signTable : Bool[][] = [];
    for i in 0..nCoeffs - 1 {
        if i < Length(coefficients) {
            set signTable += [[coefficients[i] < 0.0]];
        } else {
            set signTable += [[false]];
        }
    }
    return signTable;
}
