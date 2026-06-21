// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import Std.Math.*;
import Std.Convert.*;
import Std.Arrays.*;
import Std.Arithmetic.RippleCarryCGIncByLE;
import Std.Canon.ApplyQFT;

export PreparePhaseGradientState, RyViaPhaseGradient, RzViaPhaseGradient;

/// # Summary
/// Prepares the phase gradient state |φ⟩ = (1/√2^n) Σ_k exp(-2πi·k/2^n) |k⟩_LE.
///
/// # Description
/// The phase gradient state is the eigenstate of LE addition with eigenvalue
/// e^{+2πi·x/2^n} for adding x. Prepared via QFT†|1⟩ where the QFT encoding
/// matches the LE adder basis.
///
/// Note: Q#'s ApplyQFT omits the final bit-reversal swaps, so we must use
/// the non-Reversed register to ensure the QFT output aligns with the LE
/// adder (RippleCarryCGIncByLE). With ApplyQFT(pgr) (pgr[0]=MSB for QFT),
/// the "no bit-reversal swap" output places frequency k at
/// state_idx = bit_reverse(k), which corresponds to pgr_LE = k.
///
/// # Input
/// ## phaseGradient
/// Register to prepare the phase gradient state in. Must be initialized to |0...0⟩.
/// Register is in little-endian format (pgr[0] = LSB for the adder).
operation PreparePhaseGradientState(phaseGradient : Qubit[]) : Unit is Adj + Ctl {
    let n = Length(phaseGradient);
    X(phaseGradient[n - 1]);
    Adjoint ApplyQFT(phaseGradient);
}

/// # Summary
/// Applies Rz(θ) to a target qubit using phase gradient addition.
///
/// # Description
/// Implements Rz(4π·x/2^b) where x is the integer value stored in angleQubits
/// and b is the number of bits. Uses the CNOT-adder-CNOT pattern:
///   1. CNOT target into each PGR qubit (flips PGR conditioned on target=|1⟩)
///   2. Unconditional add angleQubits into PGR
///   3. CNOT target into each PGR qubit (unflips)
/// Net effect: PGR += angle when target=|0⟩, PGR -= angle when target=|1⟩.
/// Cost: n Toffoli (adder) + 2b CNOTs, vs 2n Toffoli for Controlled adder.
///
/// Reference: Sanders et al. (arXiv:2007.07391, §IIA1, Figure 4a).
///
/// # Input
/// ## targetQubit
/// The qubit to apply the rotation to.
/// ## angleQubits
/// Register containing the binary representation of the rotation angle.
/// ## phaseGradient
/// The phase gradient ancilla register.
operation RzViaPhaseGradient(
    targetQubit : Qubit,
    angleQubits : Qubit[],
    phaseGradient : Qubit[]
) : Unit is Adj + Ctl {
    // CNOT-adder-CNOT: cheaper than Controlled RippleCarryCGIncByLE
    for k in 0..Length(phaseGradient) - 1 {
        CNOT(targetQubit, phaseGradient[k]);
    }
    RippleCarryCGIncByLE(angleQubits, phaseGradient);
    for k in 0..Length(phaseGradient) - 1 {
        CNOT(targetQubit, phaseGradient[k]);
    }
}

/// # Summary
/// Applies Ry(θ) to a target qubit using phase gradient addition.
///
/// # Description
/// Implements Ry(θ) via the decomposition Ry = S†·H·Rz·H·S.
/// The angle is encoded in the angleQubits register.
///
/// # Input
/// ## targetQubit
/// The qubit to apply the Y-rotation to.
/// ## angleQubits
/// Register containing the binary representation of the rotation angle.
/// ## phaseGradient
/// The phase gradient ancilla register.
operation RyViaPhaseGradient(
    targetQubit : Qubit,
    angleQubits : Qubit[],
    phaseGradient : Qubit[]
) : Unit is Adj + Ctl {
    S(targetQubit);
    H(targetQubit);
    RzViaPhaseGradient(targetQubit, angleQubits, phaseGradient);
    H(targetQubit);
    Adjoint S(targetQubit);
}
