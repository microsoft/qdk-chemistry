// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Phase gradient operations for resource-efficient quantum rotations.
///
/// Implements Ry and Rz rotations via phase gradient addition,
/// avoiding costly T-gate synthesis for arbitrary angles.
///
/// The phase gradient state |φ⟩ = (1/√2^n) Σ_k exp(-2πi·k/2^n) |k⟩ is an
/// eigenstate of the LE adder with eigenvalue e^{+2πi·x/2^n}. Adding x into
/// the phase gradient register applies a phase e^{2πi·x/2^n}, which
/// implements Rz when conditioned on a target qubit via CNOT.
///
/// Reference: Sanders et al. (arXiv:2007.07391).
namespace QDKChemistry.Utils.PhaseGradient {

    import Std.Arithmetic.RippleCarryCGIncByLE;
    import Std.Canon.ApplyQFT;

    /// Prepares the phase gradient state |φ⟩ = (1/√2^n) Σ_k exp(-2πi·k/2^n) |k⟩_LE.
    ///
    /// Prepared via QFT†|1⟩. The QFT output (without bit-reversal swaps)
    /// aligns with the LE adder (RippleCarryCGIncByLE).
    ///
    /// # Input
    /// ## phaseGradient
    /// Register to prepare, initialized to |0...0⟩. LE format (pgr[0] = LSB).
    operation PreparePhaseGradientState(phaseGradient : Qubit[]) : Unit is Adj + Ctl {
        let n = Length(phaseGradient);
        X(phaseGradient[n - 1]);
        Adjoint ApplyQFT(phaseGradient);
    }

    /// Applies Rz(4π·x/2^b) using phase gradient addition.
    ///
    /// Uses CNOT-adder-CNOT pattern: n Toffoli (adder) + 2b CNOTs,
    /// cheaper than a Controlled adder (2n Toffoli).
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
        phaseGradient : Qubit[],
    ) : Unit is Adj + Ctl {
        for k in 0..Length(phaseGradient) - 1 {
            CNOT(targetQubit, phaseGradient[k]);
        }
        RippleCarryCGIncByLE(angleQubits, phaseGradient);
        for k in 0..Length(phaseGradient) - 1 {
            CNOT(targetQubit, phaseGradient[k]);
        }
    }

    /// Applies Ry(θ) via the decomposition Ry = S†·H·Rz·H·S.
    ///
    /// The rotation angle is encoded in angleQubits as an integer x,
    /// giving Ry(4π·x/2^b) where b = Length(phaseGradient).
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
        phaseGradient : Qubit[],
    ) : Unit is Adj + Ctl {
        S(targetQubit);
        H(targetQubit);
        RzViaPhaseGradient(targetQubit, angleQubits, phaseGradient);
        H(targetQubit);
        Adjoint S(targetQubit);
    }
}
