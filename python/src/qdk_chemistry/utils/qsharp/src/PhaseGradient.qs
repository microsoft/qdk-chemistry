// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Phase gradient operations for multiplexed rotations.
///
/// Implements Ry and Rz rotations via phase gradient addition.
/// Given a phase gradient state |φ⟩ = (1/√2^n) Σ_k exp(-2πi·k/2^n) |k⟩,
/// adding x into the phase gradient register applies a phase e^{2πi·x/2^n},
/// corresponding to Rz when conditioned on a target qubit via CNOT.
///
/// Reference: Sanders et al. (arXiv:2007.07391). Appendix A.
namespace QDKChemistry.Utils.PhaseGradient {

    import Std.Arithmetic.RippleCarryCGIncByLE;
    import Std.Canon.ApplyQFT;
    import Std.Core.Length;

    /// Prepares the phase gradient state |φ⟩ = (1/√2^n) Σ_k exp(-2πi·k/2^n) |k⟩_LE.
    ///
    /// Prepared via QFT†|1⟩. The QFT output (without bit-reversal swaps)
    /// aligns with the LE adder (RippleCarryCGIncByLE).
    /// Ideally this is prepared at the beginning of a circuit and reused throughout.
    operation PreparePhaseGradientState(phaseGradient : Qubit[]) : Unit is Adj + Ctl {
        let n = Length(phaseGradient);
        X(phaseGradient[n - 1]);
        Adjoint ApplyQFT(phaseGradient);
    }

    /// # Summary
    /// Applies Rz(4π·x/2^b) to a target qubit using phase gradient addition.
    ///
    /// # Description
    /// x is the integer value stored in angleQubits and b is the number of bits.
    /// Adding c into the phase gradient register kicks back a phase e^{2πi·c/2^b}, so
    /// negating the register conditionally turns the adder into a subtractor on one branch
    /// of the target. Conditioning that negation on the target being |0⟩ realizes
    /// diag(e^{-2πi·x/2^b}, e^{+2πi·x/2^b}) = Rz(4π·x/2^b), matching Sanders et al.
    /// Appendix A and Qualtran's `RzViaPhaseGradient`.
    /// Cost: b-1 CCZ and b-1 measurements (the adder) plus 2b CNOTs. Preparing and
    /// unpreparing the phase gradient register is extra and is amortized when the register
    /// is reused across rotations.
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
        within {
            X(targetQubit);
            for k in 0..Length(phaseGradient) - 1 {
                CNOT(targetQubit, phaseGradient[k]);
            }
        } apply {
            RippleCarryCGIncByLE(angleQubits, phaseGradient);
        }
    }

    /// # Summary
    /// Applies Ry(4π·x/2^b) to a target qubit using phase gradient addition.
    ///
    /// # Description
    /// Conjugating by `Adjoint S` and `H` maps the Z axis onto +Y, so the positive
    /// Rz above yields a positive Ry.
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
        within {
            Adjoint S(targetQubit);
            H(targetQubit);
        } apply {
            RzViaPhaseGradient(targetQubit, angleQubits, phaseGradient);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test wrappers — Qubit layout: target[0], angle[0..n-1], pg[0..n-1].
    // ═══════════════════════════════════════════════════════════════════════════

    /// Test wrapper: apply Ry via phase gradient on |0⟩ and leave state.
    internal operation TestRy(angleValue : Int, nBits : Int) : Unit {
        let target = QIR.Runtime.AllocateQubitArray(1);
        let angle = QIR.Runtime.AllocateQubitArray(nBits);
        let pg = QIR.Runtime.AllocateQubitArray(nBits);

        for k in 0..nBits - 1 {
            if (angleValue >>> k) &&& 1 == 1 { X(angle[k]); }
        }

        within {
            PreparePhaseGradientState(pg);
        } apply {
            RyViaPhaseGradient(target[0], angle, pg);
        }
    }

    /// Test wrapper: apply Rz via phase gradient to |+⟩ so the relative phase, and
    /// therefore the sign convention, is observable in the dumped state.
    internal operation TestRzOnPlus(angleValue : Int, nBits : Int) : Unit {
        let target = QIR.Runtime.AllocateQubitArray(1);
        let angle = QIR.Runtime.AllocateQubitArray(nBits);
        let pg = QIR.Runtime.AllocateQubitArray(nBits);

        H(target[0]);

        for k in 0..nBits - 1 {
            if (angleValue >>> k) &&& 1 == 1 { X(angle[k]); }
        }

        within {
            PreparePhaseGradientState(pg);
        } apply {
            RzViaPhaseGradient(target[0], angle, pg);
        }
    }

    /// Test wrapper: apply Ry then Adjoint Ry (round-trip identity check).
    internal operation TestRyRoundtrip(angleValue : Int, nBits : Int) : Unit {
        let target = QIR.Runtime.AllocateQubitArray(1);
        let angle = QIR.Runtime.AllocateQubitArray(nBits);
        let pg = QIR.Runtime.AllocateQubitArray(nBits);

        H(target[0]);

        for k in 0..nBits - 1 {
            if (angleValue >>> k) &&& 1 == 1 { X(angle[k]); }
        }
        within {
            PreparePhaseGradientState(pg);
            RyViaPhaseGradient(target[0], angle, pg);
        } apply {}
    }
}
