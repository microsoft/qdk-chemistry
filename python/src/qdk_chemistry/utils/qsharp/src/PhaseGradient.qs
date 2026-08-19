// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Phase gradient operations for multiplexed rotations.
///
/// Implements Ry and Rz rotations via phase gradient addition.///
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
    /// Idealy this is prepared at the beginning of a circuit and reused throughout.
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
    /// and b is the number of bits. Uses the CNOT-adder-CNOT pattern with
    /// net effect: PGR += angle when target=|0⟩, PGR -= angle when target=|1⟩.
    /// Cost: n Toffoli (adder) + 2b CNOTs.
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
            for k in 0..Length(phaseGradient) - 1 {
                CNOT(targetQubit, phaseGradient[k]);
            }
        } apply {
            RippleCarryCGIncByLE(angleQubits, phaseGradient);
        }
    }

    /// # Summary
    /// Applies Ry(θ) to a target qubit using phase gradient addition.
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
            S(targetQubit);
            H(targetQubit);
        } apply {
            RzViaPhaseGradient(targetQubit, angleQubits, phaseGradient);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test wrappers — Qubit layout: target[0], angle[0..n-1], pg[0..n-1].
    // ═══════════════════════════════════════════════════════════════════════════

    /// Test wrapper: apply Ry via phase gradient on |0⟩ and leave state.
    operation TestRy(angleValue : Int, nBits : Int) : Unit {
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

    // ═══════════════════════════════════════════════════════════════════════════
    // Ancilla preparation helpers
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build an ancillaPrep callback that prepares the phase gradient state
    /// on the last `numPhaseGradientQubits` qubits of the beAncillas array.
    /// Returns a no-op when numPhaseGradientQubits == 0.
    function MakePhaseGradientAncillaPrep(numPhaseGradientQubits : Int) : Qubit[] => Unit is Adj {
        (beAncillas) => {
            if numPhaseGradientQubits > 0 {
                let n = Length(beAncillas);
                let pgReg = beAncillas[n - numPhaseGradientQubits..n - 1];
                PreparePhaseGradientState(pgReg);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test wrappers
    // ═══════════════════════════════════════════════════════════════════════════

    /// Test wrapper: apply Ry then Adjoint Ry (round-trip identity check).
    operation TestRyRoundtrip(angleValue : Int, nBits : Int) : Unit {
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
