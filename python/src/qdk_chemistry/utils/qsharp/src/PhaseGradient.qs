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
    import Std.Arrays.Mapped;
    import Std.Canon.ApplyQFT;
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;
    import Std.Math.AbsD;
    import Std.Math.PI;
    import Std.Math.Round;
    import QDKChemistry.Utils.SelectSwap.SelectSwap;

    /// Prepares the phase gradient state |φ⟩ = (1/√2^n) Σ_k exp(-2πi·k/2^n) |k⟩_LE.
    ///
    /// The QFT output (without bit-reversal swaps) aligns with the LE adder (RippleCarryCGIncByLE).
    /// Ideally this is prepared at the beginning of a circuit and reused throughout.
    operation PreparePhaseGradientState(phaseGradient : Qubit[]) : Unit is Adj + Ctl {
        let n = Length(phaseGradient);
        Fact(n > 0, "phase gradient register cannot be empty");
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

    /// # Summary
    /// Returns the `bits`-bit word x for which `RyViaPhaseGradient` applies Ry(angle).
    ///
    /// # Description
    /// Ry(4π·x/2^b) = Ry(angle) gives x = Round(2^b · angle / (4π)) mod 2^b.
    function QuantizeRyAngle(angle : Double, bits : Int) : Int {
        Fact(AbsD(angle) <= 4.0 * PI(), "QuantizeRyAngle: angle must be finite and within [-4π, 4π]");
        let scale = 1 <<< bits;
        let raw = Round(IntAsDouble(scale) * angle / (4.0 * PI()));
        ((raw % scale) + scale) % scale
    }

    /// # Summary
    /// Quantizes each angle with `QuantizeRyAngle` into a `SelectSwap` table of `bits`-bit words.
    function QuantizeRyAngles(angles : Double[], bits : Int) : Bool[][] {
        Mapped(angle -> IntAsBoolArray(QuantizeRyAngle(angle, bits), bits), angles)
    }

    /// # Summary
    /// Applies Ry(4π·data[a]/2^b) to `targetQubit` for each address state |a⟩.
    ///
    /// # Description
    /// Loads the addressed word into `angleReg` with `SelectSwap`, rotates through the phase
    /// gradient, and unloads the word.
    ///
    /// # Input
    /// ## data
    /// Bool[N][b]: words from `QuantizeRyAngles`.
    /// ## address
    /// Little-endian address register with at least ⌈log₂ N⌉ qubits.
    /// ## targetQubit
    /// The qubit to apply the Y-rotation to.
    /// ## phaseGradient
    /// The phase gradient ancilla register (b qubits), pre-initialized.
    /// ## angleReg
    /// Clean b-qubit register that receives the loaded word.
    operation ApplyMultiplexedRy(
        data : Bool[][],
        address : Qubit[],
        targetQubit : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj + Ctl {
        within {
            SelectSwap(-1, data, address, angleReg);
        } apply {
            RyViaPhaseGradient(targetQubit, angleReg, phaseGradient);
        }
    }

    /// # Summary
    /// `ApplyMultiplexedRy` that rotates only when every qubit of `controls` is |1⟩.
    ///
    /// # Description
    /// Only the lookup is controlled: otherwise the angle register stays zero and the
    /// rotation is the identity. This is cheaper than `Controlled ApplyMultiplexedRy`, which
    /// also controls the adder.
    operation ApplyControlledMultiplexedRy(
        data : Bool[][],
        address : Qubit[],
        controls : Qubit[],
        targetQubit : Qubit,
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj + Ctl {
        within {
            Controlled SelectSwap(controls, (-1, data, address, angleReg));
        } apply {
            RyViaPhaseGradient(targetQubit, angleReg, phaseGradient);
        }
    }

    /// Test wrapper: Ry via phase gradient on `[target | angle | gradient]`.
    internal function MakeTestRyOp(angleValue : Int, nBits : Int) : Qubit[] => Unit {
        (qs) => {
            let angle = qs[1..nBits];
            let pg = qs[nBits + 1..2 * nBits];
            ApplyXorInPlace(angleValue, angle);
            within {
                PreparePhaseGradientState(pg);
            } apply {
                RyViaPhaseGradient(qs[0], angle, pg);
            }
        }
    }

    /// Test wrapper: Rz via phase gradient on `[target | angle | gradient]`.
    internal function MakeTestRzOnPlusOp(angleValue : Int, nBits : Int) : Qubit[] => Unit {
        (qs) => {
            let angle = qs[1..nBits];
            let pg = qs[nBits + 1..2 * nBits];
            H(qs[0]);
            ApplyXorInPlace(angleValue, angle);
            within {
                PreparePhaseGradientState(pg);
            } apply {
                RzViaPhaseGradient(qs[0], angle, pg);
            }
        }
    }

    /// Test wrapper: Ry round-trip on `[target | angle | gradient]`.
    internal function MakeTestRyRoundtripOp(angleValue : Int, nBits : Int) : Qubit[] => Unit {
        (qs) => {
            let angle = qs[1..nBits];
            let pg = qs[nBits + 1..2 * nBits];
            H(qs[0]);
            ApplyXorInPlace(angleValue, angle);
            within {
                PreparePhaseGradientState(pg);
                RyViaPhaseGradient(qs[0], angle, pg);
            } apply {}
        }
    }
}
