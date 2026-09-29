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
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;

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
    /// Prepares the product state whose qubit j is (|0⟩ + e^{i·angles[j]}|1⟩)/√2.
    ///
    /// # Description
    /// With `GeneralizedPhaseGradientAngles(phi, n)` this is the catalyst
    /// Σ_k e^{-i·phi·k}|k⟩_LE that `PhaseByGeneralizedGradient` consumes. Its cost is one
    /// arbitrary rotation per qubit, paid once for every use of the register.
    operation PrepareGeneralizedPhaseGradient(angles : Double[], catalyst : Qubit[]) : Unit is Adj + Ctl {
        Fact(Length(angles) == Length(catalyst), "PrepareGeneralizedPhaseGradient needs one angle per qubit.");
        for j in 0..Length(catalyst) - 1 {
            H(catalyst[j]);
            R1(angles[j], catalyst[j]);
        }
    }

    /// # Summary
    /// Prepares consecutive phase gradient registers, each Σ_k e^{-i·phase·k}|k⟩_LE.
    ///
    /// # Description
    /// Each entry is `(phase, numQubits, binary)`. A binary entry has phase 2π/2^numQubits
    /// and is prepared by `PreparePhaseGradientState`; any other is prepared qubit by qubit
    /// by `PrepareGeneralizedPhaseGradient`. The registers fill `register` in order.
    operation PreparePhaseGradients(gradients : (Double, Int, Bool)[], register : Qubit[]) : Unit is Adj + Ctl {
        let offsets = PhaseGradientOffsets(gradients);
        Fact(
            offsets[Length(gradients)] == Length(register),
            "PreparePhaseGradients needs a register exactly as large as its gradients."
        );
        for index in 0..Length(gradients) - 1 {
            let (phase, numQubits, binary) = gradients[index];
            let slice = register[offsets[index]..offsets[index + 1] - 1];
            if binary {
                PreparePhaseGradientState(slice);
            } else {
                PrepareGeneralizedPhaseGradient(GeneralizedPhaseGradientAngles(phase, numQubits), slice);
            }
        }
    }

    /// Where each gradient's register starts, followed by their total size.
    internal function PhaseGradientOffsets(gradients : (Double, Int, Bool)[]) : Int[] {
        mutable offsets = [0];
        for (_, numQubits, _) in gradients {
            set offsets += [offsets[Length(offsets) - 1] + numQubits];
        }
        return offsets;
    }

    /// Returns a preparation of the given phase gradients, as `PreparePhaseGradients` describes.
    function MakePhaseGradientsPrep(gradients : (Double, Int, Bool)[]) : Qubit[] => Unit is Adj + Ctl {
        PreparePhaseGradients(gradients, _)
    }

    /// # Summary
    /// Wraps a controlled operation that expects caller-prepared phase gradients at the end
    /// of its targets, so that it prepares them itself.
    ///
    /// # Description
    /// For callers that cannot share one register across several such operations, for
    /// example because their gradients differ. The registers are prepared around every call.
    function MakeSelfPreparingControlledOp(
        gradients : (Double, Int, Bool)[],
        op : (Qubit, Qubit[]) => Unit is Adj + Ctl
    ) : (Qubit, Qubit[]) => Unit is Adj + Ctl {
        SelfPreparingControlled(gradients, op, _, _)
    }

    internal operation SelfPreparingControlled(
        gradients : (Double, Int, Bool)[],
        op : (Qubit, Qubit[]) => Unit is Adj + Ctl,
        control : Qubit,
        targets : Qubit[]
    ) : Unit is Adj + Ctl {
        let size = PhaseGradientOffsets(gradients)[Length(gradients)];
        within {
            PreparePhaseGradients(gradients, targets[Length(targets) - size...]);
        } apply {
            op(control, targets);
        }
    }

    /// The per-qubit angles of the n-qubit catalyst for the phase `phi`: -phi·2^j.
    function GeneralizedPhaseGradientAngles(phi : Double, n : Int) : Double[] {
        mutable angles = [];
        for j in 0..n - 1 {
            set angles += [-phi * IntAsDouble(1 <<< j)];
        }
        return angles;
    }

    /// Computes into `carries[i]` the carry out of bit i of `xs + ys`, leaving both unchanged.
    internal operation ComputeCarriesLE(xs : Qubit[], ys : Qubit[], carries : Qubit[]) : Unit is Adj {
        for i in 0..Length(xs) - 1 {
            if i == 0 {
                AND(xs[0], ys[0], carries[0]);
            } else {
                // MAJ(x, y, c) = (x ⊕ c)(y ⊕ c) ⊕ c
                within {
                    CNOT(carries[i - 1], xs[i]);
                    CNOT(carries[i - 1], ys[i]);
                } apply {
                    AND(xs[i], ys[i], carries[i]);
                }
                CNOT(carries[i - 1], carries[i]);
            }
        }
    }

    /// # Summary
    /// Applies e^{i·phi·w} to the little-endian integer w in `weight`, for any angle phi.
    ///
    /// # Description
    /// The generalized phase-gradient addition (GPGA) of Sec. 4.4 of :cite:`Apel2026`.
    /// Adding w into the catalyst Σ_k e^{-i·phi·k}|k⟩ kicks back e^{i·phi·w} and leaves the
    /// catalyst unchanged, except that the terms which wrap past 2^n lose e^{-i·phi·2^n}.
    /// Those are exactly the terms whose addition carries out of the top bit, so one
    /// payload rotation R1(phi·2^n) on that carry restores them. Unlike
    /// `RzViaPhaseGradient`, phi need not be a multiple of 2π/2^n, so the angle is exact
    /// and the catalyst only needs as many qubits as w has bits.
    ///
    /// Cost: one arbitrary rotation and 2n - 1 AND operations (n for the carries, n - 1
    /// for the modular adder). Under control, the catalyst stays an eigenstate on both
    /// branches if w is masked by the control, which costs n more AND operations and
    /// leaves the adder and the payload rotation uncontrolled.
    ///
    /// # Input
    /// ## phi
    /// The phase per unit of w.
    /// ## weight
    /// The integer w, little-endian.
    /// ## catalyst
    /// A register prepared by `PrepareGeneralizedPhaseGradient` with
    /// `GeneralizedPhaseGradientAngles(phi, Length(weight))`, returned in that state.
    operation PhaseByGeneralizedGradient(phi : Double, weight : Qubit[], catalyst : Qubit[]) : Unit is Adj + Ctl {
        body (...) {
            let n = Length(weight);
            Fact(Length(catalyst) == n, "PhaseByGeneralizedGradient needs one catalyst qubit per weight bit.");
            if n > 0 {
                use carries = Qubit[n];
                within {
                    ComputeCarriesLE(weight, catalyst, carries);
                } apply {
                    R1(phi * IntAsDouble(1 <<< n), carries[n - 1]);
                }
                RippleCarryCGIncByLE(weight, catalyst);
            }
        }
        controlled (controls, ...) {
            if Length(controls) == 0 {
                PhaseByGeneralizedGradient(phi, weight, catalyst);
            } else {
                use masked = Qubit[Length(weight)];
                use joint = Qubit[Length(controls) > 1 ? 1 | 0];
                let control = Length(controls) > 1 ? joint[0] | controls[0];
                within {
                    if Length(controls) > 1 {
                        Controlled X(controls, control);
                    }
                    for j in 0..Length(weight) - 1 {
                        AND(control, weight[j], masked[j]);
                    }
                } apply {
                    PhaseByGeneralizedGradient(phi, masked, catalyst);
                }
            }
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
