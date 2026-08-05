// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Generic amplitude amplification.
///
/// The good subspace is supplied as a good state oracle that flips a flag qubit;
/// nothing here knows what the oracle tests.
///
/// Reference: L. Lin, *Lecture Notes on Quantum Algorithms for Scientific
/// Computation*, arXiv:2201.08309, Chapter 2.
namespace QDKChemistry.Utils.AmplitudeAmplification {

    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyToEachCA;
    import Std.Arithmetic.*;
    import Std.Arrays.Subarray;
    import Std.Core.Length;
    import Std.Measurement.MResetEachZ;
    import Std.Measurement.MResetZ;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;

    /// # Summary
    /// Flips `target` when the little-endian phase-register value lies in the
    /// half-open interval [`lowerBound`, `upperBound`).
    operation MarkPhaseRange(
        numPhaseQubits : Int,
        lowerBound : Int,
        upperBound : Int,
        register : Qubit[],
        target : Qubit,
    ) : Unit is Adj {
        let phaseRegister = register[0..numPhaseQubits - 1];
        let phaseBinCount = 1 <<< numPhaseQubits;
        if lowerBound == 0 and upperBound == phaseBinCount {
            X(target);
        } elif upperBound == lowerBound + 1 {
            ApplyControlledOnInt(lowerBound, X, phaseRegister, target);
        } elif lowerBound == 0 {
            use encodedUpper = Qubit[numPhaseQubits];
            within {
                ApplyXorInPlace(upperBound, encodedUpper);
            } apply {
                ApplyIfGreaterLE(X, encodedUpper, phaseRegister, target);
            }
        } elif upperBound == phaseBinCount {
            use encodedLower = Qubit[numPhaseQubits];
            within {
                ApplyXorInPlace(lowerBound, encodedLower);
            } apply {
                ApplyIfGreaterLE(X, encodedLower, phaseRegister, target);
                X(target);
            }
        } else {
            use encodedLower = Qubit[numPhaseQubits];
            use encodedUpper = Qubit[numPhaseQubits];
            use lowerFlag = Qubit();
            use upperFlag = Qubit();
            within {
                ApplyXorInPlace(lowerBound, encodedLower);
                ApplyXorInPlace(upperBound, encodedUpper);
                ApplyIfGreaterLE(X, encodedLower, phaseRegister, lowerFlag);
                X(lowerFlag);
                ApplyIfGreaterLE(X, encodedUpper, phaseRegister, upperFlag);
            } apply {
                Controlled X([lowerFlag, upperFlag], target);
            }
        }
    }

    /// # Summary
    /// Flips `target` when the phase register lies in [`lowerBound`,
    /// `upperBound`) and every signal ancilla is $|0\rangle$. Signal-ancilla
    /// indices are relative to the register that follows the phase qubits.
    operation MarkAcceptedPhase(
        numPhaseQubits : Int,
        signalAncillaIndices : Int[],
        lowerBound : Int,
        upperBound : Int,
        register : Qubit[],
        target : Qubit,
    ) : Unit is Adj {
        let signalAncillas = Subarray(signalAncillaIndices, register[numPhaseQubits...]);
        if Length(signalAncillas) == 0 {
            MarkPhaseRange(numPhaseQubits, lowerBound, upperBound, register, target);
        } else {
            use inRange = Qubit();
            within {
                MarkPhaseRange(numPhaseQubits, lowerBound, upperBound, register, inRange);
                ApplyToEachCA(X, signalAncillas);
            } apply {
                Controlled X(signalAncillas + [inRange], target);
            }
        }
    }

    function MarkTargetStateOp(
        numPhaseQubits : Int,
        signalAncillaIndices : Int[],
        lowerBound : Int,
        upperBound : Int,
    ) : (Qubit[], Qubit) => Unit is Adj {
        MarkAcceptedPhase(numPhaseQubits, signalAncillaIndices, lowerBound, upperBound, _, _)
    }

    /// # Summary
    /// Applies the marking oracle to an all-zeros register and measures the flag.
    /// An entry point so the oracle can be run, drawn and costed on its own;
    /// `MarkTargetStateOp` returns a callable and cannot be executed directly.
    operation MakeMarkedPhaseCircuit(
        numPhaseQubits : Int,
        signalAncillaIndices : Int[],
        lowerBound : Int,
        upperBound : Int,
        numQubits : Int,
    ) : Result[] {
        use register = Qubit[numQubits];
        use flag = Qubit();
        MarkAcceptedPhase(numPhaseQubits, signalAncillaIndices, lowerBound, upperBound, register, flag);
        let outcome = MResetZ(flag);
        ResetAll(register);
        return [outcome];
    }

    //
    // Elementary reflections
    //

    /// # Summary
    /// $2|\psi\rangle\langle\psi| - I$, with
    /// $|\psi\rangle = \text{statePrepOracle}|0\rangle$.
    operation ReflectAboutPreparedState(
        statePrepOracle : Qubit[] => Unit is Adj,
        register : Qubit[],
    ) : Unit is Adj {
        within {
            Adjoint statePrepOracle(register);
        } apply {
            Reflect(register);
        }
    }

    /// # Summary
    /// $I - 2\Pi_G$, where $\Pi_G$ projects onto the good subspace.
    operation ReflectAboutGoodSubspace(
        goodStateOracle : (Qubit[], Qubit) => Unit is Adj,
        register : Qubit[],
    ) : Unit is Adj {
        use flag = Qubit();
        within {
            goodStateOracle(register, flag);
        } apply {
            Z(flag);
        }
    }

    //
    // Amplification loops
    //

    /// # Summary
    /// One Grover iterate $Q = -S_\psi S_G$, a rotation by $2\vartheta$ in the
    /// invariant plane.
    operation AmplitudeAmplificationStep(
        statePrepOracle : Qubit[] => Unit is Adj,
        goodStateOracle : (Qubit[], Qubit) => Unit is Adj,
        register : Qubit[],
    ) : Unit is Adj {
        ReflectAboutGoodSubspace(goodStateOracle, register);
        ReflectAboutPreparedState(statePrepOracle, register);
    }

    /// # Summary
    /// Prepares $|\psi\rangle$ and applies `rounds` Grover iterates in place.
    /// Neither measures nor resets.
    operation ApplyAmplitudeAmplification(
        statePrepOracle : Qubit[] => Unit is Adj,
        goodStateOracle : (Qubit[], Qubit) => Unit is Adj,
        rounds : Int,
        register : Qubit[],
    ) : Unit is Adj {
        if rounds < 0 {
            fail "The number of amplitude-amplification rounds must be nonnegative.";
        }
        statePrepOracle(register);
        for _ in 1..rounds {
            AmplitudeAmplificationStep(statePrepOracle, goodStateOracle, register);
        }
    }

    //
    // Circuit entry points
    //
    // Handed to `Circuit`/`CircuitExecutor`. No measurement-dependent classical
    // control flow, so they compile under the restricted target profiles.

    /// # Summary
    /// Builds an amplitude-amplified circuit and measures the whole register.
    operation MakeAmplifiedCircuit(
        statePrepOracle : Qubit[] => Unit is Adj,
        goodStateOracle : (Qubit[], Qubit) => Unit is Adj,
        rounds : Int,
        numQubits : Int,
    ) : Result[] {
        use register = Qubit[numQubits];
        ApplyAmplitudeAmplification(statePrepOracle, goodStateOracle, rounds, register);
        return MResetEachZ(register);
    }

    /// # Summary
    /// The amplified state preparation as a callable, without measurement, so a
    /// caller can append its own measurement or compose it further.
    function MakeAmplifiedStateOp(
        statePrepOracle : Qubit[] => Unit is Adj,
        goodStateOracle : (Qubit[], Qubit) => Unit is Adj,
        rounds : Int,
    ) : Qubit[] => Unit is Adj {
        ApplyAmplitudeAmplification(statePrepOracle, goodStateOracle, rounds, _)
    }
}
