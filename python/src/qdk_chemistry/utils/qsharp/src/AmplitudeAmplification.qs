// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Generic amplitude amplification.
///
/// The good subspace is supplied as a marking oracle that flips a flag qubit;
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

    //
    // Elementary reflections
    //

    /// # Summary
    /// $2|\psi\rangle\langle\psi| - I$, with
    /// $|\psi\rangle = \text{statePrep}|0\rangle$.
    operation ReflectAboutPreparedState(
        statePrep : Qubit[] => Unit is Adj,
        register : Qubit[],
    ) : Unit is Adj {
        within {
            Adjoint statePrep(register);
        } apply {
            Reflect(register);
        }
    }

    /// # Summary
    /// $I - 2\Pi_G$, where $\Pi_G$ projects onto the marked subspace.
    operation ReflectAboutMarkedSubspace(
        markingOracle : (Qubit[], Qubit) => Unit is Adj,
        register : Qubit[],
    ) : Unit is Adj {
        use flag = Qubit();
        within {
            markingOracle(register, flag);
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
        statePrep : Qubit[] => Unit is Adj,
        markingOracle : (Qubit[], Qubit) => Unit is Adj,
        register : Qubit[],
    ) : Unit is Adj {
        ReflectAboutMarkedSubspace(markingOracle, register);
        ReflectAboutPreparedState(statePrep, register);
    }

    /// # Summary
    /// Prepares $|\psi\rangle$ and applies `rounds` Grover iterates in place.
    /// Neither measures nor resets.
    operation ApplyAmplitudeAmplification(
        statePrep : Qubit[] => Unit is Adj,
        markingOracle : (Qubit[], Qubit) => Unit is Adj,
        rounds : Int,
        register : Qubit[],
    ) : Unit is Adj {
        if rounds < 0 {
            fail "The number of amplitude-amplification rounds must be nonnegative.";
        }
        statePrep(register);
        for _ in 1..rounds {
            AmplitudeAmplificationStep(statePrep, markingOracle, register);
        }
    }

    //
    // Circuit entry points
    //
    // Handed to `Circuit`/`CircuitExecutor`. No measurement-dependent classical
    // control flow, so they compile under the restricted target profiles.

    /// # Summary
    /// Builds and measures an amplitude-amplified circuit.
    ///
    /// # Input
    /// ## measuredIndices
    /// Register indices to measure, in output order.
    operation MakeAmplifiedCircuit(
        preparation : Qubit[] => Unit is Adj,
        markingOracle : (Qubit[], Qubit) => Unit is Adj,
        rounds : Int,
        numQubits : Int,
        measuredIndices : Int[],
    ) : Result[] {
        use register = Qubit[numQubits];
        ApplyAmplitudeAmplification(preparation, markingOracle, rounds, register);
        let results = MeasureSelected(register, measuredIndices);
        ResetAll(register);
        return results;
    }

    /// # Summary
    /// Measures the requested register indices, in order.
    operation MeasureSelected(register : Qubit[], measuredIndices : Int[]) : Result[] {
        mutable results = [Zero, size = Length(measuredIndices)];
        for index in 0..Length(measuredIndices) - 1 {
            set results w/= index <- MResetZ(register[measuredIndices[index]]);
        }
        return results;
    }

    /// # Summary
    /// Prepares the state, marks it and measures the flag. The fraction of `One`
    /// outcomes estimates the overlap $a$ that sets the number of rounds.
    operation MakeAcceptanceCircuit(
        preparation : Qubit[] => Unit is Adj,
        markingOracle : (Qubit[], Qubit) => Unit is Adj,
        numQubits : Int,
    ) : Result[] {
        use register = Qubit[numQubits];
        use flag = Qubit();
        preparation(register);
        markingOracle(register, flag);
        let outcome = MResetZ(flag);
        ResetAll(register);
        return [outcome];
    }
}
