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
    import Std.Core.Length;
    import Std.Measurement.MResetZ;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;

    /// # Summary
    /// Flips `target` when the little-endian phase register equals any target index.
    operation MarkPhaseIndices(
        numPhaseQubits : Int,
        targetIndices : Int[],
        register : Qubit[],
        target : Qubit,
    ) : Unit is Adj {
        let phaseRegister = register[0..numPhaseQubits - 1];
        for targetIndex in targetIndices {
            ApplyControlledOnInt(targetIndex, X, phaseRegister, target);
        }
    }

    /// # Summary
    /// Flips `target` when the little-endian phase-register value is at most `threshold`.
    operation MarkPhaseAtOrBelow(
        numPhaseQubits : Int,
        threshold : Int,
        register : Qubit[],
        target : Qubit,
    ) : Unit is Adj {
        let phaseRegister = register[0..numPhaseQubits - 1];
        use encodedThreshold = Qubit[numPhaseQubits];
        within {
            ApplyXorInPlace(threshold, encodedThreshold);
        } apply {
            ApplyIfGreaterLE(X, phaseRegister, encodedThreshold, target);
            X(target);
        }
    }

    function MakePhaseIndexMarkerOp(
        numPhaseQubits : Int,
        targetIndices : Int[],
    ) : (Qubit[], Qubit) => Unit is Adj {
        MarkPhaseIndices(numPhaseQubits, targetIndices, _, _)
    }

    function MakePhaseThresholdMarkerOp(
        numPhaseQubits : Int,
        threshold : Int,
    ) : (Qubit[], Qubit) => Unit is Adj {
        MarkPhaseAtOrBelow(numPhaseQubits, threshold, _, _)
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
}
