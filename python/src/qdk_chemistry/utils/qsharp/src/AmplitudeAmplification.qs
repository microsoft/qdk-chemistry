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

    import Std.Canon.ApplyToEachCA;
    import Std.Core.Length;
    import Std.Intrinsic.R1;
    import Std.Measurement.MResetZ;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;

    //
    // Elementary reflections
    //

    /// # Summary
    /// $I - (1 - e^{i\phi})|0\rangle\langle 0|$: phases the all-zeros state only.
    ///
    /// `phase = PI()` gives $I - 2|0\rangle\langle 0|$.
    operation ApplyPhaseToAllZeros(phase : Double, register : Qubit[]) : Unit is Adj + Ctl {
        let numQubits = Length(register);
        if numQubits == 0 {
            // Trivial space: the phase is global and cannot be represented.
        } else {
            within {
                ApplyToEachCA(X, register);
            } apply {
                if numQubits == 1 {
                    R1(phase, register[0]);
                } else {
                    Controlled R1(register[1...], (phase, register[0]));
                }
            }
        }
    }

    /// # Summary
    /// $I - (1 - e^{i\phi})|\psi\rangle\langle\psi|$, with
    /// $|\psi\rangle = \text{statePrep}|0\rangle$.
    operation ApplyPhaseToPreparedState(
        statePrep : Qubit[] => Unit is Adj,
        phase : Double,
        register : Qubit[],
    ) : Unit is Adj {
        within {
            Adjoint statePrep(register);
        } apply {
            ApplyPhaseToAllZeros(phase, register);
        }
    }

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
    /// $I - (1 - e^{i\phi})\Pi_G$, phasing the marked subspace.
    ///
    /// `markingOracle` must leave the register unchanged, so conjugating by it
    /// uncomputes the flag.
    operation ApplyPhaseToMarkedSubspace(
        markingOracle : (Qubit[], Qubit) => Unit is Adj,
        phase : Double,
        register : Qubit[],
    ) : Unit is Adj {
        use flag = Qubit();
        within {
            markingOracle(register, flag);
        } apply {
            R1(phase, flag);
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
    /// The phase-matched iterate $G(\alpha,\beta)$ used by fixed-point
    /// amplification. $\alpha = \beta = \pi$ recovers the Grover iterate.
    operation GeneralizedAmplitudeAmplificationStep(
        statePrep : Qubit[] => Unit is Adj,
        markingOracle : (Qubit[], Qubit) => Unit is Adj,
        markPhase : Double,
        statePhase : Double,
        register : Qubit[],
    ) : Unit is Adj {
        ApplyPhaseToMarkedSubspace(markingOracle, markPhase, register);
        ApplyPhaseToPreparedState(statePrep, statePhase, register);
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

    /// # Summary
    /// Prepares $|\psi\rangle$ and applies a phase-matched iterate sequence.
    ///
    /// Phases come from `AmplitudeAmplification.fixed_point_phases`, in this
    /// order and reflection convention.
    operation ApplyFixedPointAmplitudeAmplification(
        statePrep : Qubit[] => Unit is Adj,
        markingOracle : (Qubit[], Qubit) => Unit is Adj,
        markPhases : Double[],
        statePhases : Double[],
        register : Qubit[],
    ) : Unit is Adj {
        if Length(markPhases) != Length(statePhases) {
            fail "The mark and state phase sequences must have equal length.";
        }
        statePrep(register);
        for index in 0..Length(markPhases) - 1 {
            GeneralizedAmplitudeAmplificationStep(
                statePrep,
                markingOracle,
                markPhases[index],
                statePhases[index],
                register,
            );
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
    /// `MakeAmplifiedCircuit` with the round count replaced by a phase sequence.
    operation MakeFixedPointAmplifiedCircuit(
        preparation : Qubit[] => Unit is Adj,
        markingOracle : (Qubit[], Qubit) => Unit is Adj,
        markPhases : Double[],
        statePhases : Double[],
        numQubits : Int,
        measuredIndices : Int[],
    ) : Result[] {
        use register = Qubit[numQubits];
        ApplyFixedPointAmplitudeAmplification(
            preparation,
            markingOracle,
            markPhases,
            statePhases,
            register,
        );
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
