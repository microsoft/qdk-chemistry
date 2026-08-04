// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Generic amplitude amplification.
///
/// The module is deliberately factored into two swappable halves:
///
/// * a **reflection about the target**, which defines *what* is amplified, and
/// * an **amplification loop**, which defines *how hard* it is amplified.
///
/// The target is supplied as a marking oracle that flips a flag qubit on the
/// good subspace (`ReflectAboutMarkedSubspace`).  Nothing here knows what the
/// oracle tests; whoever defines the good subspace also owns the classical
/// decision of which shots landed in it.
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
    /// Applies the phase $e^{i\,\text{phase}}$ to the all-zeros basis state and
    /// leaves every other basis state unchanged.
    ///
    /// $$
    ///     I - (1 - e^{i\phi})|0\rangle\langle 0|
    /// $$
    ///
    /// With `phase = PI()` this is $I - 2|0\rangle\langle 0|$, the partial-phase
    /// generalization used by fixed-point amplitude amplification.
    operation ApplyPhaseToAllZeros(phase : Double, register : Qubit[]) : Unit is Adj + Ctl {
        let numQubits = Length(register);
        if numQubits == 0 {
            // The all-zeros state spans the whole (trivial) space, so the phase
            // is global and cannot be represented.
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
    /// Applies the phase $e^{i\,\text{phase}}$ to
    /// $|\psi\rangle = \text{statePrep}|0\rangle$.
    ///
    /// $$
    ///     I - (1 - e^{i\phi})|\psi\rangle\langle\psi|
    /// $$
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
    /// Reflection about a state given by a preparation unitary.
    ///
    /// $$
    ///     2|\psi\rangle\langle\psi| - I,
    ///     \qquad |\psi\rangle = \text{statePrep}|0\rangle
    /// $$
    ///
    /// This is the reflection that appears inside the Grover iterate.
    /// `Reflect` special-cases the degenerate sizes (global phase for $n = 0$, a
    /// single `Z` for $n = 1$) and, when controlled, folds the outer controls
    /// into the same multi-controlled `Z` rather than nesting `Controlled` on
    /// top of it.
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
    /// Applies the phase $e^{i\,\text{phase}}$ to the subspace selected by a
    /// marking oracle.
    ///
    /// `markingOracle` must flip its target qubit exactly on the good subspace
    /// and must leave the register otherwise unchanged, so that conjugating by it
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
    /// The general reflection unitary about a marked subspace.
    ///
    /// $$
    ///     I - 2\Pi_G
    /// $$
    ///
    /// where $\Pi_G$ projects onto the states on which `markingOracle` flips its
    /// target.  This is the pluggable half of the amplification loop.
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
    /// One Grover / amplitude-amplification iterate.
    ///
    /// $$
    ///     Q = -\,S_\psi S_G,
    ///     \qquad
    ///     S_G = I - 2\Pi_G,
    ///     \qquad
    ///     S_\psi = I - 2|\psi\rangle\langle\psi|
    /// $$
    ///
    /// Applied to $|\psi\rangle = \sin\vartheta\,|G\rangle +
    /// \cos\vartheta\,|B\rangle$ this rotates by $2\vartheta$ inside the
    /// two-dimensional invariant subspace, so `rounds` iterates give an
    /// acceptance probability of $\sin^2((2r+1)\vartheta)$.
    operation AmplitudeAmplificationStep(
        statePrep : Qubit[] => Unit is Adj,
        markingOracle : (Qubit[], Qubit) => Unit is Adj,
        register : Qubit[],
    ) : Unit is Adj {
        ReflectAboutMarkedSubspace(markingOracle, register);
        ReflectAboutPreparedState(statePrep, register);
    }

    /// # Summary
    /// The phase-matched (generalized) iterate used by fixed-point amplitude
    /// amplification.
    ///
    /// $$
    ///     G(\alpha,\beta)
    ///     = \left(I - (1 - e^{i\beta})|\psi\rangle\langle\psi|\right)
    ///       \left(I - (1 - e^{i\alpha})\Pi_G\right)
    /// $$
    ///
    /// Setting $\alpha = \beta = \pi$ recovers `AmplitudeAmplificationStep` up to
    /// a global phase of $-1$.
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
    ///
    /// The caller owns `register`; this operation neither measures nor resets.
    ///
    /// # Input
    /// ## statePrep
    /// Preparation of the initial state on the whole register.
    /// ## markingOracle
    /// Flips its target qubit on the good subspace.
    /// ## rounds
    /// Number of iterates.  Choosing this from a *lower* bound on the overlap
    /// overshoots the first maximum; see
    /// `qdk_chemistry.algorithms.amplitude_amplification.AmplitudeAmplification`.
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
    /// Supplying the Yoder–Low–Chuang phase sequence replaces the sinusoidal
    /// acceptance probability by a Chebyshev plateau: it climbs monotonically and
    /// then stays inside $[1-\delta^2, 1]$ for every larger overlap, so the
    /// overshoot cliff disappears at the cost of a constant-factor increase in
    /// queries.  The phases are generated by
    /// `AmplitudeAmplification.fixed_point_phases`,
    /// which returns them in exactly the `(markPhases, statePhases)` order and
    /// reflection convention used here.
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
    // These are the operations handed to `Circuit`/`CircuitExecutor`.  They
    // deliberately contain no measurement-dependent classical control flow, so
    // they compile under the restricted target profiles used for QIR generation
    // and resource estimation.  Which shots are good is decided classically by
    // whoever defined the marking oracle.

    /// # Summary
    /// Builds and measures an amplitude-amplified circuit.
    ///
    /// Fully generic: `preparation` is any adjointable state preparation and
    /// `markingOracle` is any adjointable predicate on the good subspace.
    ///
    /// # Input
    /// ## measuredIndices
    /// Register indices to measure, in the order they should appear in the
    /// returned array.
    ///
    /// # Output
    /// One `Result` per entry of `measuredIndices`, in that order.
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
    /// Builds and measures a fixed-point amplitude-amplified circuit.
    ///
    /// Identical to `MakeAmplifiedCircuit` except that the round count is
    /// replaced by the Yoder–Low–Chuang phase sequence, which removes the
    /// overshoot cliff entirely.
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
    /// Measures the requested register indices in the requested order.
    operation MeasureSelected(register : Qubit[], measuredIndices : Int[]) : Result[] {
        mutable results = [Zero, size = Length(measuredIndices)];
        for index in 0..Length(measuredIndices) - 1 {
            set results w/= index <- MResetZ(register[measuredIndices[index]]);
        }
        return results;
    }
}
