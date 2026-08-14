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
    import Std.Arithmetic.ApplyIfGreaterOrEqualL;
    import Std.Arithmetic.ApplyIfLessOrEqualL;
    import Std.Convert.IntAsBigInt;
    import Std.Core.Length;
    import Std.Measurement.MResetEachZ;
    import Std.Measurement.MResetZ;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;

    /// # Summary
    /// Flips `target` when the little-endian `phase` register holds a value in the
    /// half-open interval [`lowerBound`, `upperBound`).
    ///
    /// # Description
    /// The `ApplyIf...L` comparisons take the classical constant first:
    /// `ApplyIfGreaterOrEqualL(action, c, x, target)` acts when `c >= x`, so each call below
    /// reads as the opposite of the bound it enforces.
    operation MarkPhaseRange(
        lowerBound : Int,
        upperBound : Int,
        phase : Qubit[],
        target : Qubit,
    ) : Unit is Adj {
        let phaseBinCount = 1 <<< Length(phase);
        if lowerBound < 0 or upperBound > phaseBinCount {
            fail $"Phase range [{lowerBound}, {upperBound}) does not fit a {Length(phase)}-qubit register.";
        }
        if lowerBound >= upperBound {
            fail $"Phase range [{lowerBound}, {upperBound}) is empty, so it would mark no phase at all.";
        }

        if lowerBound == 0 and upperBound == phaseBinCount {
            // Every value is in range.
            X(target);
        } elif upperBound == lowerBound + 1 {
            ApplyControlledOnInt(lowerBound, X, phase, target);
        } elif lowerBound == 0 {
            // phase >= 0 always holds, so only the upper bound has to be tested:
            // upperBound - 1 >= phase, that is phase < upperBound.
            ApplyIfGreaterOrEqualL(X, IntAsBigInt(upperBound - 1), phase, target);
        } elif upperBound == phaseBinCount {
            // phase <= phaseBinCount - 1 always holds, so only the lower bound has to be
            // tested: lowerBound <= phase.
            ApplyIfLessOrEqualL(X, IntAsBigInt(lowerBound), phase, target);
        } else {
            use aboveLower = Qubit();
            use belowUpper = Qubit();
            within {
                // lowerBound <= phase
                ApplyIfLessOrEqualL(X, IntAsBigInt(lowerBound), phase, aboveLower);
                // upperBound - 1 >= phase
                ApplyIfGreaterOrEqualL(X, IntAsBigInt(upperBound - 1), phase, belowUpper);
            } apply {
                Controlled X([aboveLower, belowUpper], target);
            }
        }
    }

    /// # Summary
    /// Flips `target` when `phase` lies in one of the half-open intervals
    /// [`lowerBounds[i]`, `upperBounds[i]`).
    ///
    /// # Description
    /// The bounds are read pairwise, so the two arrays must be the same length, and the
    /// intervals must be pairwise disjoint: a value covered twice is flipped twice and so
    /// left unmarked. More than one interval is needed when the accepted energies wrap
    /// around $\varphi = 1$.
    operation MarkAcceptedPhase(
        lowerBounds : Int[],
        upperBounds : Int[],
        phase : Qubit[],
        target : Qubit,
    ) : Unit is Adj {
        if Length(lowerBounds) != Length(upperBounds) {
            fail $"Got {Length(lowerBounds)} lower bounds and {Length(upperBounds)} upper bounds, but each phase range needs one of each.";
        }
        use inRange = Qubit();
        within {
            for index in 0..Length(lowerBounds) - 1 {
                MarkPhaseRange(lowerBounds[index], upperBounds[index], phase, inRange);
            }
        } apply {
            CNOT(inRange, target);
        }
    }

    /// # Summary
    /// Flips `target` when phase estimation of `system` lands in one of the accepted phase
    /// windows, then undoes the estimation.
    ///
    /// # Description
    /// `qpe` acts on `phase + system + signalAncillas` and must not prepare a state of its
    /// own, because `system` already holds the one being tested.
    ///
    /// Only the phase register is tested. Requiring the signal ancillas to be $|0\rangle$
    /// would project inside a walk eigenspace rather than select on energy. Testing the phase
    /// alone stays diagonal in the walk eigenbasis, because the accepted phases are symmetric
    /// under $\varphi \mapsto 1 - \varphi$ and so cover both branches of an eigenspace or neither.
    ///
    /// This reflects about the marked eigenspaces only when the estimation is exact, that is
    /// when every eigenphase of the state under test is a multiple of
    /// $2^{-\texttt{numPhaseQubits}}$. Off a bin the phase register comes back spread rather
    /// than to $|0\rangle$, and the released ancillas carry away part of the state.
    operation MarkQPEPhase(
        qpe : Qubit[] => Unit is Adj,
        numPhaseQubits : Int,
        numSignalAncillas : Int,
        lowerBounds : Int[],
        upperBounds : Int[],
        system : Qubit[],
        target : Qubit,
    ) : Unit is Adj {
        use phase = Qubit[numPhaseQubits];
        use signalAncillas = Qubit[numSignalAncillas];
        within {
            qpe(phase + system + signalAncillas);
        } apply {
            MarkAcceptedPhase(lowerBounds, upperBounds, phase, target);
        }
    }

    function MarkQPEPhaseOp(
        qpe : Qubit[] => Unit is Adj,
        numPhaseQubits : Int,
        numSignalAncillas : Int,
        lowerBounds : Int[],
        upperBounds : Int[],
    ) : (Qubit[], Qubit) => Unit is Adj {
        MarkQPEPhase(qpe, numPhaseQubits, numSignalAncillas, lowerBounds, upperBounds, _, _)
    }

    /// # Summary
    /// Applies the oracle to an all-zeros system register and measures the flag. An entry
    /// point so the oracle can be run, drawn and costed on its own; `MarkQPEPhaseOp` returns
    /// a callable and cannot be executed directly.
    operation MakeMarkedPhaseCircuit(
        qpe : Qubit[] => Unit is Adj,
        numPhaseQubits : Int,
        numSignalAncillas : Int,
        lowerBounds : Int[],
        upperBounds : Int[],
        numSystemQubits : Int,
    ) : Result[] {
        use system = Qubit[numSystemQubits];
        use flag = Qubit();
        MarkQPEPhase(qpe, numPhaseQubits, numSignalAncillas, lowerBounds, upperBounds, system, flag);
        let outcome = MResetZ(flag);
        ResetAll(system);
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
