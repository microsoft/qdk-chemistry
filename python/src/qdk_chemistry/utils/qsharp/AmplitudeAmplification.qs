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
/// The target can be supplied either as a state-preparation unitary
/// (`ReflectAboutTargetState`) or as a marking oracle that flips a flag qubit on
/// the good subspace (`ReflectAboutMarkedSubspace`).  Quantum phase estimation
/// plugs into the second form through `MakeQpeAcceptanceMarkerOp`, which marks
/// the branches whose phase register falls inside an accepted energy window and
/// whose block-encoding signal ancillas are all zero.
///
/// Reference: L. Lin, *Lecture Notes on Quantum Algorithms for Scientific
/// Computation*, arXiv:2201.08309, Chapter 2.
namespace QDKChemistry.Utils.AmplitudeAmplification {

    import Std.Arithmetic.ApplyIfGreaterL;
    import Std.Arithmetic.ApplyIfLessOrEqualL;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyQFT;
    import Std.Canon.ApplyToEachCA;
    import Std.Convert.IntAsBigInt;
    import Std.Core.Length;
    import Std.Intrinsic.R1;
    import Std.Math.PI;
    import Std.Measurement.MResetZ;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;

    //
    // Elementary reflections
    //

    /// # Summary
    /// Reflection about the all-zeros basis state.
    ///
    /// $$
    ///     2|0\rangle\langle 0| - I
    /// $$
    ///
    /// Note the sign convention: this is the *negative* of
    /// $I - 2|0\rangle\langle 0|$.  The two differ by a global phase, which is
    /// unobservable in the uncontrolled amplification loop but *is* observable
    /// when the reflection is applied as the target of a `Controlled` functor.
    operation ReflectAboutAllZeros(register : Qubit[]) : Unit is Adj + Ctl {
        Reflect(register);
    }

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
    operation ReflectAboutPreparedState(
        statePrep : Qubit[] => Unit is Adj,
        register : Qubit[],
    ) : Unit is Adj {
        within {
            Adjoint statePrep(register);
        } apply {
            ReflectAboutAllZeros(register);
        }
    }

    /// # Summary
    /// The general reflection unitary about a target state.
    ///
    /// $$
    ///     I - 2|t\rangle\langle t|,
    ///     \qquad |t\rangle = \text{targetPrep}|0\rangle
    /// $$
    ///
    /// Use this when the target is known as a circuit.  When the target is only
    /// known as a *predicate* — for example "the QPE phase register decodes to an
    /// energy below the cutoff" — use `ReflectAboutMarkedSubspace` instead; the
    /// amplification loop accepts either.
    operation ReflectAboutTargetState(
        targetPrep : Qubit[] => Unit is Adj,
        register : Qubit[],
    ) : Unit is Adj {
        ApplyPhaseToPreparedState(targetPrep, PI(), register);
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
    /// `qdk_chemistry.algorithms.amplitude_amplification.schedule`.
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
    /// `qdk_chemistry.algorithms.amplitude_amplification.schedule.fixed_point_phases`,
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
    // Phase-estimation acceptance oracle
    //

    /// # Summary
    /// Whether a sorted accepted set is the union of a prefix and a suffix of the
    /// phase-index range, together with the lengths of those two intervals.
    ///
    /// Energy windows produced by qubitization QPE are wrapped intervals of this
    /// shape because the walk phase enters through $\mu = \alpha\cos(2\pi\phi)$,
    /// so the highest energies sit at both ends of the index range.
    function AcceptedPhaseIntervalLengths(
        numBits : Int,
        acceptedPhaseIndices : Int[],
    ) : (Bool, Int, Int) {
        let dimension = 1 <<< numBits;
        let numAccepted = Length(acceptedPhaseIndices);
        mutable prefixLength = 0;
        for listIndex in 0..numAccepted - 1 {
            if acceptedPhaseIndices[listIndex] == prefixLength {
                set prefixLength += 1;
            }
        }

        let suffixLength = numAccepted - prefixLength;
        mutable isWrappedInterval = true;
        for suffixOffset in 0..suffixLength - 1 {
            if acceptedPhaseIndices[prefixLength + suffixOffset]
                != dimension - suffixLength + suffixOffset {
                set isWrappedInterval = false;
            }
        }
        return (isWrappedInterval, prefixLength, suffixLength);
    }

    /// # Summary
    /// Fails unless the accepted set is strictly increasing and fits `numBits`.
    ///
    /// Kept as a function so that `ApplyAcceptedPhaseMark` can still have a
    /// generated adjoint: mutable bindings are not allowed inside operation
    /// bodies that require one.
    function ValidateAcceptedPhaseIndices(numBits : Int, acceptedPhaseIndices : Int[]) : Unit {
        let dimension = 1 <<< numBits;
        mutable previous = -1;
        for phaseIndex in acceptedPhaseIndices {
            if phaseIndex <= previous or phaseIndex >= dimension {
                fail "Accepted phase indices must be sorted, unique, and fit the phase register.";
            }
            set previous = phaseIndex;
        }
    }

    /// # Summary
    /// Flips `target` if and only if the phase register holds an accepted index.
    ///
    /// `phaseRegister` is interpreted little-endian (`phaseRegister[0]` is the
    /// least significant bit).  Wrapped prefix/suffix windows use two
    /// linear-depth comparisons; arbitrary sets fall back to one
    /// multiply-controlled flip per accepted index.
    ///
    /// # Input
    /// ## acceptedPhaseIndices
    /// Strictly increasing indices, each smaller than $2^n$.
    operation ApplyAcceptedPhaseMark(
        phaseRegister : Qubit[],
        acceptedPhaseIndices : Int[],
        target : Qubit,
    ) : Unit is Adj + Ctl {
        let numBits = Length(phaseRegister);
        let dimension = 1 <<< numBits;
        let numAccepted = Length(acceptedPhaseIndices);

        ValidateAcceptedPhaseIndices(numBits, acceptedPhaseIndices);

        if numAccepted == dimension {
            X(target);
        } elif numAccepted > 0 {
            let (isWrappedInterval, prefixLength, suffixLength) =
                AcceptedPhaseIntervalLengths(numBits, acceptedPhaseIndices);

            if isWrappedInterval {
                if prefixLength > 0 {
                    ApplyIfGreaterL(X, IntAsBigInt(prefixLength), phaseRegister, target);
                }
                if suffixLength > 0 {
                    ApplyIfLessOrEqualL(
                        X,
                        IntAsBigInt(dimension - suffixLength),
                        phaseRegister,
                        target,
                    );
                }
            } else {
                for phaseIndex in acceptedPhaseIndices {
                    ApplyControlledOnInt(phaseIndex, X, phaseRegister, target);
                }
            }
        }
    }

    /// # Summary
    /// Flips `target` on the good subspace of an amplitude-amplified QPE:
    /// the phase register decodes to an accepted energy bin *and* every
    /// block-encoding signal ancilla is $|0\rangle$.
    ///
    /// Both conditions are required.  A nonzero signal ancilla means the
    /// block encoding did not project onto the signal block, so the phase
    /// register carries no eigenvalue information for that branch.
    ///
    /// # Input
    /// ## numPhaseQubits
    /// Size of the leading phase register.  The register is little-endian
    /// (`register[0]` is the least significant bit), matching the layout
    /// produced by `QDKChemistry.Utils.StandardPhaseEstimation` after its
    /// inverse quantum Fourier transform.
    /// ## signalAncillaIndices
    /// Indices into the trailing target register identifying the block-encoding
    /// ancillas.  Empty for encodings without ancillas (for example Trotter).
    operation ApplyQpeAcceptanceMark(
        numPhaseQubits : Int,
        signalAncillaIndices : Int[],
        acceptedPhaseIndices : Int[],
        register : Qubit[],
        target : Qubit,
    ) : Unit is Adj {
        let phaseRegister = register[0..numPhaseQubits - 1];
        let signalAncillas = Subarray(signalAncillaIndices, register[numPhaseQubits...]);

        use accepted = Qubit();
        within {
            ApplyAcceptedPhaseMark(phaseRegister, acceptedPhaseIndices, accepted);
            ApplyToEachCA(X, signalAncillas);
        } apply {
            Controlled X([accepted] + signalAncillas, target);
        }
    }

    /// # Summary
    /// Builds the marking oracle consumed by the amplification loops.
    ///
    /// The returned callable has the generic marking-oracle signature, so the
    /// energy-window test can be swapped for any other predicate without
    /// touching the amplification code.
    function MakeQpeAcceptanceMarkerOp(
        numPhaseQubits : Int,
        signalAncillaIndices : Int[],
        acceptedPhaseIndices : Int[],
    ) : (Qubit[], Qubit) => Unit is Adj {
        ApplyQpeAcceptanceMark(numPhaseQubits, signalAncillaIndices, acceptedPhaseIndices, _, _)
    }

    //
    // Amplified standard phase estimation
    //

    /// # Summary
    /// Coherent, adjointable standard QPE.
    ///
    /// Identical to the circuit built by
    /// `QDKChemistry.Utils.StandardPhaseEstimation.RunStandardQPE` but without the
    /// terminal measurement and reset, so it can serve as the state preparation
    /// $U_{\psi}$ of an amplification loop.
    ///
    /// `register` is laid out as `phase register ++ system qubits ++ block-encoding
    /// ancillas`.  After the inverse quantum Fourier transform the phase register
    /// is little-endian (`register[0]` is the least significant bit), matching
    /// `QDKChemistry.Utils.StandardPhaseEstimation.RunStandardQPE`.
    ///
    /// # Input
    /// ## controlledUnitary
    /// One adjointable callable per phase qubit, each already implementing the
    /// correct power of the walk operator.  Build these with
    /// `QDKChemistry.Utils.PrepSelPrep.MakeAdjointableControlledPSPWalkOp`; the
    /// resource-estimation-cached factories are not adjointable and cannot be
    /// used here.
    operation ApplyCoherentStandardQPE(
        statePrep : Qubit[] => Unit is Adj,
        controlledUnitary : ((Qubit, Qubit[]) => Unit is Adj)[],
        phaseQubitPrep : Qubit[] => Unit is Adj,
        numPhaseQubits : Int,
        numSystemQubits : Int,
        register : Qubit[],
    ) : Unit is Adj {
        let phaseRegister = register[0..numPhaseQubits - 1];
        let allTargets = register[numPhaseQubits...];
        let systems = allTargets[0..numSystemQubits - 1];

        statePrep(systems);
        phaseQubitPrep(phaseRegister);
        for phaseIndex in 0..numPhaseQubits - 1 {
            controlledUnitary[phaseIndex](phaseRegister[phaseIndex], allTargets);
        }
        Adjoint ApplyQFT(phaseRegister);
    }

    /// # Summary
    /// Builds the coherent QPE preparation as a single-register callable.
    function MakeCoherentStandardQPEOp(
        statePrep : Qubit[] => Unit is Adj,
        controlledUnitary : ((Qubit, Qubit[]) => Unit is Adj)[],
        phaseQubitPrep : Qubit[] => Unit is Adj,
        numPhaseQubits : Int,
        numSystemQubits : Int,
    ) : Qubit[] => Unit is Adj {
        ApplyCoherentStandardQPE(
            statePrep,
            controlledUnitary,
            phaseQubitPrep,
            numPhaseQubits,
            numSystemQubits,
            _,
        )
    }

    /// # Summary
    /// Applies amplitude-amplified standard QPE in place.
    ///
    /// Amplification changes *how often* the energy window is accepted; it does
    /// not change the accepted state itself, and it cannot repair a badly chosen
    /// window or insufficient phase resolution.
    operation ApplyAmplifiedStandardQPE(
        statePrep : Qubit[] => Unit is Adj,
        controlledUnitary : ((Qubit, Qubit[]) => Unit is Adj)[],
        phaseQubitPrep : Qubit[] => Unit is Adj,
        numPhaseQubits : Int,
        numSystemQubits : Int,
        signalAncillaIndices : Int[],
        acceptedPhaseIndices : Int[],
        amplificationRounds : Int,
        register : Qubit[],
    ) : Unit is Adj {
        let preparation = MakeCoherentStandardQPEOp(
            statePrep,
            controlledUnitary,
            phaseQubitPrep,
            numPhaseQubits,
            numSystemQubits,
        );
        let marker = MakeQpeAcceptanceMarkerOp(
            numPhaseQubits,
            signalAncillaIndices,
            acceptedPhaseIndices,
        );
        ApplyAmplitudeAmplification(preparation, marker, amplificationRounds, register);
    }

    //
    // Circuit entry points
    //
    // These are the operations handed to `Circuit`/`CircuitExecutor`.  They
    // deliberately contain no measurement-dependent classical control flow, so
    // they compile under the restricted target profiles used for QIR generation
    // and resource estimation.  Acceptance is decided classically in Python from
    // the returned bits; see
    // `qdk_chemistry.algorithms.amplitude_amplification.AmplifiedPhaseEstimation`.

    /// # Summary
    /// Builds and measures an amplitude-amplified circuit.
    ///
    /// Fully generic: `preparation` is any adjointable state preparation (for
    /// example the coherent QPE circuit built by `MakeCoherentStandardQPEOp`) and
    /// `markingOracle` is any adjointable predicate on the good subspace (for
    /// example the energy-window test built by `MakeQpeAcceptanceMarkerOp`).
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
    /// Builds and measures the coherent QPE circuit without any amplification.
    ///
    /// Provided so that the amplified and unamplified circuits differ only in the
    /// amplification loop, which makes the acceptance ratio between them a clean
    /// measurement of what amplification bought.
    operation MakeCoherentStandardQPECircuit(
        statePrep : Qubit[] => Unit is Adj,
        controlledUnitary : ((Qubit, Qubit[]) => Unit is Adj)[],
        phaseQubitPrep : Qubit[] => Unit is Adj,
        numPhaseQubits : Int,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        measuredIndices : Int[],
    ) : Result[] {
        use register = Qubit[numPhaseQubits + numSystemQubits + numAncillaQubits];
        ApplyCoherentStandardQPE(
            statePrep,
            controlledUnitary,
            phaseQubitPrep,
            numPhaseQubits,
            numSystemQubits,
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
