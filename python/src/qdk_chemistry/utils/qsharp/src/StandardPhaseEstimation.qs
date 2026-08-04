// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.StandardPhaseEstimation {

    import Std.Arithmetic.ApplyIfGreaterL;
    import Std.Arithmetic.ApplyIfLessOrEqualL;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyQFT;
    import Std.Canon.ApplyToEachCA;
    import Std.Convert.IntAsBigInt;
    import Std.Core.Length;
    import QDKChemistry.Utils.MeasurementBasis.MeasureInBasis;

    /// # Summary
    /// Applies standard (QFT-based) quantum phase estimation in place.
    ///
    /// This is the measurement-free core of standard QPE: it leaves the phase
    /// register entangled with the system so the whole circuit can be used as the
    /// preparation `U` of a reflection-based algorithm such as amplitude
    /// amplification.  `MakeStandardQPECircuit` adds the measurement layer.
    ///
    /// `register` is laid out as `phase register ++ system qubits ++ block-encoding
    /// ancillas`.  After the inverse quantum Fourier transform the phase register
    /// is little-endian, so `register[0]` is the least significant phase bit.
    ///
    /// # Input
    /// ## controlledUnitary
    /// One adjointable callable per phase qubit, each already implementing the
    /// correct power of the unitary.  `controlledUnitary[k]` is controlled on
    /// `register[k]` and acts on the system plus block-encoding ancillas.
    operation ApplyStandardQPE(
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
    /// Builds standard QPE as a single-register adjointable callable.
    function MakeStandardQPEOp(
        statePrep : Qubit[] => Unit is Adj,
        controlledUnitary : ((Qubit, Qubit[]) => Unit is Adj)[],
        phaseQubitPrep : Qubit[] => Unit is Adj,
        numPhaseQubits : Int,
        numSystemQubits : Int,
    ) : Qubit[] => Unit is Adj {
        ApplyStandardQPE(
            statePrep,
            controlledUnitary,
            phaseQubitPrep,
            numPhaseQubits,
            numSystemQubits,
            _,
        )
    }

    /// # Summary
    /// Builds a standard QPE circuit with a caller-chosen final measurement
    /// (factory entry point).
    ///
    /// The circuit body is always `ApplyStandardQPE`; `measuredIndices` and
    /// `bases` alone decide what is read out, which is how the Python
    /// `measurement` setting selects between measuring the phase register,
    /// measuring the eigenvector in some Pauli basis, and measuring nothing.
    ///
    /// # Input
    /// ## measuredIndices
    /// Register indices to measure, in the order they should appear in the
    /// returned array.  Pass an empty array for a measurement-free circuit.
    /// ## bases
    /// One Pauli per entry of `measuredIndices`.  `PauliI` resets the qubit
    /// without recording a result.
    ///
    /// # Output
    /// One `Result` per non-identity entry of `bases`, in `measuredIndices` order.
    operation MakeStandardQPECircuit(
        statePrep : Qubit[] => Unit is Adj,
        controlledUnitary : ((Qubit, Qubit[]) => Unit is Adj)[],
        phaseQubitPrep : Qubit[] => Unit is Adj,
        numPhaseQubits : Int,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        measuredIndices : Int[],
        bases : Pauli[],
    ) : Result[] {
        if Length(bases) != Length(measuredIndices) {
            fail "Length of bases must match the number of measured indices.";
        }
        use register = Qubit[numPhaseQubits + numSystemQubits + numAncillaQubits];
        ApplyStandardQPE(
            statePrep,
            controlledUnitary,
            phaseQubitPrep,
            numPhaseQubits,
            numSystemQubits,
            register,
        );
        let results = MeasureInBasis(bases, Subarray(measuredIndices, register));
        ResetAll(register);
        return results;
    }

    //
    // Acceptance oracle
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
    /// Flips `target` on the good subspace of a QPE run: the phase register
    /// decodes to an accepted energy bin *and* every block-encoding signal
    /// ancilla is $|0\rangle$.
    ///
    /// Both conditions are required.  A nonzero signal ancilla means the
    /// block encoding did not project onto the signal block, so the phase
    /// register carries no eigenvalue information for that branch.
    ///
    /// The clean-ancilla condition is folded directly into the (already
    /// controllable) phase-index marking: conjugating the signal ancillas by `X`
    /// turns "all zero" into "all one", and the marking then runs *once*
    /// controlled on them.  This avoids a separate accepted-flag qubit, the
    /// second (uncompute) pass over the phase comparators, and the extra
    /// multiply-controlled flip that a compute/flag/uncompute structure would
    /// need.
    ///
    /// # Input
    /// ## numPhaseQubits
    /// Size of the leading phase register.  The register is little-endian
    /// (`register[0]` is the least significant bit), matching the layout
    /// produced by `ApplyStandardQPE` after its inverse quantum Fourier
    /// transform.
    /// ## signalAncillaIndices
    /// Indices into the trailing target register identifying the block-encoding
    /// ancillas.  Empty for encodings without ancillas (for example Trotter).
    operation ApplyAcceptanceMark(
        numPhaseQubits : Int,
        signalAncillaIndices : Int[],
        acceptedPhaseIndices : Int[],
        register : Qubit[],
        target : Qubit,
    ) : Unit is Adj {
        let phaseRegister = register[0..numPhaseQubits - 1];
        let signalAncillas = Subarray(signalAncillaIndices, register[numPhaseQubits...]);

        within {
            ApplyToEachCA(X, signalAncillas);
        } apply {
            Controlled ApplyAcceptedPhaseMark(
                signalAncillas,
                (phaseRegister, acceptedPhaseIndices, target),
            );
        }
    }

    /// # Summary
    /// Builds the marking oracle that identifies this module's good subspace.
    ///
    /// The returned callable has the generic marking-oracle signature consumed
    /// by `QDKChemistry.Utils.AmplitudeAmplification`, so the energy-window test
    /// can be swapped for any other predicate without touching that code.
    function MakeAcceptanceMarkerOp(
        numPhaseQubits : Int,
        signalAncillaIndices : Int[],
        acceptedPhaseIndices : Int[],
    ) : (Qubit[], Qubit) => Unit is Adj {
        ApplyAcceptanceMark(numPhaseQubits, signalAncillaIndices, acceptedPhaseIndices, _, _)
    }
}
