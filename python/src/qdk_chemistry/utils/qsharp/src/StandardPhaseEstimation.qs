// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.StandardPhaseEstimation {

    import Std.Arrays.Subarray;
    import Std.Canon.ApplyQFT;
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
}
