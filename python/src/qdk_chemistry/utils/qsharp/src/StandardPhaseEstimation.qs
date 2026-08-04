// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.StandardPhaseEstimation {

    import Std.Arrays.Subarray;
    import Std.Canon.ApplyQFT;
    import QDKChemistry.Utils.MeasurementBasis.MeasureInBasis;

    /// # Summary
    /// Applies standard (QFT-based) QPE in place, without measuring.
    ///
    /// `register` is `phase ++ system ++ block-encoding ancillas`. After the
    /// inverse QFT the phase register is little-endian.
    ///
    /// # Input
    /// ## controlledUnitary
    /// One adjointable callable per phase qubit, already raised to the correct
    /// power; `controlledUnitary[k]` is controlled on `register[k]`.
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
    /// `ApplyStandardQPE` plus a caller-chosen measurement (factory entry point).
    ///
    /// # Input
    /// ## measuredIndices
    /// Register indices to measure, in output order. Empty measures nothing.
    /// ## bases
    /// One Pauli per measured index; `PauliI` resets without recording.
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
