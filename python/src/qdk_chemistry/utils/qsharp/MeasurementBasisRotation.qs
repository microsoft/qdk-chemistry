// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.MeasurementBasisRotation {

    /// Applies measurement basis rotation and measures active qubits.
    /// For each qubit, the Pauli basis determines the pre-measurement rotation:
    /// - PauliI: qubit is not measured.
    /// - PauliX: apply H, then measure.
    /// - PauliY: apply Adjoint S then H, then measure.
    /// - PauliZ: measure directly (computational basis).
    ///
    /// # Parameters
    /// - `bases`: An array of Pauli values specifying the measurement basis
    ///   for each qubit.
    /// - `qubits`: The qubits to apply basis rotation and measurement on.
    ///
    /// # Returns
    /// An array of measurement results for the non-identity (active) qubits.
    operation MeasureInBasis(bases : Pauli[], qubits : Qubit[]) : Result[] {
        mutable results : Result[] = [];
        for idx in 0..Length(bases) - 1 {
            if bases[idx] == PauliX {
                H(qubits[idx]);
            } elif bases[idx] == PauliY {
                Adjoint S(qubits[idx]);
                H(qubits[idx]);
            }
            if bases[idx] != PauliI {
                set results += [M(qubits[idx])];
            }
        }
        return results;
    }

    /// Creates a measurement circuit that prepares a state and measures in
    /// the specified basis.
    ///
    /// # Parameters
    /// - `statePrep`: An operation that prepares the quantum state on the
    ///   given qubits.
    /// - `bases`: An array of Pauli values specifying the measurement basis.
    /// - `numQubits`: The total number of qubits to allocate.
    ///
    /// # Returns
    /// An array of measurement results from measuring in the specified basis.
    operation MakeMeasurementCircuit(
        statePrep : Qubit[] => Unit,
        bases : Pauli[],
        numQubits : Int,
    ) : Result[] {
        use qs = Qubit[numQubits];
        statePrep(qs);
        return MeasureInBasis(bases, qs);
    }
}
