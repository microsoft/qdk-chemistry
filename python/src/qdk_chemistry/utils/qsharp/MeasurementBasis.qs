// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.MeasurementBasis {

    /// Measures each qubit in the specified Pauli basis and resets it to |0⟩.
    /// - PauliI: qubit is not measured.
    /// - PauliX/Y/Z: measure in that basis via MResetX/Y/Z.
    ///
    /// # Parameters
    /// - `bases`: An array of Pauli values specifying the measurement basis
    ///   for each qubit.
    /// - `qubits`: The qubits to measure.
    ///
    /// # Returns
    /// An array of measurement results for the non-identity (active) qubits.
    operation MeasureInBasis(bases : Pauli[], qubits : Qubit[]) : Result[] {
        mutable results : Result[] = [];
        for idx in 0..Length(bases) - 1 {
            if bases[idx] == PauliX {
                set results += [MResetX(qubits[idx])];
            } elif bases[idx] == PauliY {
                set results += [MResetY(qubits[idx])];
            } elif bases[idx] == PauliZ {
                set results += [MResetZ(qubits[idx])];
            }
        }
        return results;
    }

    /// Creates a measurement circuit that prepares a state and measures in
    /// the specified basis.
    ///
    /// # Parameters
    /// - `baseCircuit`: An operation that prepares the quantum state on the
    ///   given qubits.
    /// - `bases`: An array of Pauli values specifying the measurement basis.
    /// - `numQubits`: The total number of qubits to allocate.
    ///
    /// # Returns
    /// An array of measurement results from measuring in the specified basis.
    operation MakeMeasurementCircuit(
        baseCircuit : Qubit[] => Unit,
        bases : Pauli[],
        numQubits : Int,
    ) : Result[] {
        use qs = Qubit[numQubits];
        baseCircuit(qs);
        return MeasureInBasis(bases, qs);
    }
}
