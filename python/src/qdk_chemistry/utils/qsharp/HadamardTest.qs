// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HadamardTest {

    /// Prepare a Hadamard test circuit.
    ///
    /// Qubits are allocated and laid out in a fixed order: `[control, system, ancilla]`.
    /// The control qubit is always the first qubit, followed by the system qubits and then
    /// the ancilla qubits. This avoids any possibility of index collisions between the
    /// control, system, and ancilla registers.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledEvolution`: A function to perform repeated controlled evolution.
    /// - `testBasis`: Measurement basis for the control qubit. Supported values are `PauliX`, `PauliY`, and `PauliZ`.
    /// - `numSystemQubits`: Number of system qubits.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled evolution (0 if none).
    /// # Returns
    /// A single-element result array containing the control-qubit measurement in the selected basis.
    operation HadamardTest(
        statePrep : Qubit[] => Unit,
        repControlledEvolution : (Qubit, Qubit[]) => Unit,
        testBasis : Pauli,
        numSystemQubits : Int,
        numAncillaQubits : Int,
    ) : Result[] {
        use qs = Qubit[1 + numSystemQubits + numAncillaQubits];
        let control_q = qs[0];
        let system_q = qs[1..numSystemQubits];
        let ancillas = qs[1 + numSystemQubits..Length(qs) - 1];
        let allTargets = system_q + ancillas;

        statePrep(system_q);

        H(control_q);
        repControlledEvolution(control_q, allTargets);

        ResetAll(allTargets);
        if (testBasis == PauliX) {
            return [MResetX(control_q)];
        } elif (testBasis == PauliY) {
            return [MResetY(control_q)];
        } elif (testBasis == PauliZ) {
            return [MResetZ(control_q)];
        } else {
            fail $"Invalid measurement basis: {testBasis}. Supported values are PauliX, PauliY, and PauliZ.";
        }
    }
}
