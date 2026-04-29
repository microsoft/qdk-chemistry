// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HadamardTest {

    import Std.Arrays.Subarray;

    /// Prepare a Hadamard test circuit.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledEvolution`: A function to perform repeated controlled evolution.
    /// - `testBasis`: Measurement basis for the control qubit. Supported values are `PauliX`, `PauliY`, and `PauliZ`.
    /// - `control`: The index of the control qubit in the allocated register.
    /// - `systems`: An array of indices representing the system qubits.
    /// # Returns
    /// A single-element result array containing the control-qubit measurement in the selected basis.
    operation HadamardTest(
        statePrep : Qubit[] => Unit,
        repControlledEvolution : (Qubit, Qubit[]) => Unit,
        testBasis : Pauli,
        control : Int,
        systems : Int[],
    ) : Result[] {
        use qs = Qubit[Length(systems) + 1];
        let control_q = qs[control];
        let system_q = Subarray(systems, qs);

        statePrep(system_q);

        H(control_q);
        repControlledEvolution(control_q, system_q);

        if (testBasis == PauliX) {
            H(control_q);
        } elif (testBasis == PauliY) {
            Adjoint S(control_q);
            H(control_q);
        } elif (testBasis == PauliZ) {
            // Direct Z-basis measurement: no additional rotation is required.
        } else {
            fail $"Invalid measurement basis: {testBasis}. Supported values are PauliX, PauliY, and PauliZ.";
        }
        ResetAll(system_q);
        return [MResetZ(control_q)];
    }
}
