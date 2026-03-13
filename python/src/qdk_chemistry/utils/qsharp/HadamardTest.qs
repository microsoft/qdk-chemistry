// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HadamardTest {

    import Std.Arrays.Subarray;

    /// Prepare a Hadamard test circuit.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledEvolution`: A function to perform repeated controlled evolution.
    /// - `testBasis`: Measurement basis for the control qubit. Supported values are "X", "Y", and "Z".
    /// - `control`: The index of the control qubit in the allocated register.
    /// - `systems`: An array of indices representing the system qubits.
    /// # Returns
    /// A single-element result array containing the control-qubit measurement in the selected basis.
    operation MakeHadamardCircuit(
        statePrep : Qubit[] => Unit,
        repControlledEvolution : (Qubit, Qubit[]) => Unit,
        testBasis : String,
        control : Int,
        systems : Int[],
    ) : Result[] {
        use qs = Qubit[Length(systems) + 1];
        let control_q = qs[control];
        let system_q = Subarray(systems, qs);

        statePrep(system_q);

        H(control_q);
        repControlledEvolution(control_q, system_q);

        mutable basis = PauliX;
        if (testBasis == "X") {
            // Do nothing.
        } else if (testBasis == "Y") {
            set basis = PauliY;
        } else if (testBasis == "Z") {
            set basis = PauliZ;
        } else {
            fail "Invalid measurement basis.";
        }
        let result = Measure([basis], [control_q]);
        ResetAll(qs);
        return [result];
    }
}
