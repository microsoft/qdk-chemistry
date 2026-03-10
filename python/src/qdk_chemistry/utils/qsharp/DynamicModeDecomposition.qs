// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.DynamicModeDecomposition {

    import Std.Arrays.Subarray;

    /// Prepare observable dynamic mode decomposition (ODMD) circuit.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledEvolution`: A function to perform repeated controlled evolution.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of indices representing the system qubits.
    /// # Returns
    /// The result of measuring the control qubit after the IQPE circuit is executed.
    operation MakeODMDCircuit(
        statePrep : Qubit[] => Unit,
        repControlledEvolution : (Qubit, Qubit[]) => Unit,
        control : Int,
        systems : Int[],
    ) : Result[] {
        use qs = Qubit[Length(systems) + 1];
        let control = qs[control];
        let system = Subarray(systems, qs);

        statePrep(system);

        within {
            H(control);
        } apply {
            repControlledEvolution(control, system);
        }
        ResetAll(system);
        return [MResetZ(control)];
    }
}
