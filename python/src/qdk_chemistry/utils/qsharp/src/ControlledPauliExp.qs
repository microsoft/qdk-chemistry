// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledPauliExp {

    import QDKChemistry.Utils.CircuitComposition.MakeControlledOp;
    import QDKChemistry.Utils.CircuitComposition.MaxInt;
    import QDKChemistry.Utils.PauliExp.SparseRepPauliExp;
    import QDKChemistry.Utils.PauliExp.SparseRepPauliExpParams;
    import Std.Arrays.Subarray;

    /// A helper operation to create a circuit for repeated Controlled Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: The sparse repeated Pauli evolution parameters.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of integers representing the indices of the system qubits.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated controlled time evolution on the allocated qubits.
    operation MakeRepControlledPauliExpCircuit(
        params : SparseRepPauliExpParams,
        control : Int,
        systems : Int[]
    ) : Unit {
        use qs = Qubit[MaxInt([control] + systems) + 1];
        let controlledOp = MakeControlledOp(SparseRepPauliExp);
        controlledOp([qs[control]], (params, Subarray(systems, qs)));
    }

    /// Returns a single-control callable for repeated sparse Pauli evolution.
    function MakeRepControlledPauliExpOp(
        params : SparseRepPauliExpParams
    ) : ((Qubit, Qubit[]) => Unit is Adj + Ctl) {
        let controlledOp = MakeControlledOp(SparseRepPauliExp);
        (control, systems) => controlledOp([control], (params, systems))
    }
}
