// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledPauliExp {

    import QDKChemistry.Utils.CircuitComposition.MaxInt;
    import QDKChemistry.Utils.PauliExp.SegmentedSparseRepPauliExp;
    import QDKChemistry.Utils.PauliExp.SegmentedSparseRepPauliExpParams;
    import QDKChemistry.Utils.PauliExp.SparseRepPauliExp;
    import QDKChemistry.Utils.PauliExp.SparseRepPauliExpParams;
    import Std.Arrays.Subarray;

    /// Applies repeated sparse Pauli evolution controlled on a single qubit.
    ///
    /// This is a named operation rather than a closure so that callables produced by
    /// `MakeRepControlledPauliExpOp` stay resolvable by the Q# defunctionalizer, which
    /// runs when a caller such as `HadamardTest` is lowered to QIR.
    /// # Parameters
    /// - `params`: The sparse repeated Pauli evolution parameters.
    /// - `control`: The control qubit.
    /// - `systems`: The system qubits the evolution acts on.
    operation RepControlledPauliExp(
        params : SparseRepPauliExpParams,
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        Controlled SparseRepPauliExp([control], (params, systems));
    }

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
        RepControlledPauliExp(params, qs[control], Subarray(systems, qs));
    }

    /// Returns a single-control callable for repeated sparse Pauli evolution.
    function MakeRepControlledPauliExpOp(
        params : SparseRepPauliExpParams
    ) : ((Qubit, Qubit[]) => Unit is Adj + Ctl) {
        RepControlledPauliExp(params, _, _)
    }

    /// Applies segmented sparse Pauli evolution controlled on a single qubit.
    operation SegmentedRepControlledPauliExp(
        params : SegmentedSparseRepPauliExpParams,
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        Controlled SegmentedSparseRepPauliExp([control], (params, systems));
    }

    /// Allocates a register and applies segmented controlled Pauli evolution.
    operation MakeSegmentedRepControlledPauliExpCircuit(
        params : SegmentedSparseRepPauliExpParams,
        control : Int,
        systems : Int[]
    ) : Unit {
        use qs = Qubit[MaxInt([control] + systems) + 1];
        SegmentedRepControlledPauliExp(params, qs[control], Subarray(systems, qs));
    }

    /// Returns a single-control callable for segmented sparse Pauli evolution.
    function MakeSegmentedRepControlledPauliExpOp(
        params : SegmentedSparseRepPauliExpParams
    ) : ((Qubit, Qubit[]) => Unit is Adj + Ctl) {
        SegmentedRepControlledPauliExp(params, _, _)
    }
}
