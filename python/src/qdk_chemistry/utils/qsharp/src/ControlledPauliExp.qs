// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledPauliExp {

    import Std.Arrays.Subarray;
    import Std.ResourceEstimation.*;

    /// Performs Controlled Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `control`: The index of the control qubit.
    /// - `system`: An array of integers representing the indices of the system qubits.
    /// # Returns
    /// - `Unit`: The operation prepares the controlled time evolution on the allocated qubits.
    operation ControlledPauliExp(
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        for idx in 0..Length(pauliExponents) - 1 {
            let paulis = pauliExponents[idx];
            let coeff = pauliCoefficients[idx];
            Controlled Exp([control], (paulis, -coeff, systems));
        }
    }


    /// Performs repeated Controlled Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `repetitions`: The number of times to repeat the controlled evolution.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of integers representing the indices of the system qubits.
    struct RepControlledPauliExpParams {
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        control : Int,
        systems : Int[],
    }

    /// Performs repeated Controlled Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `RepControlledPauliExpParams` struct containing the parameters for the operation.
    /// - `control`: The control qubit for the operation.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated controlled time evolution on the allocated qubits.
    operation RepControlledPauliExp(
        params : RepControlledPauliExpParams,
        control : Qubit,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        for i in 1..params.repetitions {
            if BeginEstimateCaching("ControlledPauliExp", 0) {
                ControlledPauliExp(params.pauliExponents, params.pauliCoefficients, control, systems);
                EndEstimateCaching();
            }
        }
    }

    /// A helper operation to create a circuit for repeated Controlled Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `repetitions`: The number of times to repeat the controlled evolution.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of integers representing the indices of the system qubits.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated controlled time evolution on the allocated qubits.
    operation MakeRepControlledPauliExpCircuit(
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        control : Int,
        systems : Int[]
    ) : Unit {
        use qs = Qubit[Length(systems) + 1];
        RepControlledPauliExp(
            new RepControlledPauliExpParams { pauliExponents = pauliExponents, pauliCoefficients = pauliCoefficients, repetitions = repetitions, control = control, systems = systems },
            qs[control],
            Subarray(systems, qs)
        );
    }

    /// A helper function to create a callable for repeated Controlled Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `RepControlledPauliExpParams` struct containing the parameters for the operation.
    /// # Returns
    /// - `(Qubit, Qubit[]) => Unit is Adj + Ctl`: A callable that takes a control qubit and an array of system qubits, and prepares the repeated controlled time evolution on the allocated qubits.
    function MakeRepControlledPauliExpOp(params : RepControlledPauliExpParams) : (Qubit, Qubit[]) => Unit is Adj + Ctl {
        RepControlledPauliExp(params, _, _)
    }

    /// Sparse form of `RepControlledPauliExpParams`.
    ///
    /// `pauliExponents` in the dense form carries one entry per system qubit for
    /// every term, which is O(terms * qubits). Jordan-Wigner terms are far from
    /// dense, so here each term instead lists only its non-identity positions:
    /// `pauliIndices[t]` indexes into `systems` and `pauliOps[t]` holds the
    /// matching axis. A term with no entries is the identity term, which still
    /// contributes a phase once controlled.
    struct SparseRepControlledPauliExpParams {
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        control : Int,
        systems : Int[],
    }

    /// Performs Controlled Time Evolution for a sparsely encoded set of Pauli exponentials.
    /// # Parameters
    /// - `pauliIndices`: For each term, the positions in `systems` carrying a non-identity Pauli.
    /// - `pauliOps`: For each term, the Pauli axis at each position in `pauliIndices`.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `control`: The control qubit.
    /// - `systems`: An array of qubits representing the system.
    /// # Returns
    /// - `Unit`: The operation prepares the controlled time evolution on the allocated qubits.
    operation SparseControlledPauliExp(
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        // Terms differ in weight and angle, so RepeatEstimates cannot be used here:
        // charging one term's cost Length(pauliOps) times mis-counts both qubits and
        // runtime. Only the outer repetition loop is uniform enough to collapse.
        for idx in 0..Length(pauliOps) - 1 {
            Controlled Exp(
                [control],
                (pauliOps[idx], -pauliCoefficients[idx], Subarray(pauliIndices[idx], systems))
            );
        }
    }

    /// Performs repeated Controlled Time Evolution for a sparsely encoded set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `SparseRepControlledPauliExpParams` struct containing the parameters for the operation.
    /// - `control`: The control qubit for the operation.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated controlled time evolution on the allocated qubits.
    operation SparseRepControlledPauliExp(
        params : SparseRepControlledPauliExpParams,
        control : Qubit,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        // RepeatEstimates is a no-op outside the estimator, so the loop is still
        // needed for simulation and circuit generation.
        if IsResourceEstimating() {
            within {
                RepeatEstimates(params.repetitions);
            } apply {
                SparseControlledPauliExp(
                    params.pauliIndices,
                    params.pauliOps,
                    params.pauliCoefficients,
                    control,
                    systems
                );
            }
        } else {
            for _ in 1..params.repetitions {
                SparseControlledPauliExp(
                    params.pauliIndices,
                    params.pauliOps,
                    params.pauliCoefficients,
                    control,
                    systems
                );
            }
        }
    }

    /// A helper operation to create a circuit for repeated sparse Controlled Time Evolution.
    /// # Parameters
    /// - `pauliIndices`: For each term, the positions in `systems` carrying a non-identity Pauli.
    /// - `pauliOps`: For each term, the Pauli axis at each position in `pauliIndices`.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `repetitions`: The number of times to repeat the controlled evolution.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of integers representing the indices of the system qubits.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated controlled time evolution on the allocated qubits.
    operation MakeSparseRepControlledPauliExpCircuit(
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        control : Int,
        systems : Int[]
    ) : Unit {
        use qs = Qubit[Length(systems) + 1];
        SparseRepControlledPauliExp(
            new SparseRepControlledPauliExpParams {
                pauliIndices = pauliIndices,
                pauliOps = pauliOps,
                pauliCoefficients = pauliCoefficients,
                repetitions = repetitions,
                control = control,
                systems = systems
            },
            qs[control],
            Subarray(systems, qs)
        );
    }

    /// A helper function to create a callable for repeated sparse Controlled Time Evolution.
    /// # Parameters
    /// - `params`: A `SparseRepControlledPauliExpParams` struct containing the parameters for the operation.
    /// # Returns
    /// - `(Qubit, Qubit[]) => Unit is Adj + Ctl`: A callable that takes a control qubit and an array of system qubits.
    function MakeSparseRepControlledPauliExpOp(
        params : SparseRepControlledPauliExpParams
    ) : (Qubit, Qubit[]) => Unit is Adj + Ctl {
        SparseRepControlledPauliExp(params, _, _)
    }
}
