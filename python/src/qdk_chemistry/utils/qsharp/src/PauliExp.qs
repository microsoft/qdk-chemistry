// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.PauliExp {

    import Std.Arrays.Subarray;
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.ResourceEstimation.RepeatEstimates;

    /// Performs Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    /// # Returns
    /// - `Unit`: The operation prepares the time evolution on the allocated qubits.
    operation PauliExp(
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        for idx in 0..Length(pauliExponents) - 1 {
            let paulis = pauliExponents[idx];
            let coeff = pauliCoefficients[idx];
            Exp(paulis, -coeff, systems);
        }
    }


    /// Performs repeated Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `repetitions`: The number of times to repeat the evolution.
    struct RepPauliExpParams {
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
    }

    /// Performs repeated Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `RepPauliExpParams` struct containing the parameters for the operation.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated time evolution on the allocated qubits.
    operation RepPauliExp(
        params : RepPauliExpParams,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        for i in 1..params.repetitions {
            PauliExp(params.pauliExponents, params.pauliCoefficients, systems);
        }
    }

    /// A helper operation to create a circuit for repeated Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `RepPauliExpParams` struct containing the parameters for the operation.
    /// - `system`: An array of integers representing the indices of the system qubits.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated time evolution on the allocated qubits.
    operation MakeRepPauliExpCircuit(
        params : RepPauliExpParams,
        system : Int[],
    ) : Unit {
        // If no system indices are provided, there is nothing to do.
        if Length(system) == 0 {
            return ();
        }

        // Determine the maximum index in the system array to size the qubit register safely.
        mutable maxIndex = system[0];
        for idx in 1..Length(system) - 1 {
            let current = system[idx];
            if current > maxIndex {
                set maxIndex = current;
            }
        }

        // Allocate enough qubits so that all indices in `system` are valid.
        use qs = Qubit[maxIndex + 1];
        RepPauliExp(params, Subarray(system, qs));
    }

    /// Uncontrolled entry point for `RepPauliExp`.
    ///
    /// Declared without functors on purpose. Callables built by partially applying an
    /// `Adj + Ctl` operation cannot be resolved by the Q# defunctionalizer once they are
    /// stored in a plain `Qubit[] => Unit` slot (such as the arguments of
    /// `CircuitComposition.ApplySequential` or `MeasurementBasis.MakeMeasurementCircuit`),
    /// which makes lowering those compositions to QIR fail. Forwarding through a plain
    /// operation keeps the composed circuits statically resolvable.
    operation ApplyRepPauliExp(params : RepPauliExpParams, systems : Qubit[]) : Unit {
        RepPauliExp(params, systems);
    }

    /// A helper function to create a callable for repeated Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `RepPauliExpParams` struct containing the parameters for the operation.
    /// # Returns
    /// - `Qubit[] => Unit`: A callable that takes an array of system qubits, and prepares the repeated time evolution on the allocated qubits.
    function MakeRepPauliExpOp(params : RepPauliExpParams) : Qubit[] => Unit {
        ApplyRepPauliExp(params, _)
    }

    /// Returns an `Adj + Ctl` callable for repeated dense Time Evolution.
    internal function MakeRepPauliExpAdjCtlOp(params : RepPauliExpParams) : (Qubit[] => Unit is Adj + Ctl) {
        RepPauliExp(params, _)
    }

    /// Sparse form of `RepPauliExpParams`.
    ///
    /// The dense `pauliExponents` carries one Pauli per system qubit for every term,
    /// which is O(terms * qubits). Jordan-Wigner terms are far from dense, so here each
    /// term instead lists only its non-identity positions: `pauliIndices[t]` indexes
    /// into `systems` and `pauliOps[t]` holds the matching axis. A term with no entries
    /// is the identity term.
    struct SparseRepPauliExpParams {
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
    }

    /// Performs Time Evolution for a sparsely encoded set of Pauli exponentials.
    /// # Parameters
    /// - `pauliIndices`: For each term, the positions in `systems` carrying a non-identity Pauli.
    /// - `pauliOps`: For each term, the Pauli axis at each position in `pauliIndices`.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    operation SparsePauliExp(
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        if Length(pauliIndices) != Length(pauliCoefficients) or Length(pauliOps) != Length(pauliCoefficients) {
            fail "SparsePauliExp: pauliIndices, pauliOps, and pauliCoefficients must have the same length.";
        }

        for idx in 0..Length(pauliCoefficients) - 1 {
            // `Exp` takes the opposite sign to the container's exp(-i theta P) convention.
            Exp(pauliOps[idx], -pauliCoefficients[idx], Subarray(pauliIndices[idx], systems));
        }
    }

    /// Performs repeated Time Evolution for a sparsely encoded set of Pauli exponentials.
    operation SparseRepPauliExp(
        params : SparseRepPauliExpParams,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {

        if IsResourceEstimating() {
            within {
                RepeatEstimates(params.repetitions);
            } apply {
                SparsePauliExp(
                    params.pauliIndices,
                    params.pauliOps,
                    params.pauliCoefficients,
                    systems
                );
            }
        } else {
            for _ in 1..params.repetitions {
                SparsePauliExp(
                    params.pauliIndices,
                    params.pauliOps,
                    params.pauliCoefficients,
                    systems
                );
            }
        }
    }

    /// A helper operation to create a circuit for repeated sparse Time Evolution.
    operation MakeSparseRepPauliExpCircuit(
        params : SparseRepPauliExpParams,
        system : Int[],
    ) : Unit {
        // If no system indices are provided, there is nothing to do.
        if Length(system) == 0 {
            return ();
        }

        // Determine the maximum index in the system array to size the qubit register safely.
        mutable maxIndex = system[0];
        for idx in 1..Length(system) - 1 {
            let current = system[idx];
            if current > maxIndex {
                set maxIndex = current;
            }
        }

        // Allocate enough qubits so that all indices in `system` are valid.
        use qs = Qubit[maxIndex + 1];
        SparseRepPauliExp(params, Subarray(system, qs));
    }

    /// Uncontrolled entry point for `SparseRepPauliExp`; see `ApplyRepPauliExp` for why
    /// this forwarding operation carries no functors.
    operation ApplySparseRepPauliExp(params : SparseRepPauliExpParams, systems : Qubit[]) : Unit {
        SparseRepPauliExp(params, systems);
    }

    /// A helper function to create a callable for repeated sparse Time Evolution.
    function MakeSparseRepPauliExpOp(params : SparseRepPauliExpParams) : Qubit[] => Unit {
        ApplySparseRepPauliExp(params, _)
    }

    /// Returns an `Adj + Ctl` callable for repeated sparse Time Evolution.
    internal function MakeSparseRepPauliExpAdjCtlOp(
        params : SparseRepPauliExpParams
    ) : (Qubit[] => Unit is Adj + Ctl) {
        SparseRepPauliExp(params, _)
    }
}
