// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.PauliExp {

    import Std.Arrays.Subarray;
    import Std.ResourceEstimation.*;

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
    ) : Unit {
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
    ) : Unit {
        if IsResourceEstimating() {
            within {
                RepeatEstimates(params.repetitions);
            } apply {
                PauliExp(params.pauliExponents, params.pauliCoefficients, systems);
            }
        } else {
            for _ in 1..params.repetitions {
                PauliExp(params.pauliExponents, params.pauliCoefficients, systems);
            }
        }
    }

    operation SparsePauliExp(
        termOffsets : Int[],
        qubitIndices : Int[],
        paulis : Pauli[],
        pauliCoefficients : Double[],
        systems : Qubit[],
    ) : Unit {
        for term in 0..Length(pauliCoefficients) - 1 {
            let first = termOffsets[term];
            let last = termOffsets[term + 1] - 1;
            if first <= last {
                let range = first..last;
                let activeQubits = Subarray(qubitIndices[range], systems);
                Exp(paulis[range], -pauliCoefficients[term], activeQubits);
            }
        }
    }

    operation RepSparsePauliExp(
        termOffsets : Int[],
        qubitIndices : Int[],
        paulis : Pauli[],
        pauliCoefficients : Double[],
        repetitions : Int,
        systems : Qubit[],
    ) : Unit {
        if IsResourceEstimating() {
            within {
                RepeatEstimates(repetitions);
            } apply {
                SparsePauliExp(termOffsets, qubitIndices, paulis, pauliCoefficients, systems);
            }
        } else {
            for _ in 1..repetitions {
                SparsePauliExp(termOffsets, qubitIndices, paulis, pauliCoefficients, systems);
            }
        }
    }

    operation MakeRepSparsePauliExpCircuit(
        termOffsets : Int[],
        qubitIndices : Int[],
        paulis : Pauli[],
        pauliCoefficients : Double[],
        repetitions : Int,
        system : Int[],
    ) : Unit {
        if Length(system) == 0 {
            return ();
        }
        mutable maxIndex = system[0];
        for idx in 1..Length(system) - 1 {
            if system[idx] > maxIndex {
                set maxIndex = system[idx];
            }
        }
        use qs = Qubit[maxIndex + 1];
        RepSparsePauliExp(
            termOffsets, qubitIndices, paulis, pauliCoefficients, repetitions, Subarray(system, qs)
        );
    }

    function MakeRepSparsePauliExpOp(
        termOffsets : Int[],
        qubitIndices : Int[],
        paulis : Pauli[],
        pauliCoefficients : Double[],
        repetitions : Int,
    ) : Qubit[] => Unit {
        RepSparsePauliExp(termOffsets, qubitIndices, paulis, pauliCoefficients, repetitions, _)
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

    /// A helper function to create a callable for repeated Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `RepPauliExpParams` struct containing the parameters for the operation.
    /// # Returns
    /// - `Qubit[] => Unit`: A callable that takes an array of system qubits, and prepares the repeated time evolution on the allocated qubits.
    function MakeRepPauliExpOp(params : RepPauliExpParams) : Qubit[] => Unit {
        RepPauliExp(params, _)
    }
}
