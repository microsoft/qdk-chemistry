// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.PauliExp {

    import Std.Arrays.Subarray;

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
        use qs = Qubit[Length(system)];
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
