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
    ) : Unit {
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
    /// - `(Qubit, Qubit[]) => Unit`: A callable that takes a control qubit and an array of system qubits, and prepares the repeated controlled time evolution on the allocated qubits.
    function MakeRepControlledPauliExpOp(params : RepControlledPauliExpParams) : (Qubit, Qubit[]) => Unit {
        RepControlledPauliExp(params, _, _)
    }

    /// Performs repeated Controlled Time Evolution without resource-estimation caching.
    ///
    /// `RepControlledPauliExp` wraps every repetition in `BeginEstimateCaching`,
    /// which is not adjointable. Amplitude amplification reflects about the state
    /// preparation and therefore needs `Adjoint`, so it uses this variant instead
    /// and pays the full resource-estimation cost.
    /// # Parameters
    /// - `pauliExponents`: The Pauli strings of the Trotter step.
    /// - `pauliCoefficients`: The rotation angle of each Pauli string.
    /// - `repetitions`: How many times the Trotter step is repeated.
    /// - `control`: The control qubit for the operation.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    operation AdjointableRepControlledPauliExp(
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        control : Qubit,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        for _ in 1..repetitions {
            ControlledPauliExp(pauliExponents, pauliCoefficients, control, systems);
        }
    }

    /// A helper function to create an adjointable callable for repeated Controlled Time Evolution.
    ///
    /// The parameters are passed individually rather than as a
    /// `RepControlledPauliExpParams` struct so that the callable can be built
    /// from Python; the struct's `control` and `systems` index fields are not
    /// needed here because the qubits are supplied directly.
    /// # Parameters
    /// - `pauliExponents`: The Pauli strings of the Trotter step.
    /// - `pauliCoefficients`: The rotation angle of each Pauli string.
    /// - `repetitions`: How many times the Trotter step is repeated.
    /// # Returns
    /// - `(Qubit, Qubit[]) => Unit is Adj + Ctl`: An adjointable callable suitable for
    ///   amplitude amplification and coherent phase estimation.
    function MakeAdjointableRepControlledPauliExpOp(
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
    ) : (Qubit, Qubit[]) => Unit is Adj + Ctl {
        AdjointableRepControlledPauliExp(pauliExponents, pauliCoefficients, repetitions, _, _)
    }
}
