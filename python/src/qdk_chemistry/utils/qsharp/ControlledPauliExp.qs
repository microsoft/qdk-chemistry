// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledPauliExp {

    import Std.Math.ArcCos;
    import Std.Math.PI;
    import Std.Convert.IntAsDouble;
    import Std.Arrays.Subarray;
    import Std.Arrays.Mapped;

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
    ) : Unit {
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
    struct RepControlledPauliExpParams {
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
    }

    operation RepControlledPauliExp(
        params : RepControlledPauliExpParams,
        control : Qubit,
        systems : Qubit[],
    ) : Unit {
        for i in 1..params.repetitions {
            ControlledPauliExp(params.pauliExponents, params.pauliCoefficients, control, systems);
        }
    }


    operation MakeRepControlledPauliExpCircuit(
        params : RepControlledPauliExpParams,
        control : Int,
        system : Int[],
    ) : Unit {
        use qs = Qubit[Length(system) + 1];
        RepControlledPauliExp(params, qs[control], Subarray(system, qs));
    }

    function MakeRepControlledPauliExpOp(params : RepControlledPauliExpParams) : (Qubit, Qubit[]) => Unit {
        RepControlledPauliExp(params, _, _)
    }
}
