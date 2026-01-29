// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
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
operation ControlledEvolution(
    pauliExponents : Pauli[][],
    pauliCoefficients : Double[],
    control : Int,
    system : Int[]
) : Unit {
    let numQubits = Length(system) + 1;
    use qs = Qubit[numQubits];
    for idx in 0..Length(pauliExponents) - 1 {
        let paulis = pauliExponents[idx];
        let coeff = pauliCoefficients[idx];
        Controlled Exp([qs[control]], (paulis, -coeff, Subarray(system, qs)));
    }
}

/// Performs repeated Controlled Time Evolution for a set of Pauli exponentials.
/// # Parameters
/// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
/// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
/// - `control`: The index of the control qubit.
/// - `system`: An array of integers representing the indices of the system qubits.
/// - `repetitions`: The number of times to repeat the controlled evolution.
/// # Returns
/// - `Unit`: The operation prepares the repeated controlled time evolution on the allocated qubits.
operation RepControlledEvolution(
    pauliExponents : Pauli[][],
    pauliCoefficients : Double[],
    control : Int,
    system : Int[],
    repetitions : Int,
) : Unit {
    for i in 1..repetitions {
        ControlledEvolution(pauliExponents, pauliCoefficients, control, system);
    }
}

/// Performs rescaled Controlled Time Evolution for a set of Pauli exponentials.
/// # Parameters
/// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
/// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
/// - `control`: The index of the control qubit.
/// - `system`: An array of integers representing the indices of the system qubits.
/// - `scaleFactor`: The factor by which to rescale the coefficients.
/// # Returns
/// - `Unit`: The operation prepares the rescaled controlled time evolution on the allocated qubits
operation RescaleControlledEvolution(
    pauliExponents : Pauli[][],
    pauliCoefficients : Double[],
    control : Int,
    system : Int[],
    scaleFactor : Double,
) : Unit {
    let rescaledCoefficients = Mapped(coeff -> coeff * scaleFactor, pauliCoefficients);
    ControlledEvolution(pauliExponents, rescaledCoefficients, control, system);
}
