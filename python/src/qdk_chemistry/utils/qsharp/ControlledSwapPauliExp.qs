// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledSwapPauliExp {

    import Std.Arrays.Subarray;
    import Std.ResourceEstimation.*;

    /// Performs a controlled time evolution for a set of Pauli exponentials using
    /// the "CSWAP-sandwich" construction.
    ///
    /// Instead of directly applying `Controlled Exp`, an internally allocated
    /// `vacuum` register (initialized to the |0...0> state) is conditionally
    /// swapped with the system register based on the control qubit. The
    /// *uncontrolled* Pauli evolution is then applied to the vacuum register and
    /// the swap is uncomputed. When the control qubit is |0>, the evolution acts
    /// on the vacuum reference |0...0> (leaving the system state untouched); when
    /// it is |1>, the system state is parked in the vacuum register and is
    /// evolved. The target eigenphase therefore accumulates on the |1> branch,
    /// matching the standard controlled-U convention.
    ///
    /// This trades the cost of controlling every gate of `Exp` for a single layer
    /// of controlled-`SWAP` gates, allowing the (repeated) evolution to be applied
    /// with uncontrolled gates only.
    ///
    /// # Parameters
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `repetitions`: The number of times to repeat the (uncontrolled) evolution inside the sandwich.
    /// - `control`: The control qubit.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    /// # Returns
    /// - `Unit`: The operation prepares the controlled time evolution on the allocated qubits.
    operation ControlledSwapPauliExp(
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        control : Qubit,
        systems : Qubit[]
    ) : Unit {
        use vacuum = Qubit[Length(systems)];
        within {
            for i in 0..Length(systems) - 1 {
                Controlled SWAP([control], (systems[i], vacuum[i]));
            }
        } apply {
            for _ in 1..repetitions {
                if BeginEstimateCaching("Exp", 0) {
                    for idx in 0..Length(pauliExponents) - 1 {
                        Exp(pauliExponents[idx], -pauliCoefficients[idx], vacuum);
                    }
                    EndEstimateCaching();
                }
            }
        }
        ResetAll(vacuum);
    }

    /// Parameters for the repeated CSWAP-sandwich controlled Pauli evolution.
    /// # Fields
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `repetitions`: The number of times to repeat the (uncontrolled) evolution inside the sandwich.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of integers representing the indices of the system qubits.
    struct ControlledSwapPauliExpParams {
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        control : Int,
        systems : Int[],
    }

    /// A helper operation to create a circuit for the repeated CSWAP-sandwich controlled
    /// time evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `repetitions`: The number of times to repeat the (uncontrolled) evolution inside the sandwich.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of integers representing the indices of the system qubits.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated controlled time evolution on the allocated qubits.
    operation MakeRepControlledSwapPauliExpCircuit(
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        control : Int,
        systems : Int[]
    ) : Unit {
        // Determine the maximum index across `control` and `systems` to size the qubit register
        // safely, so that non-contiguous indices (e.g. control=2, systems=[3,4]) stay in range.
        mutable maxIndex = control;
        for idx in systems {
            if idx > maxIndex {
                set maxIndex = idx;
            }
        }

        // Allocate enough qubits so that `control` and every index in `systems` are valid.
        use qs = Qubit[maxIndex + 1];
        ControlledSwapPauliExp(
            pauliExponents,
            pauliCoefficients,
            repetitions,
            qs[control],
            Subarray(systems, qs)
        );
    }

    /// A helper function to create a callable for the repeated CSWAP-sandwich controlled
    /// time evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `ControlledSwapPauliExpParams` struct containing the parameters for the operation.
    /// # Returns
    /// - `(Qubit, Qubit[]) => Unit`: A callable that takes a control qubit and an array of system qubits, and prepares the repeated controlled time evolution on the allocated qubits.
    function MakeRepControlledSwapPauliExpOp(params : ControlledSwapPauliExpParams) : (Qubit, Qubit[]) => Unit {
        ControlledSwapPauliExp(params.pauliExponents, params.pauliCoefficients, params.repetitions, _, _)
    }
}
