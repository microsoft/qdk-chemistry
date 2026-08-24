// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledSwapPauliExp {

    import Std.Arrays.Subarray;
    import Std.Math.AbsD;
    import Std.ResourceEstimation.*;

    /// Performs a controlled time evolution for a set of Pauli exponentials using
    /// the "CSWAP-sandwich" construction.
    ///
    /// An internally allocated `vacuum` register (initialized to |0...0>) is conditionally
    /// swapped with the system register, the *uncontrolled* Pauli evolution is applied to the
    /// vacuum register, and the swap is uncomputed. The eigenphase accumulates on the |1>
    /// branch, matching the standard controlled-U convention, for the cost of a single layer
    /// of controlled-`SWAP` instead of controlling every gate of `Exp`.
    ///
    /// The `repetitions` loop lives *inside* the sandwich, so one layer of controlled-`SWAP`
    /// covers the whole repeated evolution.
    ///
    /// The evolution must leave the vacuum invariant, `U|0...0> = e^{i phi_0}|0...0>`, as a
    /// particle-conserving Hamiltonian does. That `phi_0` lands on the |0> branch and is passed
    /// in as `vacuumPhase`, applied to the control as an `R1` so the result is a genuine
    /// controlled-`U` up to a global phase.
    ///
    /// # Parameters
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `repetitions`: The number of times to repeat the (uncontrolled) evolution inside the sandwich.
    /// - `vacuumPhase`: The phase `phi_0` the repeated evolution imprints on the vacuum register.
    /// - `control`: The control qubit.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    /// # Returns
    /// - `Unit`: The operation prepares the controlled time evolution on the allocated qubits.
    operation RepControlledSwapPauliExp(
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        vacuumPhase : Double,
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj {
        use vacuum = Qubit[Length(systems)];
        within {
            for i in 0..Length(systems) - 1 {
                Controlled SWAP([control], (systems[i], vacuum[i]));
            }
        } apply {
            for _ in 1..repetitions {
                if BeginEstimateCaching("ControlledSwapPauliExp", 0) {
                    for idx in 0..Length(pauliExponents) - 1 {
                        Exp(pauliExponents[idx], -pauliCoefficients[idx], vacuum);
                    }
                    EndEstimateCaching();
                }
            }
        }
        // Skipped when negligible so a vacuum-annihilating evolution costs no extra rotation.
        if AbsD(vacuumPhase) > 1e-12 {
            R1(vacuumPhase, control);
        }
    }

    /// Parameters for the repeated CSWAP-sandwich controlled Pauli evolution.
    /// # Fields
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `repetitions`: The number of times to repeat the (uncontrolled) evolution inside the sandwich.
    /// - `vacuumPhase`: The phase the repeated evolution imprints on the vacuum register.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of integers representing the indices of the system qubits.
    struct RepControlledSwapPauliExpParams {
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        vacuumPhase : Double,
        control : Int,
        systems : Int[],
    }

    /// A helper operation to create a circuit for the repeated CSWAP-sandwich controlled
    /// time evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `repetitions`: The number of times to repeat the (uncontrolled) evolution inside the sandwich.
    /// - `vacuumPhase`: The phase the repeated evolution imprints on the vacuum register.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of integers representing the indices of the system qubits.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated controlled time evolution on the allocated qubits.
    operation MakeRepControlledSwapPauliExpCircuit(
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        vacuumPhase : Double,
        control : Int,
        systems : Int[]
    ) : Unit {
        // Size the register from the largest index across `control` and `systems` so that
        // non-contiguous layouts (e.g. control=2, systems=[3,4]) stay in range.
        mutable maxIndex = control;
        for idx in systems {
            if idx > maxIndex {
                set maxIndex = idx;
            }
        }

        use qs = Qubit[maxIndex + 1];
        RepControlledSwapPauliExp(
            pauliExponents,
            pauliCoefficients,
            repetitions,
            vacuumPhase,
            qs[control],
            Subarray(systems, qs)
        );
    }

    /// A helper function to create a callable for the repeated CSWAP-sandwich controlled
    /// time evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `RepControlledSwapPauliExpParams` struct containing the parameters for the operation.
    /// # Returns
    /// - `(Qubit, Qubit[]) => Unit`: A callable that takes a control qubit and an array of system qubits, and prepares the repeated controlled time evolution on the allocated qubits.
    function MakeRepControlledSwapPauliExpOp(params : RepControlledSwapPauliExpParams) : (Qubit, Qubit[]) => Unit is Adj {
        RepControlledSwapPauliExp(
            params.pauliExponents,
            params.pauliCoefficients,
            params.repetitions,
            params.vacuumPhase,
            _,
            _
        )
    }
}
