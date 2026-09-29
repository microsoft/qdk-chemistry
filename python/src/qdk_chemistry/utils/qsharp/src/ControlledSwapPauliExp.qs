// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledSwapPauliExp {

    import QDKChemistry.Utils.PauliExp.SparseRepPauliExp;
    import QDKChemistry.Utils.PauliExp.SparseRepPauliExpParams;
    import Std.Arrays.Subarray;
    import Std.Math.AbsD;

    /// Controls sparse Pauli evolution using a CSWAP sandwich.
    ///
    /// For an n-qubit system, this uses one additional n-qubit vacuum register and
    /// two layers of n controlled-`SWAP` operations. The shared sparse evolution applies
    /// its prefix once, repeats the body, and applies its suffix once inside the sandwich.
    ///
    /// The full evolution must satisfy `U|0...0> = e^{i phi_0}|0...0>`. Its vacuum phase
    /// lands on the |0> branch; `R1(vacuumPhase)` corrects the relative phase to give
    /// controlled-`U` up to a global phase.
    ///
    /// # Parameters
    /// - `evolution`: Sparse evolution, including repetition and one-time boundary terms.
    /// - `vacuumPhase`: The phase `phi_0` of the full evolution on the vacuum.
    /// - `control`: The control qubit.
    /// - `systems`: System qubits indexed by the sparse evolution.
    operation RepControlledSwapPauliExp(
        evolution : SparseRepPauliExpParams,
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
            SparseRepPauliExp(evolution, vacuum);
        }
        // Skipped when negligible so a vacuum-annihilating evolution costs no extra rotation.
        if AbsD(vacuumPhase) > 1e-12 {
            R1(vacuumPhase, control);
        }
    }

    /// Parameters for the repeated CSWAP-sandwich controlled Pauli evolution.
    /// `control` and `systems` are register indices for circuit allocation.
    struct RepControlledSwapPauliExpParams {
        evolution : SparseRepPauliExpParams,
        vacuumPhase : Double,
        control : Int,
        systems : Int[],
    }

    /// Creates a CSWAP-sandwich circuit at the specified control and system indices.
    operation MakeRepControlledSwapPauliExpCircuit(
        evolution : SparseRepPauliExpParams,
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
            evolution,
            vacuumPhase,
            qs[control],
            Subarray(systems, qs)
        );
    }

    /// Returns an adjointable CSWAP-sandwich callable for supplied control and system qubits.
    function MakeRepControlledSwapPauliExpOp(params : RepControlledSwapPauliExpParams) : (Qubit, Qubit[]) => Unit is Adj {
        RepControlledSwapPauliExp(
            params.evolution,
            params.vacuumPhase,
            _,
            _
        )
    }
}
