// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.IterativePhaseEstimation {

    import Std.Arrays.Subarray;

    /// Creates a callable implementing phase-corrected repeated controlled evolution.
    /// # Parameters
    /// - `repControlledEvolution`: A function to perform repeated controlled evolution.
    /// - `accumulatePhase`: The phase to accumulate before controlled evolution.
    /// # Returns
    /// A callable that applies phase correction on the control qubit and then controlled evolution.
    function MakePhaseCorrectedControlledEvolutionOp(
        repControlledEvolution : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
    ) : (Qubit, Qubit[]) => Unit {
        (control, system) => {
            Rz(accumulatePhase, control);
            repControlledEvolution(control, system);
        }
    }

    /// Helper operation to render a phase-corrected controlled-evolution circuit artifact.
    /// # Parameters
    /// - `repControlledEvolution`: A function to perform repeated controlled evolution.
    /// - `accumulatePhase`: The phase to accumulate before controlled evolution.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of indices representing the system qubits.
    /// # Returns
    /// A single-element result array containing the control-qubit measurement.
    operation MakePhaseCorrectedControlledEvolutionCircuit(
        repControlledEvolution : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        control : Int,
        systems : Int[],
    ) : Result[] {
        use qs = Qubit[Length(systems) + 1];
        let control_q = qs[control];
        let system_q = Subarray(systems, qs);

        Rz(accumulatePhase, control_q);
        repControlledEvolution(control_q, system_q);
        ResetAll(system_q);
        return [MResetZ(control_q)];
    }

}
