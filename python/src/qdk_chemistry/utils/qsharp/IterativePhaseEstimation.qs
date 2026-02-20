// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.IterativePhaseEstimation {

    import Std.Math.ArcCos;
    import Std.Math.PI;
    import Std.Convert.IntAsDouble;
    import Std.Arrays.Subarray;
    import Std.Arrays.Mapped;

    /// Prepare iterative Quantum Phase Estimation (IQPE) circuit.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledEvolution`: A function to perform repeated controlled evolution.
    /// - `accumulatePhase`: The phase to accumulate during the evolution.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of indices representing the system qubits.
    /// # Returns
    /// The result of measuring the control qubit after the IQPE circuit is executed.
    operation MakeIQPECircuit(
        statePrep : Qubit[] => Unit,
        repControlledEvolution : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        control : Int,
        systems : Int[],
    ) : Result[] {
        use qs = Qubit[Length(systems) + 1];
        let control = qs[control];
        let system = Subarray(systems, qs);

        statePrep(system);

        within {
            H(control);
        } apply {
            Rz(accumulatePhase, control);
            repControlledEvolution(control, system);
        }
        ResetAll(system);
        return [MResetZ(control)]
    }
}
