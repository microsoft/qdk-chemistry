// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.IterativePhaseEstimation {

    import Std.Arrays.Subarray;

    /// A struct to hold parameters for iterative Quantum Phase Estimation (IQPE).
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledEvolution`: A function to perform repeated controlled evolution.
    /// - `accumulatePhase`: The phase to accumulate during the evolution.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of indices representing the system qubits.
    struct IterativePhaseEstimationParams {
        statePrep : Qubit[] => Unit,
        repControlledEvolution : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        control : Int,
        system : Int[],
    }

    /// Runs the iterative Quantum Phase Estimation (IQPE) circuit based on the provided parameters.
    /// # Parameters
    /// - `params`: An `IterativePhaseEstimationParams` struct containing the parameters for IQPE.
    /// # Returns
    /// - `Result[]`: The result of measuring the control qubit after the IQPE circuit is executed.
    operation RunIQPE(params : IterativePhaseEstimationParams) : Result[] {
        use qs = Qubit[Length(params.system) + 1];
        let control = qs[params.control];
        let system = Subarray(params.system, qs);

        params.statePrep(system);

        within {
            H(control);
        } apply {
            Rz(params.accumulatePhase, control);
            params.repControlledEvolution(control, system);
        }
        ResetAll(system);
        return [MResetZ(control)];
    }

    /// Prepare iterative Quantum Phase Estimation (IQPE) circuit.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledEvolution`: A function to perform repeated controlled evolution.
    /// - `accumulatePhase`: The phase to accumulate during the evolution.
    /// - `control`: The index of the control qubit.
    /// - `system`: An array of indices representing the system qubits.
    /// # Returns
    /// The result of measuring the control qubit after the IQPE circuit is executed.
    operation MakeIQPECircuit(
        statePrep : Qubit[] => Unit,
        repControlledEvolution : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        control : Int,
        system : Int[],
    ) : Result[] {
        return RunIQPE(new IterativePhaseEstimationParams {
            statePrep = statePrep,
            repControlledEvolution = repControlledEvolution,
            accumulatePhase = accumulatePhase,
            control = control,
            system = system
        });
    }
}
