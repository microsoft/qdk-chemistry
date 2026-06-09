// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.IterativePhaseEstimation {

    import Std.Arrays.Subarray;

    /// A struct to hold parameters for iterative Quantum Phase Estimation (IQPE).
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledUnitary`: A function to perform repeated controlled unitary operations.
    /// - `accumulatePhase`: The phase to accumulate during the evolution.
    /// - `phaseQubit`: The index of the phase qubit (ancilla used for phase readout).
    /// - `systems`: An array of indices representing the system qubits.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled unitary (0 if none).
    struct IterativePhaseEstimationParams {
        statePrep : Qubit[] => Unit,
        repControlledUnitary : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
    }

    /// Runs the iterative Quantum Phase Estimation (IQPE) circuit based on the provided parameters.
    /// # Parameters
    /// - `params`: An `IterativePhaseEstimationParams` struct containing the parameters for IQPE.
    /// # Returns
    /// - `Result[]`: The result of measuring the control qubit after the IQPE circuit is executed.
    operation RunIQPE(params : IterativePhaseEstimationParams) : Result[] {
        use qs = Qubit[Length(params.systems) + 1 + params.numAncillaQubits];
        let phaseQubit = qs[params.phaseQubit];
        let systems = Subarray(params.systems, qs);
        let ancillas = if params.numAncillaQubits == 0 {
            []
        } else {
            qs[1 + Length(params.systems)..Length(qs) - 1]
        };
        let allTargets = systems + ancillas;

        params.statePrep(systems);

        within {
            H(phaseQubit);
        } apply {
            Rz(params.accumulatePhase, phaseQubit);
            params.repControlledUnitary(phaseQubit, allTargets);
        }
        ResetAll(allTargets);
        return [MResetZ(phaseQubit)];
    }

    /// Prepare iterative Quantum Phase Estimation (IQPE) circuit.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledUnitary`: A function to perform repeated controlled unitary operations.
    /// - `accumulatePhase`: The phase to accumulate during the evolution.
    /// - `phaseQubit`: The index of the phase qubit (ancilla used for phase readout).
    /// - `systems`: An array of indices representing the system qubits.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled unitary (0 if none).
    /// # Returns
    /// The result of measuring the phase qubit after the IQPE circuit is executed.
    operation MakeIQPECircuit(
        statePrep : Qubit[] => Unit,
        repControlledUnitary : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
    ) : Result[] {
        return RunIQPE(new IterativePhaseEstimationParams {
            statePrep = statePrep,
            repControlledUnitary = repControlledUnitary,
            accumulatePhase = accumulatePhase,
            phaseQubit = phaseQubit,
            systems = systems,
            numAncillaQubits = numAncillaQubits
        });
    }
}
