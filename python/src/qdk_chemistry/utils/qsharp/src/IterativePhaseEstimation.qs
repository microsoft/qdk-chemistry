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
    /// - `prepareSharedOp`: Prepares the shared register around the controlled unitary.
    /// - `numSharedAncillas`: Size of the shared register, placed at the end of the targets.
    struct IterativePhaseEstimationParams {
        statePrep : Qubit[] => Unit,
        repControlledUnitary : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
        prepareSharedOp : Qubit[] => Unit is Adj + Ctl,
        numSharedAncillas : Int,
    }

    /// Runs the iterative Quantum Phase Estimation (IQPE) circuit based on the provided parameters.
    /// # Parameters
    /// - `params`: An `IterativePhaseEstimationParams` struct containing the parameters for IQPE.
    /// # Returns
    /// - `Result[]`: The result of measuring the phase qubit after the IQPE circuit is executed.
    operation RunIQPE(params : IterativePhaseEstimationParams) : Result[] {
        use qs = Qubit[Length(params.systems) + 1 + params.numAncillaQubits + params.numSharedAncillas];
        let phaseQubit = qs[params.phaseQubit];
        let systems = Subarray(params.systems, qs);
        let ancillaStart = 1 + Length(params.systems);
        let ancillas = qs[ancillaStart..ancillaStart + params.numAncillaQubits - 1];
        let shared = qs[ancillaStart + params.numAncillaQubits...];
        let allTargets = systems + ancillas + shared;

        params.statePrep(systems);

        within {
            H(phaseQubit);
        } apply {
            Rz(params.accumulatePhase, phaseQubit);
            within {
                params.prepareSharedOp(shared);
            } apply {
                params.repControlledUnitary(phaseQubit, allTargets);
            }
        }
        let result = MResetZ(phaseQubit);
        ResetAll(allTargets);
        return [result];
    }

    /// Prepare iterative Quantum Phase Estimation (IQPE) circuit.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledUnitary`: A function to perform repeated controlled unitary operations.
    /// - `accumulatePhase`: The phase to accumulate during the evolution.
    /// - `phaseQubit`: The index of the phase qubit (ancilla used for phase readout).
    /// - `systems`: An array of indices representing the system qubits.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled unitary (0 if none).
    /// - `prepareSharedOp`: Prepares the shared register around the controlled unitary.
    /// - `numSharedAncillas`: Size of the shared register, placed at the end of the targets.
    /// # Returns
    /// The result of measuring the phase qubit after the IQPE circuit is executed.
    operation MakeIQPECircuit(
        statePrep : Qubit[] => Unit,
        repControlledUnitary : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
        prepareSharedOp : Qubit[] => Unit is Adj + Ctl,
        numSharedAncillas : Int,
    ) : Result[] {
        return RunIQPE(new IterativePhaseEstimationParams {
            statePrep = statePrep,
            repControlledUnitary = repControlledUnitary,
            accumulatePhase = accumulatePhase,
            phaseQubit = phaseQubit,
            systems = systems,
            numAncillaQubits = numAncillaQubits,
            prepareSharedOp = prepareSharedOp,
            numSharedAncillas = numSharedAncillas
        });
    }
}
