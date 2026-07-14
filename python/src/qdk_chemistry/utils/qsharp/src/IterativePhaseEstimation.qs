// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.IterativePhaseEstimation {

    import Std.Arrays.Subarray;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Math.Ceiling;
    import Std.ResourceEstimation.EnableMemoryComputeArchitecture;
    import Std.ResourceEstimation.LeastRecentlyUsed;

    /// A struct to hold parameters for iterative Quantum Phase Estimation (IQPE).
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledUnitary`: A function to perform repeated controlled unitary operations.
    /// - `accumulatePhase`: The phase to accumulate during the evolution.
    /// - `phaseQubit`: The index of the phase qubit (ancilla used for phase readout).
    /// - `systems`: An array of indices representing the system qubits.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled unitary (0 if none).
    /// - `ancillaPrep`: A function to prepare persistent block-encoding ancillas (e.g., phase gradient state).
    struct IterativePhaseEstimationParams {
        statePrep : Qubit[] => Unit,
        repControlledUnitary : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
    }

    /// Runs the iterative Quantum Phase Estimation (IQPE) circuit based on the provided parameters.
    /// # Parameters
    /// - `params`: An `IterativePhaseEstimationParams` struct containing the parameters for IQPE.
    /// # Returns
    /// - `Result[]`: The result of measuring the phase qubit after the IQPE circuit is executed.
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
        params.ancillaPrep(ancillas);

        within {
            H(phaseQubit);
        } apply {
            Rz(params.accumulatePhase, phaseQubit);
            params.repControlledUnitary(phaseQubit, allTargets);
        }
        Adjoint params.ancillaPrep(ancillas);
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
    /// - `ancillaPrep`: A function to prepare persistent block-encoding ancillas.
    /// # Returns
    /// The result of measuring the phase qubit after the IQPE circuit is executed.
    operation MakeIQPECircuit(
        statePrep : Qubit[] => Unit,
        repControlledUnitary : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
        computeQubitPercentage : Double,
    ) : Result[] {
        let totalQubits = 1 + Length(systems) + numAncillaQubits;
        if computeQubitPercentage > 0.0 {
            let computeCapacity = Ceiling(computeQubitPercentage * IntAsDouble(totalQubits) / 100.0);
            EnableMemoryComputeArchitecture(computeCapacity, LeastRecentlyUsed());
        }
        return RunIQPE(new IterativePhaseEstimationParams {
            statePrep = statePrep,
            repControlledUnitary = repControlledUnitary,
            accumulatePhase = accumulatePhase,
            phaseQubit = phaseQubit,
            systems = systems,
            numAncillaQubits = numAncillaQubits,
            ancillaPrep = ancillaPrep
        });
    }
}
