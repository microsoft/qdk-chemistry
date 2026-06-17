// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.StandardPhaseEstimation {

    import Std.Arrays.Subarray;
    import Std.Canon.ApplyQFT;
    import Std.ResourceEstimation.*;

    /// A struct to hold parameters for standard Quantum Phase Estimation (QPE).
    /// - `statePrep`: A function to prepare the initial quantum state on system qubits.
    /// - `controlledEvolutions`: An array of functions to perform controlled-U^(2^k) on (control, systems),
    ///   one per ancilla qubit. Each operation already encapsulates the correct power.
    /// - `phaseQubitPrep`: A function to prepare the phase (ancilla) qubits (e.g., Hadamard on each qubit).
    /// - `numBits`: The number of ancilla qubits (phase bits) for QPE.
    /// - `ancillas`: An array of indices representing the ancilla qubits.
    /// - `systems`: An array of indices representing the system qubits.
    struct StandardPhaseEstimationParams {
        statePrep : Qubit[] => Unit,
        controlledEvolutions : ((Qubit, Qubit[]) => Unit)[],
        phaseQubitPrep : Qubit[] => Unit,
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
    }

    /// Runs the standard Quantum Phase Estimation (QPE) circuit based on the provided parameters.
    /// The circuit uses multiple ancilla qubits and the inverse QFT.
    /// # Parameters
    /// - `params`: A `StandardPhaseEstimationParams` struct.
    /// # Returns
    /// - `Result[]`: The measurement results of the ancilla qubits (MSB first).
    operation RunStandardQPE(params : StandardPhaseEstimationParams) : Result[] {
        let totalQubits = params.numBits + Length(params.systems);
        use qs = Qubit[totalQubits];
        let ancillas = Subarray(params.ancillas, qs);
        let systems = Subarray(params.systems, qs);

        // Step 1: Prepare the initial state on system qubits
        params.statePrep(systems);

        // Step 2: Prepare phase (ancilla) qubits
        params.phaseQubitPrep(ancillas);

        // Step 3: Apply controlled-U^(2^k) for each ancilla qubit k
        // Each controlledEvolutions[k] already implements the correct power.
        // ApplyQFT uses big-endian: ancillas[0] = MSB, so ancillas[0] controls U^(2^(n-1))
        for ancillaIdx in 0..params.numBits - 1 {
            params.controlledEvolutions[ancillaIdx](ancillas[ancillaIdx], systems);
        }

        // Step 4: Apply inverse QFT on ancilla qubits
        Adjoint ApplyQFT(ancillas);

        // Step 5: Measure ancilla qubits and reset system qubits
        ResetAll(systems);
        mutable results = [Zero, size = params.numBits];
        for idx in 0..params.numBits - 1 {
            set results w/= idx <- MResetZ(ancillas[idx]);
        }
        return results;
    }
    /// Fast resource estimation for standard QPE using RepeatEstimates.
    ///
    /// Instead of tracing through all 2^numBits - 1 Trotter steps individually,
    /// this operation tells the resource estimator to analyze a single controlled
    /// Trotter step and multiply the cost by numQueries. This is dramatically faster
    /// for large circuits.
    ///
    /// # Parameters
    /// - `numQueries`: Total number of Trotter steps (2^numBits - 1 for standard QPE).
    /// - `singleControlledEvolution`: The base controlled-U operation (power=1).
    /// - `statePrep`: State preparation operation on system qubits.
    /// - `numBits`: Number of ancilla qubits for QPE.
    /// - `numSystemQubits`: Number of system qubits.
    operation EstimateStandardQPE(
        singleControlledEvolution : (Qubit, Qubit[]) => Unit,
        statePrep : Qubit[] => Unit,
        numBits : Int,
        numSystemQubits : Int,
    ) : Unit {
        use ancillas = Qubit[numBits];
        use systems = Qubit[numSystemQubits];

        // State preparation (counted once)
        statePrep(systems);

        // Total controlled-U applications across all ancillas = 2^numBits - 1.
        // The gate cost is control-qubit-agnostic, so trace once and multiply.
        // Qubit count is correct because all numBits ancillas are allocated above.
        within { RepeatEstimates(2^numBits - 1); } apply {
            singleControlledEvolution(ancillas[0], systems);
        }

        // Inverse QFT on ancillas (counted once)
        Adjoint ApplyQFT(ancillas);

        ResetAll(ancillas);
        ResetAll(systems);
    }

    /// Prepare a standard QPE circuit (factory entry point).
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `controlledEvolutions`: An array of functions to perform controlled-U^(2^k) on (control, systems),
    ///   one per ancilla qubit. Each operation already encapsulates the correct power.
    /// - `numBits`: The number of ancilla qubits (phase bits) for QPE.
    /// - `ancillas`: An array of indices for the ancilla qubits.
    /// - `systems`: An array of indices for the system qubits.
    /// - `phaseQubitPrep`: A function to prepare the phase qubits (e.g., Hadamard on all).
    /// # Returns
    /// The measurement results of the ancilla qubits.
    operation MakeStandardQPECircuit(
        statePrep : Qubit[] => Unit,
        controlledEvolutions : ((Qubit, Qubit[]) => Unit)[],
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        phaseQubitPrep : Qubit[] => Unit,
    ) : Result[] {
        return RunStandardQPE(new StandardPhaseEstimationParams {
            statePrep = statePrep,
            controlledEvolutions = controlledEvolutions,
            phaseQubitPrep = phaseQubitPrep,
            numBits = numBits,
            ancillas = ancillas,
            systems = systems
        });
    }
}
