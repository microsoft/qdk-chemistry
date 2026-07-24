// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.StandardPhaseEstimation {

    import Std.Arrays.Subarray;
    import Std.Canon.ApplyQFT;

    /// A struct to hold parameters for standard Quantum Phase Estimation (QPE).
    /// - `statePrep`: A function to prepare the initial quantum state on system qubits.
    /// - `controlledUnitary`: An array of functions to perform controlled-U^(2^k) on (control, systems),
    ///   one per ancilla qubit. Each operation already encapsulates the correct power.
    /// - `phaseQubitPrep`: A function to prepare the phase (ancilla) qubits (e.g., Hadamard on each qubit).
    /// - `numBits`: The number of ancilla qubits (phase bits) for QPE.
    /// - `ancillas`: An array of indices representing the ancilla qubits.
    /// - `systems`: An array of indices representing the system qubits.
    /// - `numAncillaQubits`: Number of extra ancilla qubits needed by the controlled unitary (0 for Trotter, >0 for block encoding).
    struct StandardPhaseEstimationParams {
        statePrep : Qubit[] => Unit,
        controlledUnitary : ((Qubit, Qubit[]) => Unit)[],
        phaseQubitPrep : Qubit[] => Unit,
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        numAncillaQubits : Int,
    }

    /// Runs the standard Quantum Phase Estimation (QPE) circuit based on the provided parameters.
    /// The circuit uses multiple ancilla qubits and the inverse QFT.
    /// # Parameters
    /// - `params`: A `StandardPhaseEstimationParams` struct.
    /// # Returns
    /// - `Result[]`: The measurement results of the ancilla qubits (MSB first).
    operation RunStandardQPE(params : StandardPhaseEstimationParams) : Result[] {
        let totalQubits = params.numBits + Length(params.systems) + params.numAncillaQubits;
        use qs = Qubit[totalQubits];
        let ancillas = Subarray(params.ancillas, qs);
        let systems = Subarray(params.systems, qs);
        let unitaryAncillas = if params.numAncillaQubits == 0 {
            []
        } else {
            qs[params.numBits + Length(params.systems)..Length(qs) - 1]
        };
        let allTargets = systems + unitaryAncillas;

        // Step 1: Prepare the initial state on system qubits
        params.statePrep(systems);

        // Step 2: Prepare phase (ancilla) qubits
        params.phaseQubitPrep(ancillas);

        // Step 3: Apply controlled-U^(2^k) for each ancilla qubit k
        // Each controlledUnitary[k] already implements the correct power.
        // ApplyQFT uses big-endian: ancillas[0] = MSB, so ancillas[0] controls U^(2^(n-1))
        for ancillaIdx in 0..params.numBits - 1 {
            params.controlledUnitary[ancillaIdx](ancillas[ancillaIdx], allTargets);
        }

        // Step 4: Apply inverse QFT on ancilla qubits
        Adjoint ApplyQFT(ancillas);

        // Step 5: Measure ancilla qubits and reset system qubits
        ResetAll(allTargets);
        mutable results = [Zero, size = params.numBits];
        for idx in 0..params.numBits - 1 {
            set results w/= idx <- MResetZ(ancillas[idx]);
        }
        return results;
    }

    /// Prepare a standard QPE circuit (factory entry point).
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `controlledUnitary`: An array of functions to perform controlled-U^(2^k) on (control, systems),
    ///   one per ancilla qubit. Each operation already encapsulates the correct power.
    /// - `numBits`: The number of ancilla qubits (phase bits) for QPE.
    /// - `ancillas`: An array of indices for the ancilla qubits.
    /// - `systems`: An array of indices for the system qubits.
    /// - `phaseQubitPrep`: A function to prepare the phase qubits (e.g., Hadamard on all).
    /// - `numAncillaQubits`: Number of extra ancilla qubits needed by the controlled unitary (0 for Trotter).
    /// # Returns
    /// The measurement results of the ancilla qubits.
    operation MakeStandardQPECircuit(
        statePrep : Qubit[] => Unit,
        controlledUnitary : ((Qubit, Qubit[]) => Unit)[],
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        phaseQubitPrep : Qubit[] => Unit,
        numAncillaQubits : Int,
    ) : Result[] {
        return RunStandardQPE(new StandardPhaseEstimationParams {
            statePrep = statePrep,
            controlledUnitary = controlledUnitary,
            phaseQubitPrep = phaseQubitPrep,
            numBits = numBits,
            ancillas = ancillas,
            systems = systems,
            numAncillaQubits = numAncillaQubits,
        });
    }
}
