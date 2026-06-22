// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.StandardPhaseEstimation {

    import Std.Arrays.Subarray;
    import Std.Canon.ApplyQFT;

    /// A struct to hold parameters for standard Quantum Phase Estimation (QPE).
    /// - `statePrep`: A function to prepare the initial quantum state on system qubits.
    /// - `controlledEvolutions`: An array of functions to perform controlled-U^(2^k) on (control, targets),
    ///   one per ancilla qubit. Each operation already encapsulates the correct power.
    /// - `phaseQubitPrep`: A function to prepare the phase (ancilla) qubits (e.g., Hadamard on each qubit).
    /// - `numBits`: The number of ancilla qubits (phase bits) for QPE.
    /// - `ancillas`: An array of indices representing the phase ancilla qubits.
    /// - `systems`: An array of indices representing the system qubits (state prep target).
    /// - `numBlockEncodingAncillas`: Number of extra ancillas needed by block encoding (0 for LCU/Trotter).
    /// - `ancillaPrep`: A function to prepare persistent block-encoding ancillas (e.g., phase gradient state).
    ///   Called once before the walk steps; adjoint is applied after measurements. No-op when not needed.
    struct StandardPhaseEstimationParams {
        statePrep : Qubit[] => Unit,
        controlledEvolutions : ((Qubit, Qubit[]) => Unit)[],
        phaseQubitPrep : Qubit[] => Unit,
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        numBlockEncodingAncillas : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
    }

    /// Runs the standard Quantum Phase Estimation (QPE) circuit based on the provided parameters.
    /// The circuit uses multiple ancilla qubits and the inverse QFT.
    /// # Parameters
    /// - `params`: A `StandardPhaseEstimationParams` struct.
    /// # Returns
    /// - `Result[]`: The measurement results of the ancilla qubits (MSB first).
    operation RunStandardQPE(params : StandardPhaseEstimationParams) : Result[] {
        let totalQubits = params.numBits + Length(params.systems) + params.numBlockEncodingAncillas;
        use qs = Qubit[totalQubits];
        let phaseAncillas = Subarray(params.ancillas, qs);
        let systems = Subarray(params.systems, qs);
        // Block encoding ancillas sit after phase ancillas and system qubits
        let beAncillas = if params.numBlockEncodingAncillas == 0 {
            []
        } else {
            qs[params.numBits + Length(params.systems)..Length(qs) - 1]
        };
        let allTargets = systems + beAncillas;

        // Step 1: Prepare the initial state on system qubits only
        params.statePrep(systems);

        // Step 1.5: Prepare persistent block-encoding ancillas (e.g., phase gradient)
        params.ancillaPrep(beAncillas);

        // Step 2: Prepare phase (ancilla) qubits
        params.phaseQubitPrep(phaseAncillas);

        // Step 3: Apply controlled-U^(2^k) for each phase ancilla qubit k
        // Each controlledEvolutions[k] already implements the correct power.
        // ApplyQFT uses big-endian: ancillas[0] = MSB, so ancillas[0] controls U^(2^(n-1))
        for ancillaIdx in 0..params.numBits - 1 {
            params.controlledEvolutions[ancillaIdx](phaseAncillas[ancillaIdx], allTargets);
        }

        // Step 4: Apply inverse QFT on phase ancilla qubits
        Adjoint ApplyQFT(phaseAncillas);

        // Step 5: Measure phase ancilla qubits and reset everything else
        ResetAll(allTargets);
        mutable results = [Zero, size = params.numBits];
        for idx in 0..params.numBits - 1 {
            set results w/= idx <- MResetZ(phaseAncillas[idx]);
        }
        return results;
    }

    /// Prepare a standard QPE circuit (factory entry point).
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `controlledEvolutions`: An array of functions to perform controlled-U^(2^k) on (control, targets),
    ///   one per ancilla qubit. Each operation already encapsulates the correct power.
    /// - `numBits`: The number of ancilla qubits (phase bits) for QPE.
    /// - `ancillas`: An array of indices for the phase ancilla qubits.
    /// - `systems`: An array of indices for the system qubits (state prep target).
    /// - `phaseQubitPrep`: A function to prepare the phase qubits (e.g., Hadamard on all).
    /// - `numBlockEncodingAncillas`: Number of extra ancillas for block encoding (0 for LCU/Trotter).
    /// - `ancillaPrep`: A function to prepare persistent block-encoding ancillas.
    /// # Returns
    /// The measurement results of the phase ancilla qubits.
    operation MakeStandardQPECircuit(
        statePrep : Qubit[] => Unit,
        controlledEvolutions : ((Qubit, Qubit[]) => Unit)[],
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        phaseQubitPrep : Qubit[] => Unit,
        numBlockEncodingAncillas : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
    ) : Result[] {
        return RunStandardQPE(new StandardPhaseEstimationParams {
            statePrep = statePrep,
            controlledEvolutions = controlledEvolutions,
            phaseQubitPrep = phaseQubitPrep,
            numBits = numBits,
            ancillas = ancillas,
            systems = systems,
            numBlockEncodingAncillas = numBlockEncodingAncillas,
            ancillaPrep = ancillaPrep,
        });
    }
}
