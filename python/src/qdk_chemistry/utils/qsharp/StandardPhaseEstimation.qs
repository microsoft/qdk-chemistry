// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.StandardPhaseEstimation {

    import Std.Arrays.Subarray;
    import Std.Canon.ApplyQFT;

    /// A struct to hold parameters for standard Quantum Phase Estimation (QPE).
    /// - `statePrep`: A function to prepare the initial quantum state on system qubits.
    /// - `controlledEvolution`: A function to perform controlled-U on (control, systems).
    /// - `phaseQubitPrep`: A function to prepare the phase (ancilla) qubits. Defaults to applying H to each.
    /// - `numBits`: The number of ancilla qubits (phase bits) for QPE.
    /// - `ancillas`: An array of indices representing the ancilla qubits.
    /// - `systems`: An array of indices representing the system qubits.
    struct StandardPhaseEstimationParams {
        statePrep : Qubit[] => Unit,
        controlledEvolution : (Qubit, Qubit[]) => Unit,
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
        // ApplyQFT uses big-endian: ancillas[0] = MSB, so ancillas[0] controls U^(2^(n-1))
        for ancillaIdx in 0..params.numBits - 1 {
            let power = 1 <<< (params.numBits - 1 - ancillaIdx);
            for _ in 1..power {
                params.controlledEvolution(ancillas[ancillaIdx], systems);
            }
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

    /// Prepare a standard QPE circuit (factory entry point).
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `controlledEvolution`: A function to perform controlled-U on (control, systems).
    /// - `numBits`: The number of ancilla qubits (phase bits) for QPE.
    /// - `ancillas`: An array of indices for the ancilla qubits.
    /// - `systems`: An array of indices for the system qubits.
    /// - `phaseQubitPrep`: Optional function to prepare the phase qubits. Defaults to Hadamard on all.
    /// # Returns
    /// The measurement results of the ancilla qubits.
    operation MakeStandardQPECircuit(
        statePrep : Qubit[] => Unit,
        controlledEvolution : (Qubit, Qubit[]) => Unit,
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        phaseQubitPrep : Qubit[] => Unit,
    ) : Result[] {
        return RunStandardQPE(new StandardPhaseEstimationParams {
            statePrep = statePrep,
            controlledEvolution = controlledEvolution,
            phaseQubitPrep = phaseQubitPrep,
            numBits = numBits,
            ancillas = ancillas,
            systems = systems
        });
    }
}
