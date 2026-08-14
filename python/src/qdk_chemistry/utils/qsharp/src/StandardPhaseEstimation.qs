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
    /// - `ancillaPrep`: A function to prepare persistent ancillas (e.g., phase gradient state).
    ///   Called once before the controlled unitaries; because this operation is adjointable the
    ///   adjoint unprepares them. No-op when the controlled unitary needs no persistent ancillas.
    struct StandardPhaseEstimationParams {
        statePrep : Qubit[] => Unit is Adj,
        controlledUnitary : ((Qubit, Qubit[]) => Unit is Adj)[],
        phaseQubitPrep : Qubit[] => Unit is Adj,
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        numAncillaQubits : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
    }

    /// Runs the standard Quantum Phase Estimation (QPE) circuit based on the provided parameters.
    /// The circuit uses multiple ancilla qubits and the inverse QFT. Nothing is measured, so the
    /// operation is adjointable and can be composed, for example as the preparation amplitude
    /// amplification reflects about.
    /// # Parameters
    /// - `params`: A `StandardPhaseEstimationParams` struct.
    /// - `qs`: The register to act on, indexed by `params.ancillas` and `params.systems`.
    operation RunStandardQPE(params : StandardPhaseEstimationParams, qs : Qubit[]) : Unit is Adj {
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

        // Step 1.5: Prepare persistent ancillas (e.g., the phase gradient state) used by
        // every controlled unitary. Adjointing this operation unprepares them.
        params.ancillaPrep(unitaryAncillas);

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
    }

    /// Prepare a standard QPE operation that acts in place on a caller-owned register.
    /// Parameters match `MakeStandardQPECircuit`.
    /// # Returns
    /// - `Qubit[] => Unit is Adj`: A callable that applies QPE without measuring.
    function MakeStandardQPEOp(
        statePrep : Qubit[] => Unit is Adj,
        controlledUnitary : ((Qubit, Qubit[]) => Unit is Adj)[],
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        phaseQubitPrep : Qubit[] => Unit is Adj,
        numAncillaQubits : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
    ) : Qubit[] => Unit is Adj {
        RunStandardQPE(
            new StandardPhaseEstimationParams {
                statePrep = statePrep,
                controlledUnitary = controlledUnitary,
                phaseQubitPrep = phaseQubitPrep,
                numBits = numBits,
                ancillas = ancillas,
                systems = systems,
                numAncillaQubits = numAncillaQubits,
                ancillaPrep = ancillaPrep,
            },
            _
        )
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
    /// - `measurePhase`: Measure the ancilla qubits. When `false` nothing is measured.
    /// # Returns
    /// The measurement results of the ancilla qubits, or an empty array when `measurePhase` is `false`.
    operation MakeStandardQPECircuit(
        statePrep : Qubit[] => Unit is Adj,
        controlledUnitary : ((Qubit, Qubit[]) => Unit is Adj)[],
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        phaseQubitPrep : Qubit[] => Unit is Adj,
        numAncillaQubits : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
        measurePhase : Bool,
    ) : Result[] {
        let totalQubits = numBits + Length(systems) + numAncillaQubits;
        use qs = Qubit[totalQubits];
        RunStandardQPE(
            new StandardPhaseEstimationParams {
                statePrep = statePrep,
                controlledUnitary = controlledUnitary,
                phaseQubitPrep = phaseQubitPrep,
                numBits = numBits,
                ancillas = ancillas,
                systems = systems,
                numAncillaQubits = numAncillaQubits,
                ancillaPrep = ancillaPrep,
            },
            qs
        );

        mutable results : Result[] = [];
        if measurePhase {
            let phaseQubits = Subarray(ancillas, qs);
            set results = [Zero, size = numBits];
            for idx in 0..numBits - 1 {
                set results w/= idx <- MResetZ(phaseQubits[idx]);
            }
        }
        ResetAll(qs);
        return results;
    }
}
