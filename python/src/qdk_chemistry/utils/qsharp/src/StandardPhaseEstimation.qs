// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.StandardPhaseEstimation {

    import Std.Arrays.Subarray;
    import Std.Canon.ApplyQFT;
    import Std.Convert.IntAsDouble;
    import Std.Math.Ceiling;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EnableMemoryComputeArchitecture;
    import Std.ResourceEstimation.EndEstimateCaching;
    import Std.ResourceEstimation.LeastRecentlyUsed;
    import Std.ResourceEstimation.RepeatEstimates;

    /// A struct to hold parameters for standard Quantum Phase Estimation (QPE).
    /// - `statePrep`: A function to prepare the initial quantum state on system qubits.
    /// - `controlledEvolutions`: An array of functions to perform controlled-U^(2^k) on (control, targets),
    ///   one per ancilla qubit. Each operation already encapsulates the correct power.
    /// - `phaseQubitPrep`: A function to prepare the phase (ancilla) qubits (e.g., Hadamard on each qubit).
    /// - `numBits`: The number of ancilla qubits (phase bits) for QPE.
    /// - `ancillas`: An array of indices representing the phase ancilla qubits.
    /// - `systems`: An array of indices representing the system qubits (state prep target).
    /// - `numAncillas`: Number of extra ancillas needed.
    /// - `ancillaPrep`: A function to prepare persistent ancillas (e.g., phase gradient state).
    ///   Called once before the walk steps; adjoint is applied after measurements. No-op when not needed.
    struct StandardPhaseEstimationParams {
        statePrep : Qubit[] => Unit,
        controlledEvolutions : ((Qubit, Qubit[]) => Unit)[],
        phaseQubitPrep : Qubit[] => Unit,
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        numAncillas : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
    }

    /// Runs the standard Quantum Phase Estimation (QPE) circuit based on the provided parameters.
    /// The circuit uses multiple ancilla qubits and the inverse QFT.
    /// # Parameters
    /// - `params`: A `StandardPhaseEstimationParams` struct.
    /// # Returns
    /// - `Result[]`: The measurement results of the ancilla qubits (MSB first).
    operation RunStandardQPE(params : StandardPhaseEstimationParams) : Result[] {
        let totalQubits = params.numBits + Length(params.systems) + params.numAncillas;
        use qs = Qubit[totalQubits];
        let phaseAncillas = Subarray(params.ancillas, qs);
        let systems = Subarray(params.systems, qs);

        let ancillas = if params.numAncillas == 0 {
            []
        } else {
            qs[params.numBits + Length(params.systems)..Length(qs) - 1]
        };
        let allTargets = systems + ancillas;

        // Step 1: Prepare the initial state on system qubits only
        params.statePrep(systems);

        // Step 1.5: Prepare persistent ancillas (e.g., phase gradient)
        params.ancillaPrep(ancillas);

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
    /// - `numAncillas`: Number of extra ancillas.
    /// - `ancillaPrep`: A function to prepare persistent ancillas.
    /// # Returns
    /// The measurement results of the phase ancilla qubits.
    operation MakeStandardQPECircuit(
        statePrep : Qubit[] => Unit,
        controlledEvolutions : ((Qubit, Qubit[]) => Unit)[],
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        phaseQubitPrep : Qubit[] => Unit,
        numAncillas : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
        computeQubitPercentage : Double,
    ) : Result[] {
        let totalQubits = numBits + Length(systems) + numAncillas;
        if computeQubitPercentage > 0.0 {
            let computeCapacity = Ceiling(computeQubitPercentage * IntAsDouble(totalQubits) / 100.0);
            EnableMemoryComputeArchitecture(computeCapacity, LeastRecentlyUsed());
        }
        return RunStandardQPE(new StandardPhaseEstimationParams {
            statePrep = statePrep,
            controlledEvolutions = controlledEvolutions,
            phaseQubitPrep = phaseQubitPrep,
            numBits = numBits,
            ancillas = ancillas,
            systems = systems,
            numAncillas = numAncillas,
            ancillaPrep = ancillaPrep,
        });
    }

    /// Standard QPE that takes a single controlled walk operator and repeats it
    /// with powers of 2 for each ancilla qubit.
    ///
    /// This avoids building num_bits separate controlled circuits on the Python side,
    /// which is expensive for large chemistry systems.
    ///
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state on system qubits.
    /// - `controlledUnitary`: A single controlled unitary operation (control, targets) => Unit.
    ///   This is NOT pre-powered — the QPE loop applies it 2^(n-1-k) times for ancilla k.
    /// - `numBits`: The number of ancilla qubits (phase bits) for QPE.
    /// - `ancillas`: An array of indices for the phase ancilla qubits.
    /// - `systems`: An array of indices for the system qubits (state prep target).
    /// - `phaseQubitPrep`: A function to prepare the phase qubits (e.g., Hadamard on all).
    /// - `numAncillas`: Number of extra ancillas.
    /// - `ancillaPrep`: A function to prepare persistent ancillas.
    /// # Returns
    /// The measurement results of the phase ancilla qubits.
    operation MakeRepeatedQPECircuit(
        statePrep : Qubit[] => Unit,
        controlledUnitary : (Qubit, Qubit[]) => Unit,
        numBits : Int,
        ancillas : Int[],
        systems : Int[],
        phaseQubitPrep : Qubit[] => Unit,
        numAncillas : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
        computeQubitPercentage : Double,
    ) : Result[] {
        let totalQubits = numBits + Length(systems) + numAncillas;
        if computeQubitPercentage > 0.0 {
            let computeCapacity = Ceiling(computeQubitPercentage * IntAsDouble(totalQubits) / 100.0);
            EnableMemoryComputeArchitecture(computeCapacity, LeastRecentlyUsed());
        }
        use qs = Qubit[totalQubits];
        let phaseAncillas = Subarray(ancillas, qs);
        let systemQubits = Subarray(systems, qs);
        let beAncillas = if numAncillas == 0 {
            []
        } else {
            qs[numBits + Length(systems)..Length(qs) - 1]
        };
        let allTargets = systemQubits + beAncillas;

        // Step 1: Prepare the initial state on system qubits
        statePrep(systemQubits);

        // Step 1.5: Prepare persistent ancillas
        ancillaPrep(beAncillas);

        // Step 2: Prepare phase (ancilla) qubits
        phaseQubitPrep(phaseAncillas);

        // Step 3: Apply controlled walk with powers of 2 per ancilla.
        // ancillas[0] = MSB controls U^(2^(n-1)), ancillas[n-1] = LSB controls U^1.
        //
        // Detect execution mode: In simulation, BeginEstimateCaching always returns
        // true. In resource estimation, the second call with the same key returns
        // false (cache hit). This lets us choose the optimal strategy for each mode.
        mutable useRepeatEstimates = false;
        if BeginEstimateCaching("__mode_probe__", 0) {
            EndEstimateCaching();
            if not BeginEstimateCaching("__mode_probe__", 0) {
                // Second call returned false => resource estimation mode
                set useRepeatEstimates = true;
            } else {
                EndEstimateCaching();
            }
        }

        for ancillaIdx in 0..numBits - 1 {
            let power = 1 <<< (numBits - 1 - ancillaIdx);
            if useRepeatEstimates {
                // Resource estimation: RepeatEstimates multiplies the single-step
                // cost by `power` without looping. BeginEstimateCaching ensures the
                // walk is traced only once across all ancilla iterations.
                within { RepeatEstimates(power); } apply {
                    if BeginEstimateCaching("controlled_walk", numAncillas) {
                        controlledUnitary(phaseAncillas[ancillaIdx], allTargets);
                        EndEstimateCaching();
                    }
                }
            } else {
                // Simulation: actually apply the walk `power` times.
                for _ in 0..power - 1 {
                    controlledUnitary(phaseAncillas[ancillaIdx], allTargets);
                }
            }
        }

        // Step 4: Apply inverse QFT on phase ancilla qubits
        Adjoint ApplyQFT(phaseAncillas);

        // Step 5: Measure phase ancilla qubits and reset everything else
        ResetAll(allTargets);
        mutable results = [Zero, size = numBits];
        for idx in 0..numBits - 1 {
            set results w/= idx <- MResetZ(phaseAncillas[idx]);
        }
        return results;
    }
}
