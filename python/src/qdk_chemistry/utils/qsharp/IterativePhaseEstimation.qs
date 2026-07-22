// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.IterativePhaseEstimation {

    import Std.Arrays.Subarray;
    import Std.Convert.IntAsDouble;
    import Std.Math.PI;

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

    /// Runs the full iterative Quantum Phase Estimation (IQPE) as a single circuit
    /// with in-circuit classical feedback.
    ///
    /// Unlike `RunIQPE`, which measures a single phase bit per circuit execution and
    /// relies on the host to accumulate the phase correction between rounds, this
    /// operation performs every round in one circuit. It uses mid-circuit measurement
    /// and classical feed-forward to compute and apply the phase correction on device.
    /// It therefore requires a target that supports the Adaptive profile (mid-circuit
    /// measurement and classical control) and is not compatible with Base-profile-only
    /// targets.
    /// # Parameters
    /// - `numBits`: Number of phase bits to estimate.
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledUnitary`: A single controlled unitary (power 1); it is applied
    ///    `2^(numBits - 1 - k)` times in round `k`.
    /// - `phaseQubit`: The index of the phase qubit (ancilla used for phase readout).
    /// - `systems`: An array of indices representing the system qubits.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled unitary (0 if none).
    /// # Returns
    /// An array of `numBits` measurement results. `results[0]` is measured with the
    /// highest power `2^(numBits - 1)`, matching the round ordering of the per-round builder.
    operation RunFullIQPE(
        numBits : Int,
        statePrep : Qubit[] => Unit,
        repControlledUnitary : (Qubit, Qubit[]) => Unit,
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
    ) : Result[] {
        use qs = Qubit[Length(systems) + 1 + numAncillaQubits];
        let phase = qs[phaseQubit];
        let system = Subarray(systems, qs);
        let ancillas = if numAncillaQubits == 0 {
            []
        } else {
            qs[1 + Length(systems)..Length(qs) - 1]
        };
        let allTargets = system + ancillas;

        mutable results = [Zero, size = numBits];

        for k in 0..numBits - 1 {
            statePrep(system);
            let rep = 2^(numBits - 1 - k);
            // Compute accumulated phase correction from previously measured bits.
            mutable accumulatePhase = 0.0;
            for j in 0..k - 1 {
                if results[j] == One {
                    set accumulatePhase += 2.0 * PI() / IntAsDouble(1 <<< (k - j + 1));
                }
            }

            within {
                H(phase);
            } apply {
                Rz(accumulatePhase, phase);
                for _ in 1..rep {
                    repControlledUnitary(phase, allTargets);
                }
            }

            set results w/= k <- MResetZ(phase);
            ResetAll(allTargets);
        }

        return results;
    }
}
