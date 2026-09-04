// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Kept out of IterativePhaseEstimation.qs so that module stays Base-profile compatible.
namespace QDKChemistry.Utils.CombinedIterationPhaseEstimation {

    import Std.Arrays.Subarray;
    import Std.Convert.IntAsDouble;
    import Std.Math.PI;

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
                Rz(-accumulatePhase, phase);
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
