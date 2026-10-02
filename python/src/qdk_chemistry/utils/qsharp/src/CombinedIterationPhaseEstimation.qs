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
    /// Each round is repeated `shotsPerBit` times and the bit fed forward is the majority
    /// of those repetitions, so one execution of this circuit consumes the same number of
    /// controlled-unitary applications as a full pass of the per-round path and yields the
    /// same estimator -- the difference is that it needs a single job rather than one per bit.
    /// # Parameters
    /// - `numBits`: Number of phase bits to estimate.
    /// - `shotsPerBit`: Repetitions of each round that are majority-voted to decide its bit.
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `controlledUnitary`: An array of controlled-U^(2^k) operations, one per round.
    ///    Each operation already encapsulates the correct power, so the unitary builder's
    ///    `power_strategy` is honoured exactly as in the per-round path.
    /// - `phaseQubit`: The index of the phase qubit (ancilla used for phase readout).
    /// - `systems`: An array of indices representing the system qubits.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled unitary (0 if none).
    /// # Returns
    /// An array of `numBits` majority-voted results. `results[0]` is measured with the
    /// highest power `2^(numBits - 1)`, matching the round ordering of the per-round builder.
    operation RunFullIQPE(
        numBits : Int,
        shotsPerBit : Int,
        statePrep : Qubit[] => Unit,
        controlledUnitary : ((Qubit, Qubit[]) => Unit)[],
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
    ) : Result[] {
        if shotsPerBit < 1 {
            fail "shotsPerBit must be a positive integer.";
        }
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
            mutable ones = 0;
            for _ in 1..shotsPerBit {
                statePrep(system);

                within {
                    H(phase);
                } apply {
                    // Apply the phase correction from previously voted bits as one fixed-angle
                    // rotation per bit. Summing them into a mutable Double would make that Double
                    // measurement-dependent, which Adaptive_RI rejects (UseOfDynamicDouble); here
                    // only the branch depends on a measurement, and Rz angles add under composition.
                    for j in 0..k - 1 {
                        if results[j] == One {
                            Rz(-2.0 * PI() / IntAsDouble(1 <<< (k - j + 1)), phase);
                        }
                    }
                    controlledUnitary[k](phase, allTargets);
                }

                if MResetZ(phase) == One {
                    set ones += 1;
                }
                ResetAll(allTargets);
            }

            // Majority vote, matching the host's tie-to-zero rule. The vote is encoded back
            // onto the now-idle phase qubit so the round still reports a Result, keeping the
            // return shape and bit ordering identical to the per-round path.
            if ones * 2 > shotsPerBit {
                X(phase);
            }
            set results w/= k <- MResetZ(phase);
        }

        return results;
    }
}
