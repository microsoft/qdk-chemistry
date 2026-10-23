// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledSwapPauliExp {

    import QDKChemistry.Utils.ControlledPauliExp.SplitPhasedTerms;
    import QDKChemistry.Utils.HammingWeightPhasing.HammingWeightPhase;
    import QDKChemistry.Utils.PauliExp.RepPauliExp;
    import QDKChemistry.Utils.PauliExp.RepPauliExpParams;
    import Std.Arrays.IndexOf;
    import Std.Arrays.Mapped;
    import Std.Arrays.Subarray;
    import Std.Math.AbsD;
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.ResourceEstimation.RepeatEstimates;

    /// Applies the declared layers, phasing equal-angle terms as in `ControlledPauliLayers`.
    internal operation PauliLayers(
        params : RepPauliExpParams,
        layerOffsets : Int[],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        for layer in 0..Length(layerOffsets) - 2 {
            let (unphasedTerms, phasedTerms) = SplitPhasedTerms(params, layerOffsets[layer], layerOffsets[layer + 1] - 1, useHammingWeightPhasing);
            for term in unphasedTerms {
                Exp(params.pauliOps[term], -params.pauliCoefficients[term], Subarray(params.pauliIndices[term], systems));
            }
            for group in phasedTerms {
                HammingWeightPhase(
                    params.pauliCoefficients[group[0]],
                    Mapped(term -> params.pauliOps[term], group),
                    Mapped(term -> Subarray(params.pauliIndices[term], systems), group),
                    useHammingWeightPhasing,
                    maxBatchSize
                );
            }
        }
    }

    /// Controls sparse Pauli evolution using a CSWAP sandwich.
    ///
    /// For an n-qubit system, this uses one additional n-qubit vacuum register and
    /// two layers of n controlled-`SWAP` operations. The shared sparse evolution applies
    /// its prefix once, repeats the body, and applies its suffix once inside the sandwich.
    ///
    /// The full evolution must satisfy `U|0...0> = e^{i phi_0}|0...0>`. Its vacuum phase
    /// lands on the |0> branch; `R1(vacuumPhase)` corrects the relative phase to give
    /// controlled-`U` up to a global phase.
    ///
    /// # Parameters
    /// - `evolution`: Sparse evolution, including repetition and one-time boundary terms.
    /// - `layerOffsets`: Declared disjoint-layer boundaries; empty means term-by-term evolution.
    /// - `useHammingWeightPhasing`: Whether to phase equal-angle rotations through Hamming-weight registers.
    /// - `maxBatchSize`: Largest Hamming-weight phasing batch, or -1 for no cap.
    /// - `vacuumPhase`: The phase `phi_0` of the full evolution on the vacuum.
    /// - `control`: The control qubit.
    /// - `systems`: System qubits indexed by the sparse evolution.
    operation RepControlledSwapPauliExp(
        evolution : RepPauliExpParams,
        layerOffsets : Int[],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        vacuumPhase : Double,
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj {
        use vacuum = Qubit[Length(systems)];
        within {
            for i in 0..Length(systems) - 1 {
                Controlled SWAP([control], (systems[i], vacuum[i]));
            }
        } apply {
            if not useHammingWeightPhasing or Length(layerOffsets) == 0 {
                RepPauliExp(evolution, vacuum);
            } else {
                let first = IndexOf(offset -> offset == evolution.numPrefixTerms, layerOffsets);
                let last = IndexOf(offset -> offset == Length(evolution.pauliCoefficients) - evolution.numSuffixTerms, layerOffsets);
                let stepOffsets = layerOffsets[first..last];
                PauliLayers(evolution, layerOffsets[0..first], useHammingWeightPhasing, maxBatchSize, vacuum);
                if IsResourceEstimating() {
                    within {
                        RepeatEstimates(evolution.repetitions);
                    } apply {
                        PauliLayers(evolution, stepOffsets, useHammingWeightPhasing, maxBatchSize, vacuum);
                    }
                } else {
                    for _ in 1..evolution.repetitions {
                        PauliLayers(evolution, stepOffsets, useHammingWeightPhasing, maxBatchSize, vacuum);
                    }
                }
                PauliLayers(evolution, layerOffsets[last...], useHammingWeightPhasing, maxBatchSize, vacuum);
            }
        }
        // Skipped when negligible so a vacuum-annihilating evolution costs no extra rotation.
        if AbsD(vacuumPhase) > 1e-12 {
            R1(vacuumPhase, control);
        }
    }

    /// Parameters for the repeated CSWAP-sandwich controlled Pauli evolution.
    /// `control` and `systems` are register indices for circuit allocation.
    struct RepControlledSwapPauliExpParams {
        evolution : RepPauliExpParams,
        layerOffsets : Int[],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        vacuumPhase : Double,
        control : Int,
        systems : Int[],
    }

    /// Creates a CSWAP-sandwich circuit at the specified control and system indices.
    operation MakeRepControlledSwapPauliExpCircuit(
        evolution : RepPauliExpParams,
        layerOffsets : Int[],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        vacuumPhase : Double,
        control : Int,
        systems : Int[]
    ) : Unit {
        // Size the register from the largest index across `control` and `systems` so that
        // non-contiguous layouts (e.g. control=2, systems=[3,4]) stay in range.
        mutable maxIndex = control;
        for idx in systems {
            if idx > maxIndex {
                set maxIndex = idx;
            }
        }

        use qs = Qubit[maxIndex + 1];
        RepControlledSwapPauliExp(
            evolution,
            layerOffsets,
            useHammingWeightPhasing,
            maxBatchSize,
            vacuumPhase,
            qs[control],
            Subarray(systems, qs)
        );
    }

    /// Returns an adjointable CSWAP-sandwich callable for supplied control and system qubits.
    function MakeRepControlledSwapPauliExpOp(params : RepControlledSwapPauliExpParams) : (Qubit, Qubit[]) => Unit is Adj {
        RepControlledSwapPauliExp(
            params.evolution,
            params.layerOffsets,
            params.useHammingWeightPhasing,
            params.maxBatchSize,
            params.vacuumPhase,
            _,
            _
        )
    }
}
