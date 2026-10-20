// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledPauliExp {

    import QDKChemistry.Utils.HammingWeightPhasing.HammingWeightPhase;
    import QDKChemistry.Utils.PauliExp.RepPauliExp;
    import QDKChemistry.Utils.PauliExp.RepPauliExpParams;
    import Std.Arrays.IndexOf;
    import Std.Arrays.Mapped;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Sorted;
    import Std.Arrays.Subarray;
    import Std.Canon.MapPauliAxis;
    import Std.Math.Max;
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.ResourceEstimation.RepeatEstimates;

    /// Returns the summed coefficient of the identity terms in `first..last`.
    function LayerIdentityPhase(params : RepPauliExpParams, first : Int, last : Int) : Double {
        mutable phase = 0.0;
        for term in first..last {
            if Length(params.pauliIndices[term]) == 0 {
                set phase += params.pauliCoefficients[term];
            }
        }
        phase
    }

    /// Splits terms `first..last` into unphased terms and groups of phased terms with exactly
    /// equal nonzero angles. Without phasing, every term is unphased, in input order.
    function SplitPhasedTerms(
        params : RepPauliExpParams,
        first : Int,
        last : Int,
        useHammingWeightPhasing : Bool
    ) : (Int[], Int[][]) {
        if not useHammingWeightPhasing {
            return (MappedOverRange(term -> term, first..last), []);
        }
        // Sort (angle, term) pairs; an identity term takes angle 0, so it is left unphased.
        // The comparator captures nothing: a capturing one breaks Base-profile lowering.
        let sorted = Sorted(
            ((leftAngle, leftTerm), (rightAngle, rightTerm)) -> leftAngle < rightAngle or (leftAngle == rightAngle and leftTerm <= rightTerm),
            MappedOverRange(term -> (Length(params.pauliIndices[term]) > 0 ? params.pauliCoefficients[term] | 0.0, term), first..last)
        );
        let angles = Mapped((angle, _) -> angle, sorted);
        let terms = Mapped((_, term) -> term, sorted);
        mutable unphasedTerms : Int[] = [];
        mutable phasedTerms : Int[][] = [];
        mutable start = 0;
        while start < Length(terms) {
            mutable stop = start + 1;
            while stop < Length(terms) and angles[stop] == angles[start] {
                set stop += 1;
            }
            if angles[start] == 0.0 {
                set unphasedTerms += terms[start..stop - 1];
            } else {
                set phasedTerms += [terms[start..stop - 1]];
            }
            set start = stop;
        }
        (unphasedTerms, phasedTerms)
    }

    /// Applies the producer's disjoint layers without changing their boundaries.
    /// With `useHammingWeightPhasing`, each layer's equal-angle rotations are phased together
    /// :cite:`Kan2025`, at most `maxBatchSize` per register.
    operation ControlledPauliLayers(
        params : RepPauliExpParams,
        layerOffsets : Int[],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        for layer in 0..Length(layerOffsets) - 2 {
            let first = layerOffsets[layer];
            let last = layerOffsets[layer + 1] - 1;
            if first == last and not useHammingWeightPhasing {
                Controlled Exp([control], (
                    params.pauliOps[first], -params.pauliCoefficients[first],
                    Subarray(params.pauliIndices[first], systems)
                ));
            } else {
                within {
                    for term in first..last {
                        let indices = params.pauliIndices[term];
                        let paulis = params.pauliOps[term];
                        for position in 0..Length(indices) - 1 {
                            MapPauliAxis(PauliZ, paulis[position], systems[indices[position]]);
                        }
                        if Length(indices) > 0 {
                            for position in 1..Length(indices) - 1 {
                                CNOT(systems[indices[position]], systems[indices[0]]);
                            }
                        }
                    }
                } apply {
                    // Rotations share two rounds, but the CNOTs still share one control.
                    let identityPhase = LayerIdentityPhase(params, first, last);
                    if identityPhase != 0.0 {
                        // One control phase keeps every identity term's relative phase, also under further controls.
                        R1(-identityPhase, control);
                    }
                    let (unphasedTerms, phasedTerms) = SplitPhasedTerms(params, first, last, useHammingWeightPhasing);
                    for term in unphasedTerms {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
                            Rz(params.pauliCoefficients[term], systems[indices[0]]);
                        }
                    }
                    for term in unphasedTerms {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
                            CNOT(control, systems[indices[0]]);
                        }
                    }
                    for term in unphasedTerms {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
                            Rz(-params.pauliCoefficients[term], systems[indices[0]]);
                        }
                    }
                    for term in unphasedTerms {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
                            CNOT(control, systems[indices[0]]);
                        }
                    }
                    for group in phasedTerms {
                        // Every term already sits on its first qubit as a single Z.
                        Controlled HammingWeightPhase([control], (
                            params.pauliCoefficients[group[0]],
                            [[PauliZ], size = Length(group)],
                            Mapped(term -> [systems[params.pauliIndices[term][0]]], group),
                            useHammingWeightPhasing,
                            maxBatchSize
                        ));
                    }
                }
            }
        }
    }

    /// Applies repeated sparse Pauli evolution controlled on a single qubit.
    ///
    /// This is a named operation rather than a closure so that callables produced by
    /// `MakeRepControlledPauliExpOp` stay resolvable by the Q# defunctionalizer, which
    /// runs when a caller such as `HadamardTest` is lowered to QIR.
    /// # Parameters
    /// - `params`: The sparse repeated Pauli evolution parameters.
    /// - `layerOffsets`: Declared disjoint-layer boundaries; empty means term-by-term evolution.
    /// - `useHammingWeightPhasing`: Whether to phase equal-angle rotations through Hamming-weight registers.
    /// - `maxBatchSize`: Largest Hamming-weight phasing batch, or -1 for no cap.
    /// - `control`: The control qubit.
    /// - `systems`: The system qubits the evolution acts on.
    operation RepControlledPauliExp(
        params : RepPauliExpParams,
        layerOffsets : Int[],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        if Length(layerOffsets) == 0 {
            Controlled RepPauliExp([control], (params, systems));
        } else {
            let first = IndexOf(offset -> offset == params.numPrefixTerms, layerOffsets);
            let last = IndexOf(offset -> offset == Length(params.pauliCoefficients) - params.numSuffixTerms, layerOffsets);
            let stepOffsets = layerOffsets[first..last];
            ControlledPauliLayers(params, layerOffsets[0..first], useHammingWeightPhasing, maxBatchSize, control, systems);
            if IsResourceEstimating() {
                within {
                    RepeatEstimates(params.repetitions);
                } apply {
                    ControlledPauliLayers(params, stepOffsets, useHammingWeightPhasing, maxBatchSize, control, systems);
                }
            } else {
                for _ in 1..params.repetitions {
                    ControlledPauliLayers(params, stepOffsets, useHammingWeightPhasing, maxBatchSize, control, systems);
                }
            }
            ControlledPauliLayers(params, layerOffsets[last...], useHammingWeightPhasing, maxBatchSize, control, systems);
        }
    }

    /// A helper operation to create a circuit for repeated Controlled Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: The sparse repeated Pauli evolution parameters.
    /// - `layerOffsets`: Declared disjoint-layer boundaries; empty means term-by-term evolution.
    /// - `useHammingWeightPhasing`: Whether to phase equal-angle rotations through Hamming-weight registers.
    /// - `maxBatchSize`: Largest Hamming-weight phasing batch, or -1 for no cap.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of integers representing the indices of the system qubits.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated controlled time evolution on the allocated qubits.
    operation MakeRepControlledPauliExpCircuit(
        params : RepPauliExpParams,
        layerOffsets : Int[],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        control : Int,
        systems : Int[]
    ) : Unit {
        use qs = Qubit[Max([control] + systems) + 1];
        RepControlledPauliExp(params, layerOffsets, useHammingWeightPhasing, maxBatchSize, qs[control], Subarray(systems, qs));
    }

    /// Returns a single-control callable for repeated sparse Pauli evolution.
    function MakeRepControlledPauliExpOp(
        params : RepPauliExpParams,
        layerOffsets : Int[],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int
    ) : ((Qubit, Qubit[]) => Unit is Adj + Ctl) {
        RepControlledPauliExp(params, layerOffsets, useHammingWeightPhasing, maxBatchSize, _, _)
    }
}
