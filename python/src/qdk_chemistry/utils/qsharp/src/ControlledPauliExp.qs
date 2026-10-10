// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledPauliExp {

    import QDKChemistry.Utils.HammingWeightPhasing.EqualAngleTowers;
    import QDKChemistry.Utils.HammingWeightPhasing.HammingWeightPhase;
    import QDKChemistry.Utils.PauliExp.RepPauliExp;
    import QDKChemistry.Utils.PauliExp.RepPauliExpParams;
    import Std.Arrays.IndexOf;
    import Std.Arrays.Mapped;
    import Std.Arrays.MappedOverRange;
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

    /// Splits the terms in `first..last` into plain terms, in their order and including identity
    /// terms, and towers of equal-angle rotations long enough for Hamming-weight phasing. See
    /// `EqualAngleTowers`.
    function LayerRotationTowers(
        params : RepPauliExpParams,
        first : Int,
        last : Int,
        maxBatchSize : Int
    ) : (Int[], Int[][]) {
        // An identity term takes angle 0, which never joins a tower.
        let angles = MappedOverRange(
            term -> Length(params.pauliIndices[term]) > 0 ? params.pauliCoefficients[term] | 0.0,
            first..last
        );
        let (plain, towers) = EqualAngleTowers(angles, maxBatchSize);
        (Mapped(position -> first + position, plain), Mapped(tower -> Mapped(position -> first + position, tower), towers))
    }

    /// Applies the producer's disjoint layers without changing their boundaries.
    ///
    /// Within a layer, terms that share a rotation angle form towers once at least 8 of them
    /// agree, and each tower is phased through a Hamming-weight register :cite:`Kan2025` instead
    /// of one controlled rotation per term. `maxBatchSize` caps the terms per register; see
    /// `HammingWeightBatchSize`.
    operation ControlledPauliLayers(
        params : RepPauliExpParams,
        layerOffsets : Int[],
        maxBatchSize : Int,
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        for layer in 0..Length(layerOffsets) - 2 {
            let first = layerOffsets[layer];
            let last = layerOffsets[layer + 1] - 1;
            if first == last {
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
                    let (plainTerms, towers) = LayerRotationTowers(params, first, last, maxBatchSize);
                    for term in plainTerms {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
                            Rz(params.pauliCoefficients[term], systems[indices[0]]);
                        }
                    }
                    for term in plainTerms {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
                            CNOT(control, systems[indices[0]]);
                        }
                    }
                    for term in plainTerms {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
                            Rz(-params.pauliCoefficients[term], systems[indices[0]]);
                        }
                    }
                    for term in plainTerms {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
                            CNOT(control, systems[indices[0]]);
                        }
                    }
                    for tower in towers {
                        // Every term already sits on its first qubit as a single Z.
                        Controlled HammingWeightPhase([control], (
                            params.pauliCoefficients[tower[0]],
                            [[PauliZ], size = Length(tower)],
                            Mapped(term -> [systems[params.pauliIndices[term][0]]], tower),
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
    /// - `maxBatchSize`: Largest tower of equal-angle rotations in a declared layer phased through
    ///   one Hamming-weight register, or -1 for no cap. A cap below 8 turns phasing off.
    /// - `control`: The control qubit.
    /// - `systems`: The system qubits the evolution acts on.
    operation RepControlledPauliExp(
        params : RepPauliExpParams,
        layerOffsets : Int[],
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
            ControlledPauliLayers(params, layerOffsets[0..first], maxBatchSize, control, systems);
            if IsResourceEstimating() {
                within {
                    RepeatEstimates(params.repetitions);
                } apply {
                    ControlledPauliLayers(params, stepOffsets, maxBatchSize, control, systems);
                }
            } else {
                for _ in 1..params.repetitions {
                    ControlledPauliLayers(params, stepOffsets, maxBatchSize, control, systems);
                }
            }
            ControlledPauliLayers(params, layerOffsets[last...], maxBatchSize, control, systems);
        }
    }

    /// A helper operation to create a circuit for repeated Controlled Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: The sparse repeated Pauli evolution parameters.
    /// - `layerOffsets`: Declared disjoint-layer boundaries; empty means term-by-term evolution.
    /// - `maxBatchSize`: Largest Hamming-weight phasing batch, or -1 for no cap.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of integers representing the indices of the system qubits.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated controlled time evolution on the allocated qubits.
    operation MakeRepControlledPauliExpCircuit(
        params : RepPauliExpParams,
        layerOffsets : Int[],
        maxBatchSize : Int,
        control : Int,
        systems : Int[]
    ) : Unit {
        use qs = Qubit[Max([control] + systems) + 1];
        RepControlledPauliExp(params, layerOffsets, maxBatchSize, qs[control], Subarray(systems, qs));
    }

    /// Returns a single-control callable for repeated sparse Pauli evolution.
    function MakeRepControlledPauliExpOp(
        params : RepPauliExpParams,
        layerOffsets : Int[],
        maxBatchSize : Int
    ) : ((Qubit, Qubit[]) => Unit is Adj + Ctl) {
        RepControlledPauliExp(params, layerOffsets, maxBatchSize, _, _)
    }
}
