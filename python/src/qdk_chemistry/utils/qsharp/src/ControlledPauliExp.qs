// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledPauliExp {

    import QDKChemistry.Utils.CircuitComposition.MaxInt;
    import QDKChemistry.Utils.PauliExp.SparseRepPauliExp;
    import QDKChemistry.Utils.PauliExp.SparseRepPauliExpParams;
    import Std.Arrays.IndexOf;
    import Std.Arrays.Subarray;
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.ResourceEstimation.RepeatEstimates;

    /// Applies the producer's disjoint layers without changing their boundaries.
    operation ControlledPauliLayers(
        params : SparseRepPauliExpParams,
        layerOffsets : Int[],
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        for layer in 0..Length(layerOffsets) - 2 {
            let first = layerOffsets[layer];
            let last = layerOffsets[layer + 1] - 1;
            if first == last {
                if Length(params.pauliIndices[first]) == 0 {
                    R1(-params.pauliCoefficients[first], control);
                } else {
                    Controlled Exp([control], (
                        params.pauliOps[first], -params.pauliCoefficients[first],
                        Subarray(params.pauliIndices[first], systems)
                    ));
                }
            } else {
                within {
                    for term in first..last {
                        let indices = params.pauliIndices[term];
                        let paulis = params.pauliOps[term];
                        for position in 0..Length(indices) - 1 {
                            let q = systems[indices[position]];
                            if paulis[position] == PauliX {
                                H(q);
                            } elif paulis[position] == PauliY {
                                Adjoint S(q);
                                H(q);
                            }
                        }
                        if Length(indices) > 0 {
                            for position in 1..Length(indices) - 1 {
                                CNOT(systems[indices[position]], systems[indices[0]]);
                            }
                        }
                    }
                } apply {
                    // Rotations share two rounds, but the CNOTs still share one control.
                    for term in first..last {
                        let indices = params.pauliIndices[term];
                        if Length(indices) == 0 {
                            // Identity terms retain their relative phase, including under further controls.
                            R1(-params.pauliCoefficients[term], control);
                        } else {
                            Rz(params.pauliCoefficients[term], systems[indices[0]]);
                        }
                    }
                    for term in first..last {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
                            CNOT(control, systems[indices[0]]);
                        }
                    }
                    for term in first..last {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
                            Rz(-params.pauliCoefficients[term], systems[indices[0]]);
                        }
                    }
                    for term in first..last {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
                            CNOT(control, systems[indices[0]]);
                        }
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
    /// - `control`: The control qubit.
    /// - `systems`: The system qubits the evolution acts on.
    operation RepControlledPauliExp(
        params : SparseRepPauliExpParams,
        layerOffsets : Int[],
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        if Length(layerOffsets) == 0 {
            Controlled SparseRepPauliExp([control], (params, systems));
        } else {
            let first = IndexOf(offset -> offset == params.beginning, layerOffsets);
            let last = IndexOf(offset -> offset == Length(params.pauliCoefficients) - params.end, layerOffsets);
            let stepOffsets = layerOffsets[first..last];
            ControlledPauliLayers(params, layerOffsets[0..first], control, systems);
            if IsResourceEstimating() {
                within {
                    RepeatEstimates(params.repetitions);
                } apply {
                    ControlledPauliLayers(params, stepOffsets, control, systems);
                }
            } else {
                for _ in 1..params.repetitions {
                    ControlledPauliLayers(params, stepOffsets, control, systems);
                }
            }
            ControlledPauliLayers(params, layerOffsets[last...], control, systems);
        }
    }

    /// A helper operation to create a circuit for repeated Controlled Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: The sparse repeated Pauli evolution parameters.
    /// - `layerOffsets`: Declared disjoint-layer boundaries; empty means term-by-term evolution.
    /// - `control`: The index of the control qubit.
    /// - `systems`: An array of integers representing the indices of the system qubits.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated controlled time evolution on the allocated qubits.
    operation MakeRepControlledPauliExpCircuit(
        params : SparseRepPauliExpParams,
        layerOffsets : Int[],
        control : Int,
        systems : Int[]
    ) : Unit {
        use qs = Qubit[MaxInt([control] + systems) + 1];
        RepControlledPauliExp(params, layerOffsets, qs[control], Subarray(systems, qs));
    }

    /// Returns a single-control callable for repeated sparse Pauli evolution.
    function MakeRepControlledPauliExpOp(
        params : SparseRepPauliExpParams,
        layerOffsets : Int[]
    ) : ((Qubit, Qubit[]) => Unit is Adj + Ctl) {
        RepControlledPauliExp(params, layerOffsets, _, _)
    }
}
