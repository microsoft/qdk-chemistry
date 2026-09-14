// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.BatchedControlledPauliExp {

    import QDKChemistry.Utils.CircuitComposition.MaxInt;
    import QDKChemistry.Utils.PauliExp.SparseRepPauliExpParams;
    import Std.Arrays.Subarray;
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.ResourceEstimation.RepeatEstimates;

    /// Applies one step in consecutive disjoint-support batches.
    /// batchOffsets indexes terms; empty supports must be singleton batches.
    operation ControlledPauliExp(
        params : SparseRepPauliExpParams,
        batchOffsets : Int[],
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        for batch in 0..Length(batchOffsets) - 2 {
            let first = batchOffsets[batch];
            let last = batchOffsets[batch + 1] - 1;
            if Length(params.pauliIndices[first]) == 0 {
                // Retain the relative phase, including under further controls.
                R1(-params.pauliCoefficients[first], control);
            } elif first == last {
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
                            let q = systems[indices[position]];
                            if paulis[position] == PauliX {
                                H(q);
                            } elif paulis[position] == PauliY {
                                Adjoint S(q);
                                H(q);
                            }
                        }
                        let target = systems[indices[0]];
                        for position in 1..Length(indices) - 1 {
                            CNOT(systems[indices[position]], target);
                        }
                    }
                } apply {
                    // Two rotation rounds across disjoint data supports;
                    // the CNOTs still share the control and are not simultaneous.
                    for term in first..last {
                        Rz(params.pauliCoefficients[term], systems[params.pauliIndices[term][0]]);
                    }
                    for term in first..last {
                        CNOT(control, systems[params.pauliIndices[term][0]]);
                    }
                    for term in first..last {
                        Rz(-params.pauliCoefficients[term], systems[params.pauliIndices[term][0]]);
                    }
                    for term in first..last {
                        CNOT(control, systems[params.pauliIndices[term][0]]);
                    }
                }
            }
        }
    }

    /// Keeps repetitions symbolic during resource estimation.
    operation RepControlledPauliExp(
        params : SparseRepPauliExpParams,
        batchOffsets : Int[],
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        if IsResourceEstimating() {
            within {
                RepeatEstimates(params.repetitions);
            } apply {
                ControlledPauliExp(params, batchOffsets, control, systems);
            }
        } else {
            for _ in 1..params.repetitions {
                ControlledPauliExp(params, batchOffsets, control, systems);
            }
        }
    }

    /// Creates a batched controlled circuit with explicit physical register indices.
    operation MakeRepControlledPauliExpCircuit(
        params : SparseRepPauliExpParams,
        batchOffsets : Int[],
        control : Int,
        systems : Int[]
    ) : Unit {
        use qs = Qubit[MaxInt([control] + systems) + 1];
        RepControlledPauliExp(params, batchOffsets, qs[control], Subarray(systems, qs));
    }

    /// Named partial application remains composable in QPE and QIR.
    function MakeRepControlledPauliExpOp(
        params : SparseRepPauliExpParams,
        batchOffsets : Int[]
    ) : ((Qubit, Qubit[]) => Unit is Adj + Ctl) {
        RepControlledPauliExp(params, batchOffsets, _, _)
    }
}
