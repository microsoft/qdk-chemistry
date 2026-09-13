// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.BatchedControlledPauliExp {

    import QDKChemistry.Utils.CircuitComposition.MaxInt;
    import Std.Arrays.Subarray;
    import Std.ResourceEstimation.*;

    /// Batches consecutive, disjoint-support terms without changing their product.
    /// batchOffsets indexes terms; termOffsets indexes their nonidentity factors.
    /// Empty supports must be singleton batches so their controlled phases are retained.
    operation ControlledSparsePauliExp(
        termOffsets : Int[],
        qubitIndices : Int[],
        paulis : Pauli[],
        pauliCoefficients : Double[],
        batchOffsets : Int[],
        control : Qubit,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        for batch in 0..Length(batchOffsets) - 2 {
            let first = batchOffsets[batch];
            let last = batchOffsets[batch + 1] - 1;
            if termOffsets[first] == termOffsets[first + 1] {
                // A phase on |1> also remains correct under further controls;
                // a target-only Rz would introduce a control-relative phase.
                R1(-pauliCoefficients[first], control);
            } elif first == last {
                let range = termOffsets[first]..termOffsets[first + 1] - 1;
                Controlled Exp([control], (
                    paulis[range], -pauliCoefficients[first], Subarray(qubitIndices[range], systems)
                ));
            } else {
                within {
                    for position in termOffsets[first]..termOffsets[last + 1] - 1 {
                        let q = systems[qubitIndices[position]];
                        if paulis[position] == PauliX {
                            H(q);
                        } elif paulis[position] == PauliY {
                            Adjoint S(q);
                            H(q);
                        }
                    }
                    for term in first..last {
                        let start = termOffsets[term];
                        let target = systems[qubitIndices[start]];
                        for position in start + 1..termOffsets[term + 1] - 1 {
                            CNOT(systems[qubitIndices[position]], target);
                        }
                    }
                } apply {
                    // Interleave only disjoint terms' CRz gadgets: two rotation
                    // rounds, not one complete gadget at a time. CNOTs still share control.
                    for term in first..last {
                        Rz(pauliCoefficients[term], systems[qubitIndices[termOffsets[term]]]);
                    }
                    for term in first..last {
                        CNOT(control, systems[qubitIndices[termOffsets[term]]]);
                    }
                    for term in first..last {
                        Rz(-pauliCoefficients[term], systems[qubitIndices[termOffsets[term]]]);
                    }
                    for term in first..last {
                        CNOT(control, systems[qubitIndices[termOffsets[term]]]);
                    }
                }
            }
        }
    }

    /// Repeats the batched step symbolically during resource estimation.
    operation RepControlledSparsePauliExp(
        termOffsets : Int[],
        qubitIndices : Int[],
        paulis : Pauli[],
        pauliCoefficients : Double[],
        batchOffsets : Int[],
        repetitions : Int,
        control : Qubit,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        if IsResourceEstimating() {
            within {
                RepeatEstimates(repetitions);
            } apply {
                ControlledSparsePauliExp(
                    termOffsets, qubitIndices, paulis, pauliCoefficients, batchOffsets, control, systems
                );
            }
        } else {
            for _ in 1..repetitions {
                ControlledSparsePauliExp(
                    termOffsets, qubitIndices, paulis, pauliCoefficients, batchOffsets, control, systems
                );
            }
        }
    }

    /// Creates controlled sparse evolution with explicit physical register indices.
    operation MakeRepControlledSparsePauliExpCircuit(
        termOffsets : Int[],
        qubitIndices : Int[],
        paulis : Pauli[],
        pauliCoefficients : Double[],
        batchOffsets : Int[],
        repetitions : Int,
        control : Int,
        systems : Int[],
    ) : Unit {
        use qs = Qubit[MaxInt([control] + systems) + 1];
        RepControlledSparsePauliExp(
            termOffsets, qubitIndices, paulis, pauliCoefficients, batchOffsets,
            repetitions, qs[control], Subarray(systems, qs)
        );
    }

    /// Returns a named callable suitable for QPE and QIR composition.
    function MakeRepControlledSparsePauliExpOp(
        termOffsets : Int[],
        qubitIndices : Int[],
        paulis : Pauli[],
        pauliCoefficients : Double[],
        batchOffsets : Int[],
        repetitions : Int,
    ) : (Qubit, Qubit[]) => Unit is Adj + Ctl {
        RepControlledSparsePauliExp(termOffsets, qubitIndices, paulis, pauliCoefficients, batchOffsets, repetitions, _, _)
    }
}
