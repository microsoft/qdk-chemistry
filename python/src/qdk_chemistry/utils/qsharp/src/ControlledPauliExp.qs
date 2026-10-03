// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.ControlledPauliExp {

    import QDKChemistry.Utils.Loop.LoopCA;
    import QDKChemistry.Utils.PauliExp.RepPauliExp;
    import QDKChemistry.Utils.PauliExp.RepPauliExpParams;
    import Std.Arrays.IndexOf;
    import Std.Arrays.Subarray;
    import Std.Canon.MapPauliAxis;
    import Std.Math.Max;

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

    /// Applies the producer's disjoint layers without changing their boundaries.
    operation ControlledPauliLayers(
        params : RepPauliExpParams,
        layerOffsets : Int[],
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
                    for term in first..last {
                        let indices = params.pauliIndices[term];
                        if Length(indices) > 0 {
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
        params : RepPauliExpParams,
        layerOffsets : Int[],
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        if Length(layerOffsets) == 0 {
            Controlled RepPauliExp([control], (params, systems));
        } else {
            let first = IndexOf(offset -> offset == params.numPrefixTerms, layerOffsets);
            let last = IndexOf(offset -> offset == Length(params.pauliCoefficients) - params.numSuffixTerms, layerOffsets);
            let stepOffsets = layerOffsets[first..last];
            ControlledPauliLayers(params, layerOffsets[0..first], control, systems);
            LoopCA(params.repetitions, 0, _ => ControlledPauliLayers(params, stepOffsets, control, systems));
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
        params : RepPauliExpParams,
        layerOffsets : Int[],
        control : Int,
        systems : Int[]
    ) : Unit {
        use qs = Qubit[Max([control] + systems) + 1];
        RepControlledPauliExp(params, layerOffsets, qs[control], Subarray(systems, qs));
    }

    /// Returns a single-control callable for repeated sparse Pauli evolution.
    function MakeRepControlledPauliExpOp(
        params : RepPauliExpParams,
        layerOffsets : Int[]
    ) : ((Qubit, Qubit[]) => Unit is Adj + Ctl) {
        RepControlledPauliExp(params, layerOffsets, _, _)
    }
}
