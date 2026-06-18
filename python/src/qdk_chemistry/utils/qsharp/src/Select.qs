// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SELECT oracle for LCU (Linear Combination of Unitaries) block encodings.
///
/// Provides the default Pauli-based SELECT implementation.  The generic
/// PREPARE-SELECT stitching lives in ``QDKChemistry.Utils.PrepSelPrep``.
namespace QDKChemistry.Utils.Select {

    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyPauli;
    import Std.Core.Length;
    import Std.Math.PI;

    /// Parameters for the Pauli-based SELECT oracle.
    struct PauliSelectParams {
        pauliTerms : Pauli[][],
        signs : Int[],
        controlStates : Int[],
    }

    /// Pauli SELECT oracle: applies Pauli string P_j controlled on ancilla state |j⟩,
    /// with a sign flip (global phase of π) for negative coefficients.
    ///
    /// $$
    ///     \mathrm{SELECT} = \sum_j |j\rangle\langle j| \otimes \mathrm{sign}(\alpha_j) \cdot P_j
    /// $$
    operation PauliSelect(params : PauliSelectParams, selectRegister : Qubit[], targets : Qubit[]) : Unit is Adj + Ctl {
        let numUnitary = Length(params.pauliTerms);
        for i in 0..numUnitary - 1 {
            let U = params.pauliTerms[i];
            let ctrlState = params.controlStates[i];
            ApplyControlledOnInt(ctrlState, ApplyPauli(U, _), selectRegister, targets);
            if params.signs[i] < 0 {
                if Length(selectRegister) > 0 {
                    // Apply -1 phase when selectRegister is in state |ctrlState⟩.
                    // Flip bits so |ctrlState⟩ → |1…1⟩, apply multi-controlled Z, then unflip.
                    within {
                        for k in 0..Length(selectRegister) - 1 {
                            if (((ctrlState >>> k) &&& 1) == 0) {
                                X(selectRegister[k]);
                            }
                        }
                    } apply {
                        Controlled Z(selectRegister[1...], selectRegister[0]);
                    }
                } else {
                    // No select qubits (single-term Hamiltonian). Apply -1 as a global
                    // phase so the sign is still visible under outer control (e.g. QPE).
                    // R(PauliI, θ, q) applies exp(-iθ/2)·I; with θ = 2π this gives -I.
                    R(PauliI, 2.0 * PI(), targets[0]);
                }
            }
        }
    }

    /// Creates a SELECT callable from parameters.
    function MakeSelectOp(params : PauliSelectParams) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        PauliSelect(params, _, _)
    }

    /// Creates a circuit for the SELECT oracle by allocating qubits and applying the operation.
    operation MakeSelectCircuit(params : PauliSelectParams, numSelectQubits : Int, numTargetQubits : Int) : Unit {
        use selectRegister = Qubit[numSelectQubits];
        use targets = Qubit[numTargetQubits];
        PauliSelect(params, selectRegister, targets);
    }
}
