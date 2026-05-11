// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// LCU (Linear Combination of Unitaries) SELECT oracle.
///
/// Provides the default Pauli-based SELECT implementation used in LCU
/// block encodings.  The generic PREPARE-SELECT stitching lives in
/// ``QDKChemistry.Utils.PrepareSelect``.
namespace QDKChemistry.Utils.LCU {

    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyPauli;
    import Std.Core.Length;

    /// Parameters for the default Pauli SELECT oracle.
    struct DefaultSelectParams {
        pauliTerms : Pauli[][],
        signs : Int[],
    }

    /// Default SELECT oracle: applies Pauli string P_j controlled on ancilla state |j⟩,
    /// with a sign flip (global phase of π) for negative coefficients.
    ///
    /// $$
    ///     \mathrm{SELECT} = \sum_j |j\rangle\langle j| \otimes \mathrm{sign}(\alpha_j) \cdot P_j
    /// $$
    operation DefaultSelect(params : DefaultSelectParams, selectRegister : Qubit[], targets : Qubit[]) : Unit is Adj + Ctl {
        let numUnitary = Length(params.pauliTerms);
        for i in 0..numUnitary - 1 {
            let U = params.pauliTerms[i];
            ApplyControlledOnInt(i, ApplyPauli(U, _), selectRegister, targets);
            if params.signs[i] < 0 and Length(selectRegister) > 0 {
                // Apply -1 phase when selectRegister is in state |i⟩.
                // Flip bits so |i⟩ → |1…1⟩, apply multi-controlled Z, then unflip.
                within {
                    for k in 0..Length(selectRegister) - 1 {
                        if (i >>> k) &&& 1 == 0 {
                            X(selectRegister[k]);
                        }
                    }
                } apply {
                    Controlled Z(selectRegister[1...], selectRegister[0]);
                }
            }
        }
    }

    /// Creates a SELECT callable from parameters.
    function MakeSelectOp(params : DefaultSelectParams) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        DefaultSelect(params, _, _)
    }

    /// Creates a circuit for the SELECT oracle by allocating qubits and applying the operation.
    operation MakeSelectCircuit(params : DefaultSelectParams, numSelectQubits : Int, numTargetQubits : Int) : Unit {
        use selectRegister = Qubit[numSelectQubits];
        use targets = Qubit[numTargetQubits];
        DefaultSelect(params, selectRegister, targets);
    }
}
