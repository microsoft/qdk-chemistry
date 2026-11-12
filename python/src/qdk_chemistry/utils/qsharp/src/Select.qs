// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SELECT oracle for LCU (Linear Combination of Unitaries) block encodings.
///
/// Provides the default Pauli-based SELECT implementation.  The generic
/// PREPARE-SELECT stitching lives in ``QDKChemistry.Utils.PrepSelPrep``.
namespace QDKChemistry.Utils.Select {

    import Std.Arrays.Fold;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyPauli;
    import Std.Core.Length;
    import Std.Core.Repeated;
    import Std.Diagnostics.Fact;
    import Std.Math.MaxI;
    import Std.Math.PI;
    import QDKChemistry.Utils.UnaryIteration.ControlledUnaryIterationWithControl;

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

    /// `PauliSelect` by unary iteration (arXiv:1805.03662, Sec. III A): L - 2 Toffolis for L terms
    /// uncontrolled, L - 1 with one control.
    operation UnaryPauliSelect(params : PauliSelectParams, selectRegister : Qubit[], targets : Qubit[]) : Unit is Adj + Ctl {
        body ... {
            Controlled UnaryPauliSelect([], (params, selectRegister, targets));
        }
        adjoint self;
        controlled (ctls, ...) {
            let termAt = TermTable(params.controlStates);
            ControlledUnaryIterationWithControl(ctls, selectRegister, Length(termAt), (index, ctl) => {
                ApplySignedPauliTerm(params, termAt[index], ctl, targets);
            });
        }
        controlled adjoint self;
    }

    /// Term index per control state, -1 where none exists.
    internal function TermTable(controlStates : Int[]) : Int[] {
        Fact(Length(controlStates) > 0, "SELECT needs at least one term.");
        mutable termAt = Repeated(-1, Fold(MaxI, 0, controlStates) + 1);
        for i in 0..Length(controlStates) - 1 {
            Fact(controlStates[i] >= 0, "SELECT control states must be nonnegative.");
            Fact(termAt[controlStates[i]] < 0, "SELECT control states must be distinct.");
            set termAt w/= controlStates[i] <- i;
        }
        termAt
    }

    /// Applies `sign · P` for `term` when `ctl` is |1⟩; a negative `term` applies nothing.
    internal operation ApplySignedPauliTerm(params : PauliSelectParams, term : Int, ctl : Qubit, targets : Qubit[]) : Unit is Adj {
        if term >= 0 {
            Controlled ApplyPauli([ctl], (params.pauliTerms[term], targets));
            if params.signs[term] < 0 {
                Z(ctl);
            }
        }
    }

    /// Creates a SELECT callable from parameters.
    function MakeSelectOp(params : PauliSelectParams) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        PauliSelect(params, _, _)
    }

    /// Creates a unary-iteration SELECT callable from parameters.
    function MakeUnarySelectOp(params : PauliSelectParams) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        UnaryPauliSelect(params, _, _)
    }

    /// Creates a circuit for the SELECT oracle by allocating qubits and applying the operation.
    operation MakeSelectCircuit(params : PauliSelectParams, numSelectQubits : Int, numTargetQubits : Int) : Unit {
        use selectRegister = Qubit[numSelectQubits];
        use targets = Qubit[numTargetQubits];
        PauliSelect(params, selectRegister, targets);
    }

    /// Creates a circuit for the unary-iteration SELECT oracle.
    operation MakeUnarySelectCircuit(params : PauliSelectParams, numSelectQubits : Int, numTargetQubits : Int) : Unit {
        use selectRegister = Qubit[numSelectQubits];
        use targets = Qubit[numTargetQubits];
        UnaryPauliSelect(params, selectRegister, targets);
    }
}
