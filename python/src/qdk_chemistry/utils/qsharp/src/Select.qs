// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SELECT oracle for LCU (Linear Combination of Unitaries) block encodings.
///
/// Provides the default Pauli-based SELECT implementation.  The generic
/// PREPARE-SELECT stitching lives in ``QDKChemistry.Utils.PrepSelPrep``.
namespace QDKChemistry.Utils.Select {

    import Std.Arrays.Fold;
    import Std.Arrays.MostAndTail;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyPauli;
    import Std.Core.Length;
    import Std.Core.Repeated;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.BitSizeI;
    import Std.Math.MaxI;
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

    /// Pauli SELECT oracle that decodes the index by unary iteration.
    ///
    /// Implements the same map as `PauliSelect` on every index state that names a term,
    /// following Babbush et al., arXiv:1805.03662, Sec. III A (Figs. 5-7). Each term costs at
    /// most one AND and a singly-controlled Pauli string: L - 2 Toffolis for L terms (L - 1
    /// under a control), where `PauliSelect` multi-controls every factor of every Pauli string.
    /// Without mid-circuit measurement each AND is also uncomputed by a Toffoli.
    ///
    /// Index states past the largest control state are routed onto existing terms, as the
    /// iteration's binary tree aliases them, and control states missing from `params` apply
    /// the identity. A PREPARE gives the aliased states no amplitude, so the block encoding
    /// is unchanged. The oracle stays Hermitian and self-inverse.
    ///
    /// The recursion is written on classical data rather than through a callable so that it
    /// lowers to QIR under every target profile.
    operation UnaryPauliSelect(params : PauliSelectParams, selectRegister : Qubit[], targets : Qubit[]) : Unit is Adj + Ctl {
        body ... {
            UnaryPauliSelectImpl(params, [], selectRegister, targets);
        }
        adjoint self;
        controlled (ctls, ...) {
            UnaryPauliSelectImpl(params, ctls, selectRegister, targets);
        }
        controlled adjoint self;
    }

    internal operation UnaryPauliSelectImpl(
        params : PauliSelectParams,
        ctls : Qubit[],
        selectRegister : Qubit[],
        targets : Qubit[],
    ) : Unit {
        let numTerms = Length(params.pauliTerms);
        Fact(numTerms > 0, "SELECT needs at least one term.");
        Fact(Length(params.signs) == numTerms, "SELECT needs one sign per term.");
        Fact(Length(params.controlStates) == numTerms, "SELECT needs one control state per term.");
        let numActions = Fold(MaxI, 0, params.controlStates) + 1;
        let termAt = TermTable(params.controlStates, numActions);
        let n = BitSizeI(numActions - 1);
        Fact(Length(selectRegister) >= n, $"SELECT needs at least {n} index qubits.");

        if n == 0 {
            ApplyUnindexedTerm(params, termAt[0], ctls, targets);
        } elif Length(ctls) == 0 {
            // The top level needs no AND: the most significant index qubit is the control.
            let (most, tail) = MostAndTail(selectRegister[...n - 1]);
            let half = 1 <<< (n - 1);
            within {
                X(tail);
            } apply {
                SelectTermsUnder(params, termAt, tail, most, half, 0, targets);
            }
            SelectTermsUnder(params, termAt, tail, most, numActions - half, half, targets);
        } elif Length(ctls) == 1 {
            SelectTermsUnder(params, termAt, ctls[0], selectRegister, numActions, 0, targets);
        } else {
            use anc = Qubit[Length(ctls) - 1];
            within {
                AND(ctls[0], ctls[1], anc[0]);
                for i in 2..Length(ctls) - 1 {
                    AND(anc[i - 2], ctls[i], anc[i - 1]);
                }
            } apply {
                SelectTermsUnder(params, termAt, anc[Length(ctls) - 2], selectRegister, numActions, 0, targets);
            }
        }
    }

    /// Maps each control state in `0..numActions - 1` to its term, or -1 where none exists.
    internal function TermTable(controlStates : Int[], numActions : Int) : Int[] {
        mutable termAt = Repeated(-1, numActions);
        for i in 0..Length(controlStates) - 1 {
            let state = controlStates[i];
            Fact(state >= 0, "SELECT control states must be nonnegative.");
            Fact(termAt[state] < 0, "SELECT control states must be distinct.");
            set termAt w/= state <- i;
        }
        termAt
    }

    /// Applies the terms at control states `offset..offset + numActions - 1`, each when `ctl`
    /// is |1⟩ and `address` holds its offset from `offset` (one level of Fig. 5 of
    /// arXiv:1805.03662).
    internal operation SelectTermsUnder(
        params : PauliSelectParams,
        termAt : Int[],
        ctl : Qubit,
        address : Qubit[],
        numActions : Int,
        offset : Int,
        targets : Qubit[],
    ) : Unit {
        if numActions == 1 {
            ApplySignedPauliTerm(params, termAt[offset], ctl, targets);
        } else {
            let n = BitSizeI(numActions - 1);
            let half = 1 <<< (n - 1);
            let (most, tail) = MostAndTail(address[...n - 1]);
            use helper = Qubit();
            within {
                X(tail);
            } apply {
                ExactAnd(ctl, tail, helper);
            }
            SelectTermsUnder(params, termAt, helper, most, half, offset, targets);
            CNOT(ctl, helper);
            SelectTermsUnder(params, termAt, helper, most, numActions - half, offset + half, targets);
            Adjoint ExactAnd(ctl, tail, helper);
        }
    }

    /// Computes `control1 ∧ control2` into a |0⟩ target exactly, uncomputing it by measurement.
    @Config(Adaptive)
    internal operation ExactAnd(control1 : Qubit, control2 : Qubit, target : Qubit) : Unit is Adj {
        AND(control1, control2, target);
    }

    /// Computes `control1 ∧ control2` into a |0⟩ target exactly.
    ///
    /// Without mid-circuit measurement `AND` is a Toffoli only up to a relative phase, which
    /// cancels against its own adjoint but not once unary iteration has toggled the target
    /// with a CNOT in between, so this falls back to `CCNOT` both ways.
    @Config(not Adaptive)
    internal operation ExactAnd(control1 : Qubit, control2 : Qubit, target : Qubit) : Unit is Adj {
        CCNOT(control1, control2, target);
    }

    /// Applies `sign · P` for `term` when `ctl` is |1⟩; a negative `term` applies nothing.
    internal operation ApplySignedPauliTerm(params : PauliSelectParams, term : Int, ctl : Qubit, targets : Qubit[]) : Unit {
        if term >= 0 {
            Controlled ApplyPauli([ctl], (params.pauliTerms[term], targets));
            if params.signs[term] < 0 {
                Z(ctl);
            }
        }
    }

    /// The SELECT of a lone term, which needs no index qubits.
    internal operation ApplyUnindexedTerm(params : PauliSelectParams, term : Int, ctls : Qubit[], targets : Qubit[]) : Unit {
        Controlled ApplyPauli(ctls, (params.pauliTerms[term], targets));
        if params.signs[term] < 0 {
            if Length(ctls) == 0 {
                // Keep the sign visible under an outer control: R(PauliI, 2π) applies -I.
                R(PauliI, 2.0 * PI(), targets[0]);
            } else {
                Controlled Z(ctls[1...], ctls[0]);
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
