// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// General Dicke state preparation utilities.
///
/// A Dicke state $|D^n_k\rangle$ is the uniform (equal-amplitude) superposition
/// of all $n$-qubit computational-basis states of Hamming weight $k$:
///
/// $$
///     |D^n_k\rangle = \binom{n}{k}^{-1/2} \sum_{|x| = k} |x\rangle .
/// $$
///
/// This module exposes a standalone, structure-preserving preparation routine
/// that is independent of any particular block encoding.  The weight-1 case
/// reuses the balanced Dicke building block from the FOQCS utilities.
namespace QDKChemistry.Utils.Dicke {

    import Std.Core.Length;
    import QDKChemistry.Utils.Foqcs.BalancedDicke1Excitation;

    /// Description of a uniform Dicke state $|D^n_k\rangle$.
    /// - `numQubits`: The register size ``n``.
    /// - `weight`: The Hamming weight ``k`` of the superposed basis states.
    struct DickeParams {
        numQubits : Int,
        weight : Int,
    }

    /// Prepare the uniform Dicke state $|D^n_k\rangle$ on the |0...0> register.
    ///
    /// Currently only weight ``k = 1`` (the uniform one-hot superposition) is
    /// supported; other weights raise a runtime failure.
    /// # Parameters
    /// - `params`: The Dicke-state description.
    /// - `qs`: The target register (assumed to be in the all-zero state).
    operation PrepareDicke(params : DickeParams, qs : Qubit[]) : Unit is Adj + Ctl {
        if params.weight != 1 {
            fail "Dicke state preparation currently supports only weight-1 states.";
        }
        BalancedDicke1Excitation(qs, true, 0);
    }

    /// Create a callable that prepares a Dicke state on a register.
    /// # Parameters
    /// - `params`: The Dicke-state description.
    /// # Returns
    /// - A callable that takes a register and prepares the Dicke state on it.
    function MakeDickeOp(params : DickeParams) : (Qubit[] => Unit is Adj + Ctl) {
        PrepareDicke(params, _)
    }

    /// Circuit entry point for circuit extraction: allocate a register and
    /// prepare the Dicke state on it.
    /// # Parameters
    /// - `numQubits`: The register size ``n``.
    /// - `weight`: The Hamming weight ``k``.
    operation MakeDickeCircuit(numQubits : Int, weight : Int) : Unit {
        use qs = Qubit[numQubits];
        PrepareDicke(new DickeParams { numQubits = numQubits, weight = weight }, qs);
    }
}
