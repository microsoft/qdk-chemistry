// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.Arithmetic {

    import Std.Canon.ApplyPauliFromBitString;
    import Std.Convert.IntAsBoolArray;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;

    /// # Summary
    /// Increments `ys` by `xs` modulo 2^n with Gidney's ripple-carry adder, applying
    /// `carryAction` to the carry-out of the sum while it is held.
    ///
    /// # Input
    /// ## carryAction
    /// The operation applied to the carry-out qubit.
    /// ## xs
    /// The little-endian addend, returned unchanged.
    /// ## ys
    /// The little-endian register incremented by `xs`, as long as `xs`.
    internal operation RippleCarryCGIncByLEWithCarryAction(
        carryAction : Qubit => Unit is Adj,
        xs : Qubit[],
        ys : Qubit[]
    ) : Unit is Adj {
        let n = Length(xs);
        Fact(n > 0 and Length(ys) == n, "The adder needs two non-empty registers of equal length.");
        use carries = Qubit[n];
        AND(xs[0], ys[0], carries[0]);
        for i in 1..n - 1 {
            CNOT(carries[i - 1], xs[i]);
            CNOT(carries[i - 1], ys[i]);
            AND(xs[i], ys[i], carries[i]);
            CNOT(carries[i - 1], carries[i]);
        }
        carryAction(carries[n - 1]);
        for i in n - 1..-1..1 {
            CNOT(carries[i - 1], carries[i]);
            Adjoint AND(xs[i], ys[i], carries[i]);
            CNOT(carries[i - 1], xs[i]);
            CNOT(xs[i], ys[i]);
        }
        Adjoint AND(xs[0], ys[0], carries[0]);
        CNOT(xs[0], ys[0]);
    }

    /// Applies |x⟩ → |x + c mod 2^n⟩ to the little-endian `target` with the ripple-carry
    /// adder of Sanders et al. (PRX Quantum 1, 020312, 2020), Fig. 18.
    ///
    /// For n > 2 it costs n − 2 AND gates (4(n − 2) T gates) and n − 1 clean ancillas, and the
    /// AND gates are uncomputed by measurement. Smaller widths use only Clifford gates.
    operation AddConstant(c : Int, target : Qubit[]) : Unit is Adj {
        let n = Length(target);
        let cBits = IntAsBoolArray(c &&& ((1 <<< n) - 1), n);
        if n == 2 {
            // The only carry, c_0 ∧ t_0, needs no ancilla.
            if cBits[0] { CNOT(target[0], target[1]); }
        } elif n > 2 {
            // carries[i] = carry k into bit i + 1, from maj(t, c, k) = k ⊕ ((k ⊕ c) ∧ (t ⊕ k)).
            use carries = Qubit[n - 1];
            within {
                if cBits[0] { CNOT(target[0], carries[0]); }
            } apply {
                for i in 1..n - 2 {
                    CNOT(carries[i - 1], target[i]);
                    within { if cBits[i] { X(carries[i - 1]); } } apply { AND(carries[i - 1], target[i], carries[i]); }
                    CNOT(carries[i - 1], carries[i]);
                }
                CNOT(carries[n - 2], target[n - 1]);
                // Erase the carries, leaving target[i] = t_i ⊕ k_i.
                for i in n - 2..-1..1 {
                    CNOT(carries[i - 1], carries[i]);
                    within { if cBits[i] { X(carries[i - 1]); } } apply { Adjoint AND(carries[i - 1], target[i], carries[i]); }
                }
            }
        }
        ApplyPauliFromBitString(PauliX, true, cBits, target);
    }
}
