// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.Arithmetic {

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
}
