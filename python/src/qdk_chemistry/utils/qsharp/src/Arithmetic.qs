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
    /// # Description
    /// The adder of Gidney (Quantum 2, 74 (2018)) computes each carry with an AND and uncomputes
    /// it by measurement, so it costs n AND operations for n-bit registers. Std's
    /// `RippleCarryCGIncByLE` is the same adder, but a modular sum never exposes its carry-out.
    /// Here the carries are all computed first, `carryAction` acts on the last one, and the
    /// carries are then uncomputed while the sum is written. `carryAction` must leave the carry
    /// in the computational basis, as a phase does, since the uncomputation assumes its value.
    ///
    /// # Input
    /// ## carryAction
    /// The operation applied to the carry-out qubit.
    /// ## xs
    /// The little-endian addend, returned unchanged.
    /// ## ys
    /// The little-endian register incremented by `xs`, as long as `xs`.
    operation RippleCarryCGIncByLEWithCarryAction(
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
