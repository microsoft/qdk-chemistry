// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.Loop {

    import Std.Diagnostics.Fact;
    import Std.Math.MinI;
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.ResourceEstimation.RepeatEstimates;

    /// # Summary
    /// Applies `iteration(i)` for each `i` in `0..numIterations - 1`, repeating it symbolically
    /// under resource estimation.
    ///
    /// # Description
    /// During simulation and execution every iteration runs explicitly. Under resource
    /// estimation, the first `numWarmupIterations` iterations are counted exactly, and iteration
    /// `numWarmupIterations` is counted `numIterations - numWarmupIterations` times through
    /// `RepeatEstimates`. Warm-up iterations let stateful estimates, such as least-recently-used
    /// memory/compute qubit placement, reach their steady state before the repeated iteration is
    /// sampled, so the one-time cost of the first iterations is not multiplied by the loop count.
    ///
    /// Adapted from `Loop` in the QDK arithmetic library, with the warm-up count added. The
    /// auto-generated adjoint reverses the statement order, so under `Adjoint` the repeated
    /// iteration is counted before the warm-up iterations.
    ///
    /// # Input
    /// ## numIterations
    /// Number of loop iterations. Must be non-negative.
    /// ## numWarmupIterations
    /// Number of leading iterations counted exactly under resource estimation. Must be
    /// non-negative; values of at least `numIterations` count every iteration exactly.
    /// ## iteration
    /// Operation that implements one loop iteration and receives the iteration index.
    operation Loop(numIterations : Int, numWarmupIterations : Int, iteration : (Int => Unit)) : Unit {
        Fact(numIterations >= 0, "numIterations must be non-negative");
        Fact(numWarmupIterations >= 0, "numWarmupIterations must be non-negative");
        if IsResourceEstimating() {
            let numExact = MinI(numIterations, numWarmupIterations);
            for i in 0..numExact - 1 {
                iteration(i);
            }
            if numIterations > numExact {
                within {
                    RepeatEstimates(numIterations - numExact);
                } apply {
                    iteration(numExact);
                }
            }
        } else {
            for i in 0..numIterations - 1 {
                iteration(i);
            }
        }
    }

    /// # Summary
    /// Adjointable variant of `Loop` for adjointable iteration operations.
    operation LoopA(numIterations : Int, numWarmupIterations : Int, iteration : (Int => Unit is Adj)) : Unit is Adj {
        Fact(numIterations >= 0, "numIterations must be non-negative");
        Fact(numWarmupIterations >= 0, "numWarmupIterations must be non-negative");
        if IsResourceEstimating() {
            let numExact = MinI(numIterations, numWarmupIterations);
            for i in 0..numExact - 1 {
                iteration(i);
            }
            if numIterations > numExact {
                within {
                    RepeatEstimates(numIterations - numExact);
                } apply {
                    iteration(numExact);
                }
            }
        } else {
            for i in 0..numIterations - 1 {
                iteration(i);
            }
        }
    }

    /// # Summary
    /// Adjointable and controllable variant of `Loop` for adjointable and controllable
    /// iteration operations.
    operation LoopCA(
        numIterations : Int,
        numWarmupIterations : Int,
        iteration : (Int => Unit is Adj + Ctl)
    ) : Unit is Adj + Ctl {
        Fact(numIterations >= 0, "numIterations must be non-negative");
        Fact(numWarmupIterations >= 0, "numWarmupIterations must be non-negative");
        if IsResourceEstimating() {
            let numExact = MinI(numIterations, numWarmupIterations);
            for i in 0..numExact - 1 {
                iteration(i);
            }
            if numIterations > numExact {
                within {
                    RepeatEstimates(numIterations - numExact);
                } apply {
                    iteration(numExact);
                }
            }
        } else {
            for i in 0..numIterations - 1 {
                iteration(i);
            }
        }
    }
}
