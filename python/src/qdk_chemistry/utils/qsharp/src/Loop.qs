// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Warm-up-aware repetition adapted from the QDK arithmetic library.
namespace QDKChemistry.Utils.Loop {

    import Std.Diagnostics.Fact;
    import Std.Math.MinI;
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.ResourceEstimation.RepeatEstimates;

    /// Counts warm-up iterations exactly before repeating the next iteration's cost under estimation;
    /// simulation runs every iteration, and the adjoint reverses the warm-up and repeated-body order.
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
