// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.LoopTests {

    import Std.Canon.ApplyToEachCA;
    import Std.Measurement.MResetEachZ;
    import QDKChemistry.Utils.Loop.LoopA;
    import QDKChemistry.Utils.Loop.LoopCA;

    /// Applies `T` to `target` `index` times, so iteration `index` costs `index` T gates.
    internal operation TestApplyIndexedTCount(target : Qubit, index : Int) : Unit is Adj + Ctl {
        for _ in 1..index {
            T(target);
        }
    }

    /// Loops over `TestApplyIndexedTCount`, whose cost identifies which iterations were counted.
    operation TestLoopTCount(numIterations : Int, numWarmupIterations : Int, variant : Int) : Unit {
        use target = Qubit();
        if variant == 0 {
            LoopA(numIterations, numWarmupIterations, TestApplyIndexedTCount(target, _));
        } else {
            LoopCA(numIterations, numWarmupIterations, TestApplyIndexedTCount(target, _));
        }
    }

    /// Flips qubit `index` on every iteration so a simulation reveals which iterations ran.
    operation TestLoopVisitsEveryIteration(numIterations : Int, numWarmupIterations : Int, variant : Int) : Result[] {
        use qs = Qubit[numIterations];
        if variant == 0 {
            LoopA(numIterations, numWarmupIterations, index => X(qs[index]));
        } else {
            use control = Qubit();
            within {
                X(control);
            } apply {
                Controlled LoopCA([control], (numIterations, numWarmupIterations, index => X(qs[index])));
            }
        }
        MResetEachZ(qs)
    }

    /// Returns a caller-supplied-style operation that flips every qubit of its register.
    function TestMakeFlipAllOp() : (Qubit[] => Unit is Adj + Ctl) {
        qs => ApplyToEachCA(X, qs)
    }

    /// Applies `op` to a fresh register from a `LoopCA` iteration that captures it, as
    /// `ApplyRepeated` does.
    operation TestLoopAppliesCapturedOp(
        op : (Qubit[] => Unit is Adj + Ctl),
        numQubits : Int,
        numIterations : Int,
        numWarmupIterations : Int
    ) : Result[] {
        use qs = Qubit[numQubits];
        LoopCA(numIterations, numWarmupIterations, _ => op(qs));
        MResetEachZ(qs)
    }
}
