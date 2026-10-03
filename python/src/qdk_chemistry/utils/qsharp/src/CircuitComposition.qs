// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.CircuitComposition {

    import QDKChemistry.Utils.Loop.LoopCA;
    import Std.Arrays.Subarray;

    /// Returns the controlled version of `op`, taking the control register as its first argument.
    function MakeControlledOp<'T>(op : 'T => Unit is Adj + Ctl) : ((Qubit[], 'T) => Unit is Adj + Ctl) {
        Controlled op
    }

    /// Applies `op` to `target` `power` times.
    ///
    /// Under resource estimation the first `numWarmupIterations` applications are counted
    /// exactly and the next one is repeated for the rest; see `QDKChemistry.Utils.Loop.LoopCA`.
    operation ApplyRepeated<'T>(
        op : 'T => Unit is Adj + Ctl,
        power : Int,
        numWarmupIterations : Int,
        target : 'T
    ) : Unit is Adj + Ctl {
        LoopCA(power, numWarmupIterations, _ => op(target));
    }

    /// Returns an operation applying `op` `power` times.
    /// Parameters:
    /// - `numWarmupIterations`: Leading applications counted exactly under resource estimation
    function MakeRepeatedOp<'T>(
        op : 'T => Unit is Adj + Ctl,
        power : Int,
        numWarmupIterations : Int
    ) : ('T => Unit is Adj + Ctl) {
        ApplyRepeated(op, power, numWarmupIterations, _)
    }

    /// Adapts a control-register operation to the single-control-qubit shape phase estimation takes.
    function MakeSingleControlOp<'T>(op : (Qubit[], 'T) => Unit is Adj + Ctl) : ((Qubit, 'T) => Unit is Adj + Ctl) {
        (control, target) => op([control], target)
    }

    /// Applies two operations sequentially on the same system register.
    operation ApplySequential(
        first : Qubit[] => Unit,
        second : Qubit[] => Unit,
        systems : Qubit[]
    ) : Unit {
        first(systems);
        second(systems);
    }

    /// Returns a composed operation that applies ``first`` and then ``second``.
    function MakeSequentialOp(first : Qubit[] => Unit, second : Qubit[] => Unit) : Qubit[] => Unit {
        ApplySequential(first, second, _)
    }

    /// Returns `op` wrapped so it prepares and restores its own trailing `numShared` ancillas.
    function MakeSharedAncillaOp(
        op : Qubit[] => Unit is Adj + Ctl,
        prepareShared : Qubit[] => Unit is Adj + Ctl,
        numShared : Int
    ) : Qubit[] => Unit is Adj + Ctl {
        (qs) => {
            within {
                if numShared > 0 {
                    prepareShared(qs[Length(qs) - numShared...]);
                }
            } apply {
                op(qs);
            }
        }
    }

    /// Returns the maximum element of the given array of integers.
    function MaxInt(values : Int[]) : Int {
        // Caller is responsible for not passing an empty array.
        mutable max = values[0];
        for idx in 1 .. Length(values) - 1 {
            let value = values[idx];
            if (value > max) {
                set max = value;
            }
        }
        return max;
    }

    /// Creates a circuit for sequentially applying two operations on the same target qubits.
    operation MakeSequentialCircuit(
        first : Qubit[] => Unit,
        second : Qubit[] => Unit,
        targets : Int[]
    ) : Unit {
        if (Length(targets) == 0) {
            // No target indices: do nothing.
            return ();
        } else {
            // Allocate enough qubits so that all indices in 'targets' are valid.
            let maxTarget = MaxInt(targets);
            use qs = Qubit[1 + maxTarget];
            ApplySequential(first, second, Subarray(targets, qs));
        }
    }
}
