// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.CircuitComposition {

    import Std.Arrays.Subarray;

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
