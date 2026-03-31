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

    /// Creates a circuit for sequentially applying two operations on the same target qubits.
    operation MakeSequentialCircuit(
        first : Qubit[] => Unit,
        second : Qubit[] => Unit,
        targets : Int[]
    ) : Unit {
        use qs = Qubit[Length(targets)];
        ApplySequential(first, second, Subarray(targets, qs));
    }
}
