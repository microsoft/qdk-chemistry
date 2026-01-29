// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

import Std.Math.ArcCos;
import Std.Math.PI;
import Std.Convert.IntAsDouble;
import Std.Arrays.Subarray;
import Std.StatePreparation.PreparePureStateD;

/// Performs state preparation for a sparse quantum state |ψ⟩ given its
/// sparse representation and expansion operations.
/// # Parameters
/// - `rowMap`: An array mapping the indices of the non-zero, non-duplicate elements
///   in the sparse state vector.
/// - `stateVector`: The sparse representation of the initial quantum state |ψ⟩ as a vector of doubles.
/// - `expansionOps`: A list of operations (as arrays of qubit indices) to expand the initial
///   state preparation into the non-sparse |ψ⟩.
/// - `numQubits`: The number of qubits in the system register representing the quantum state |ψ⟩.
/// # Returns
/// - `Unit`: The operation prepares the desired quantum state on the allocated qubits.
operation StatePreparation(
    rowMap : Int[],
    stateVector : Double[],
    expansionOps : Int[][],
    numQubits : Int,
) : Unit {
    use qs = Qubit[numQubits];
    PreparePureStateD(stateVector, Subarray(rowMap, qs));
    for op in expansionOps {
        if Length(op) == 2 {
            CNOT(qs[op[0]], qs[op[1]]);
        } elif Length(op) == 1 {
            X(qs[op[0]]);
        } else {
            fail "Unsupported operation length in expansionOps.";
        }
    }
}

/// Prepares a single reference quantum state |ψ⟩ corresponding to a given bitstring.
/// # Parameters
/// - `bitstring`: An array of integers (0s and 1s) representing the desired quantum state.
///   For example, [0, 1, 0, 1].
/// - `numQubits`: The number of qubits in the system register.
/// # Returns
/// - `Unit`: The operation prepares the desired quantum state on the allocated qubits.
operation PrepareSingleReferenceState(
    bitstring : Int[],
    numQubits : Int,
) : Unit {
    use qs = Qubit[numQubits];
    for i in 0..numQubits - 1 {
        if bitstring[i] == 1 {
            X(qs[i]);
        }
    }
}
