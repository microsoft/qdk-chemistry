// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.StatePreparation {

    import Std.Arrays.Subarray;
    import Std.Canon.ApplyControlledOnBitString;
    import Std.Measurement.MResetZ;
    import Std.StatePreparation.PreparePureStateD;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.BinaryEncoding.MatrixCompressionOp;
    import QDKChemistry.Utils.BinaryEncoding.ApplyMatrixCompressionOp;


    /// A struct to hold parameters for state preparation.
    /// - `rowMap`: An array of integers representing the mapping of qubits to rows in the state vector.
    /// - `stateVector`: An array of doubles representing the amplitudes of the quantum state.
    /// - `expansionOps`: An array of MatrixCompressionOp representing the operations to expand the state preparation (e.g., CX, X gates).
    /// - `numQubits`: The number of qubits to allocate for the state preparation.
    struct StatePreparationParams {
        rowMap : Int[],
        stateVector : Double[],
        expansionOps : MatrixCompressionOp[],
        numQubits : Int,
    }


    /// Prepares a quantum state based on the provided parameters.
    /// # Parameters
    /// - `params`: A `StatePreparationParams` struct containing the parameters for state preparation.
    /// - `qs`: An array of qubits on which to prepare the state.
    /// # Returns
    /// - `Unit`: The operation prepares the quantum state on the allocated qubits.
    operation StatePreparation(
        params : StatePreparationParams,
        qs : Qubit[],
    ) : Unit {
        ApplyDensePreparation(params.rowMap, params.stateVector, qs);
        ApplyExpansion(params.expansionOps, qs);
    }

    /// A helper function to create a callable for state preparation.
    /// # Parameters
    /// - `params`: A `StatePreparationParams` struct containing the parameters for state preparation.
    /// # Returns
    /// - `Qubit[] => Unit`: A callable that takes an array of qubits and prepares the quantum state on those qubits.
    function MakeStatePreparationOp(params : StatePreparationParams) : Qubit[] => Unit {
        StatePreparation(params, _)
    }


    /// A helper operation to create a circuit for state preparation.
    /// # Parameters
    /// - `rowMap`: An array of integers representing the mapping of qubits to rows in the state vector.
    /// - `stateVector`: An array of doubles representing the amplitudes of the quantum state.
    /// - `expansionOps`: An array of MatrixCompressionOp representing the operations to expand the state preparation.
    /// - `numQubits`: The number of qubits to allocate for the state preparation.
    /// # Returns
    /// - `Unit`: The operation prepares the quantum state on the allocated qubits.
    operation MakeStatePreparationCircuit(
        rowMap : Int[],
        stateVector : Double[],
        expansionOps : MatrixCompressionOp[],
        numQubits : Int,
    ) : Unit {
        use qs = Qubit[numQubits];
        StatePreparation(new StatePreparationParams {
            rowMap = rowMap,
            stateVector = stateVector,
            expansionOps = expansionOps,
            numQubits = numQubits
        }, qs);
    }

    /// Prepares a single reference quantum state |ψ⟩ corresponding to a given bitstring.
    /// # Parameters
    /// - `bitStrings`: An array of integers (0s and 1s) representing the desired quantum state.
    /// - `numQubits`: The number of qubits to allocate for the state preparation.
    ///   For example, [0, 1, 0, 1].
    struct SingleReferenceParams {
        bitStrings : Int[],
        numQubits : Int,
    }


    /// Prepares a single reference quantum state |ψ⟩ corresponding to a given bitstring.
    /// # Parameters
    /// - `params`: A `SingleReferenceParams` struct containing the parameters for state preparation.
    /// - `qs`: An array of qubits on which to prepare the state.
    /// # Returns
    /// - `Unit`: The operation prepares the quantum state on the allocated qubits.
    operation PrepareSingleReferenceState(
        params : SingleReferenceParams,
        qs : Qubit[],
    ) : Unit {
        let bitLen = Length(params.bitStrings);
        if bitLen != Length(qs) {
            fail "Length of bitStrings must match the number of qubits.";
        }
        for i in 0..bitLen - 1 {
            if params.bitStrings[i] == 1 {
                X(qs[i]);
            }
        }
    }

    /// A helper operation to create a circuit for preparing a single reference quantum state.
    /// # Parameters
    /// - `bitStrings`: An array of integers (0s and 1s) representing the desired quantum state.
    /// - `numQubits`: The number of qubits to allocate for the state preparation.
    /// # Returns
    /// - `Unit`: The operation prepares the quantum state on the allocated qubits.
    operation MakeSingleReferenceStateCircuit(
        bitStrings : Int[],
        numQubits : Int
    ) : Unit {
        use qs = Qubit[numQubits];
        PrepareSingleReferenceState(new SingleReferenceParams {
            bitStrings = bitStrings,
            numQubits = numQubits
        }, qs);
    }

    /// A helper function to create a callable for preparing a single reference quantum state.
    /// # Parameters
    /// - `params`: A `SingleReferenceParams` struct containing the parameters for state preparation.
    /// # Returns
    /// - `Qubit[] => Unit`: A callable that takes an array of qubits and prepares the single reference quantum state on those qubits.
    function MakePrepareSingleReferenceStateOp(params : SingleReferenceParams) : Qubit[] => Unit {
        PrepareSingleReferenceState(params, _)
    }

    /// Prepares the dense statevector on the qubit subset given by rowMap.
    operation ApplyDensePreparation(
        rowMap : Int[],
        stateVector : Double[],
        qs : Qubit[],
    ) : Unit {
        PreparePureStateD(stateVector, Subarray(rowMap, qs));
    }

    /// Circuit entry point for the dense state preparation stage.
    operation MakeDenseStatePreparation(
        rowMap : Int[],
        stateVector : Double[],
        numQubits : Int,
    ) : Unit {
        use qs = Qubit[numQubits];
        ApplyDensePreparation(rowMap, stateVector, qs);
    }

    /// Applies the GF2+X expansion operations (CX / X gates) to the full register.
    operation ApplyExpansion(
        expansionOps : MatrixCompressionOp[],
        qs : Qubit[],
    ) : Unit {
        for gate in expansionOps {
            ApplyMatrixCompressionOp(gate, qs, []);
        }
    }

    /// Circuit entry point for the expansion (isometry) stage.
    operation MakeExpansion(
        expansionOps : MatrixCompressionOp[],
        numQubits : Int,
    ) : Unit {
        use qs = Qubit[numQubits];
        ApplyExpansion(expansionOps, qs);
    }
}
