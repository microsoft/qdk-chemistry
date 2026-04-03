// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.BinaryEncoding {

    import Std.Arrays.MostAndTail;
    import Std.Arrays.Partitioned;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyControlledOnBitString;
    import Std.Convert.IntAsDouble;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Measurement.MResetX;
    import Std.StatePreparation.PreparePureStateD;

    /// A single gate produced by the matrix compression pipeline.
    ///
    /// ``qubits`` always contains qubit indices only:
    ///   ("X",      [target],                    0,  [])
    ///   ("CX",     [control, target],           0,  [])
    ///   ("SWAP",   [a, b],                      0,  [])
    ///   ("CCX",    [ctrl1, ctrl2, target],      0,  [])
    ///   ("MCX",    [target, ctrl0, ctrl1, ...], ctrlStateBitmask, [])
    ///   ("SELECT", [addr0..addrN, data0..dataM], numAddrQubits,  data[][])
    struct MatrixCompressionOp {
        name : String,
        qubits : Int[],
        controlState : Int,
        lookupData : Bool[][],
    }

    /// Parameters for the binary-encoding state preparation.
    struct BinaryEncodingStatePreparationParams {
        /// Qubit indices for the dense state preparation (row map, reversed).
        rowMap : Int[],
        /// Amplitudes of the reduced-space statevector.
        stateVector : Double[],
        /// GF2+X expansion operations (CX / X) to reverse the GF2+X elimination.
        expansionOps : MatrixCompressionOp[],
        /// Binary-encoding gate sequence (already reversed by Python).
        binaryEncodingOps : MatrixCompressionOp[],
        /// Total number of qubits (system + ancilla).
        numQubits : Int,
        /// Number of ancilla qubits required by the binary-encoding circuit.
        numAncilla : Int,
    }

    /// Apply a single matrix-compression gate to a qubit register.
    operation ApplyMatrixCompressionOp(gate : MatrixCompressionOp, qs : Qubit[]) : Unit {
        if gate.name == "X" {
            X(qs[gate.qubits[0]]);
        } elif gate.name == "CX" {
            CX(qs[gate.qubits[0]], qs[gate.qubits[1]]);
        } elif gate.name == "SWAP" {
            SWAP(qs[gate.qubits[0]], qs[gate.qubits[1]]);
        } elif gate.name == "CCX" {
            CCNOT(qs[gate.qubits[0]], qs[gate.qubits[1]], qs[gate.qubits[2]]);
        } elif gate.name == "MCX" {
            let target = gate.qubits[0];
            let numControls = Length(gate.qubits) - 1;
            mutable controlQubits = [];
            mutable ctrlStateBools = [];
            for i in 0..numControls - 1 {
                set controlQubits += [qs[gate.qubits[1 + i]]];
                set ctrlStateBools += [((gate.controlState >>> i) &&& 1) == 1];
            }
            ApplyControlledOnBitString(ctrlStateBools, X, controlQubits, qs[target]);
        } elif gate.name == "SELECT" {
            let numAddr = gate.controlState;
            mutable addrQubits : Qubit[] = [];
            mutable targetQubits : Qubit[] = [];
            for i in 0..Length(gate.qubits) - 1 {
                if i < numAddr {
                    set addrQubits += [qs[gate.qubits[i]]];
                } else {
                    set targetQubits += [qs[gate.qubits[i]]];
                }
            }
            SparseOneHotSelect(gate.lookupData, addrQubits, targetQubits, false);
        } elif gate.name == "SELECT_AND" {
            let numAddr = gate.controlState;
            mutable addrQubits : Qubit[] = [];
            mutable targetQubits : Qubit[] = [];
            for i in 0..Length(gate.qubits) - 1 {
                if i < numAddr {
                    set addrQubits += [qs[gate.qubits[i]]];
                } else {
                    set targetQubits += [qs[gate.qubits[i]]];
                }
            }
            SparseOneHotSelect(gate.lookupData, addrQubits, targetQubits, true);
        } else {
            fail $"Unknown gate name: {gate.name}";
        }
    }

    /// Return true when every row of ``data`` is all-false.
    function IsDataAllZeros(data : Bool[][]) : Bool {
        for row in data {
            for bit in row {
                if bit { return false; }
            }
        }
        return true;
    }

    /// Apply X to each target qubit where the corresponding data bit is true.
    operation WriteOneHotData(data : Bool[], target : Qubit[]) : Unit {
        for i in 0..Length(data) - 1 {
            if data[i] { X(target[i]); }
        }
    }

    /// Controlled variant: apply CX(ctl, target[i]) for each true bit.
    operation ControlledWriteOneHotData(ctl : Qubit, data : Bool[], target : Qubit[]) : Unit {
        for i in 0..Length(data) - 1 {
            if data[i] { CX(ctl, target[i]); }
        }
    }


    /// AND gate with measurement-based adjoint uncomputation.
    operation MeasurementBasedAND(a : Qubit, b : Qubit, target : Qubit) : Unit is Adj {
        body (...) {
            CCNOT(a, b, target);
        }
        adjoint (...) {
            if MResetX(target) == One {
                CZ(a, b);
            }
        }
    }

    /// Sparse one-hot select
    ///
    /// For each row of the data, apply X to the target bits where the row is true, controlled on the address qubits being in the state corresponding to that row.
    operation SparseOneHotSelect(
        data : Bool[][],
        address : Qubit[],
        target : Qubit[],
        useMeasurementAND : Bool
    ) : Unit {
        let N = Length(data);

        if N == 0 or IsDataAllZeros(data) {
            // Nothing to apply
        } elif N == 1 {
            WriteOneHotData(data[0], target);
        } else {
            let n = Ceiling(Lg(IntAsDouble(N)));
            let (most, tail) = MostAndTail(address[...n - 1]);
            let parts = Partitioned([2^(n - 1)], data);
            let leftEmpty = IsDataAllZeros(parts[0]);
            let rightEmpty = IsDataAllZeros(parts[1]);

            if not leftEmpty and not rightEmpty {
                within { X(tail); } apply {
                    SparseOneHotSCS(tail, parts[0], most, target, useMeasurementAND);
                }
                SparseOneHotSCS(tail, parts[1], most, target, useMeasurementAND);
            } elif not rightEmpty {
                SparseOneHotSCS(tail, parts[1], most, target, useMeasurementAND);
            } elif not leftEmpty {
                within { X(tail); } apply {
                    SparseOneHotSCS(tail, parts[0], most, target, useMeasurementAND);
                }
            }
        }
    }

    /// Singly-controlled recursion for SparseOneHotSelect.
    ///
    /// When ``useMeasurementAND`` is true, uses MeasurementBasedAND + Adjoint
    operation SparseOneHotSCS(
        ctl : Qubit,
        data : Bool[][],
        address : Qubit[],
        target : Qubit[],
        useMeasurementAND : Bool
    ) : Unit {
        let N = Length(data);

        if N == 0 or IsDataAllZeros(data) {
            // Skip empty branch
        } elif N == 1 {
            ControlledWriteOneHotData(ctl, data[0], target);
        } else {
            let n = Ceiling(Lg(IntAsDouble(N)));
            let (most, tail) = MostAndTail(address[...n - 1]);
            let parts = Partitioned([2^(n - 1)], data);
            let leftEmpty = IsDataAllZeros(parts[0]);
            let rightEmpty = IsDataAllZeros(parts[1]);

            if not leftEmpty and not rightEmpty {
                use helper = Qubit();
                if useMeasurementAND {
                    within { X(tail); } apply {
                        MeasurementBasedAND(ctl, tail, helper);
                    }
                    SparseOneHotSCS(helper, parts[0], most, target, true);
                    CNOT(ctl, helper);
                    SparseOneHotSCS(helper, parts[1], most, target, true);
                    Adjoint MeasurementBasedAND(ctl, tail, helper);
                } else {
                    within { X(tail); } apply {
                        CCNOT(ctl, tail, helper);
                    }
                    SparseOneHotSCS(helper, parts[0], most, target, false);
                    CNOT(ctl, helper);
                    SparseOneHotSCS(helper, parts[1], most, target, false);
                    CCNOT(ctl, tail, helper);
                }
            } elif not rightEmpty {
                use helper = Qubit();
                if useMeasurementAND {
                    MeasurementBasedAND(ctl, tail, helper);
                    SparseOneHotSCS(helper, parts[1], most, target, true);
                    Adjoint MeasurementBasedAND(ctl, tail, helper);
                } else {
                    CCNOT(ctl, tail, helper);
                    SparseOneHotSCS(helper, parts[1], most, target, false);
                    CCNOT(ctl, tail, helper);
                }
            } elif not leftEmpty {
                use helper = Qubit();
                if useMeasurementAND {
                    X(tail);
                    MeasurementBasedAND(ctl, tail, helper);
                    SparseOneHotSCS(helper, parts[0], most, target, true);
                    Adjoint MeasurementBasedAND(ctl, tail, helper);
                    X(tail);
                } else {
                    X(tail);
                    CCNOT(ctl, tail, helper);
                    SparseOneHotSCS(helper, parts[0], most, target, false);
                    CCNOT(ctl, tail, helper);
                    X(tail);
                }
            }
        }
    }

    /// Prepare a quantum state using GF2+X elimination followed by binary-encoding circuit synthesis.
    ///
    /// The procedure is:
    ///   1. Prepare the dense statevector on the reduced qubit subset.
    ///   2. Apply the binary-encoding operations (already reversed by Python).
    ///   3. Apply the GF2+X expansion operations (CX / X).
    operation BinaryEncodingStatePreparation(
        params : BinaryEncodingStatePreparationParams,
        qs : Qubit[],
    ) : Unit {
        // Step 1: Dense state prep on reduced subspace
        PreparePureStateD(params.stateVector, Subarray(params.rowMap, qs));

        // Step 2: Apply binary-encoding ops (pre-reversed by Python)
        for gate in params.binaryEncodingOps {
            ApplyMatrixCompressionOp(gate, qs);
        }

        // Step 3: Expand back via GF2+X operations
        for gate in params.expansionOps {
            ApplyMatrixCompressionOp(gate, qs);
        }
    }

    /// Create a callable for the binary-encoding state preparation.
    function MakeBinaryEncodingStatePreparationOp(
        rowMap : Int[],
        stateVector : Double[],
        expansionOps : MatrixCompressionOp[],
        binaryEncodingOps : MatrixCompressionOp[],
        numQubits : Int,
        numAncilla : Int,
    ) : Qubit[] => Unit {
        BinaryEncodingStatePreparation(new BinaryEncodingStatePreparationParams {
            rowMap = rowMap,
            stateVector = stateVector,
            expansionOps = expansionOps,
            binaryEncodingOps = binaryEncodingOps,
            numQubits = numQubits,
            numAncilla = numAncilla,
        }, _)
    }

    /// Top-level circuit entry point for binary-encoding state preparation.
    operation MakeBinaryEncodingStatePreparationCircuit(
        rowMap : Int[],
        stateVector : Double[],
        expansionOps : MatrixCompressionOp[],
        binaryEncodingOps : MatrixCompressionOp[],
        numQubits : Int,
        numAncilla : Int,
    ) : Unit {
        use qs = Qubit[numQubits + numAncilla];
        BinaryEncodingStatePreparation(new BinaryEncodingStatePreparationParams {
            rowMap = rowMap,
            stateVector = stateVector,
            expansionOps = expansionOps,
            binaryEncodingOps = binaryEncodingOps,
            numQubits = numQubits,
            numAncilla = numAncilla,
        }, qs);
    }
}
