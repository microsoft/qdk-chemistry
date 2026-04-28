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
    import QDKChemistry.Utils.StatePreparation.ApplyDensePreparation;

    /// A single gate produced by the matrix compression pipeline.
    ///
    /// ``qubits`` always contains qubit indices only:
    ///   ("X",      [target],                    0,  [])
    ///   ("CX",     [control, target],           0,  [])
    ///   ("SWAP",   [a, b],                      0,  [])
    ///   ("CCX",    [ctrl1, ctrl2, target],      0,  [])
    ///   ("MCX",    [ctrl0, ctrl1, ..., target], ctrlStateBitmask, [])
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
        gaussianEliminationOps : MatrixCompressionOp[],
        /// Binary-encoding gate sequence (already reversed by Python).
        binaryEncodingOps : MatrixCompressionOp[],
        /// Total number of qubits.
        numQubits : Int,
        /// Qubit indices (into the main register) that are idle during binary encoding
        /// and can be borrowed as ancillas by SparseOneHotSelect.
        ancillaPool : Int[],
    }

    /// Apply a single matrix-compression gate to a qubit register.
    ///
    /// ``ancillaPool`` is a list of pre-initialised |0⟩ qubits that
    /// SparseOneHotSelect may borrow as helpers (avoids allocating new qubits).
    /// Pass an empty array when no pool is available (e.g. for GF2+X ops).
    operation ApplyMatrixCompressionOp(gate : MatrixCompressionOp, qs : Qubit[], ancillaPool : Qubit[]) : Unit {
        if gate.name == "X" {
            X(qs[gate.qubits[0]]);
        } elif gate.name == "CX" {
            CX(qs[gate.qubits[0]], qs[gate.qubits[1]]);
        } elif gate.name == "SWAP" {
            SWAP(qs[gate.qubits[0]], qs[gate.qubits[1]]);
        } elif gate.name == "CCX" {
            CCNOT(qs[gate.qubits[0]], qs[gate.qubits[1]], qs[gate.qubits[2]]);
        } elif gate.name == "MCX" {
            let numControls = Length(gate.qubits) - 1;
            let target = gate.qubits[numControls];
            mutable controlQubits = [];
            mutable ctrlStateBools = [];
            for i in 0..numControls - 1 {
                set controlQubits += [qs[gate.qubits[i]]];
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
            SparseOneHotSelect(gate.lookupData, addrQubits, targetQubits, false, ancillaPool);
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
            SparseOneHotSelect(gate.lookupData, addrQubits, targetQubits, true, ancillaPool);
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

    /// Sparse one-hot select.
    ///
    /// For each row of ``data``, applies X to the target bits where the row is
    /// true, controlled on the address qubits matching that row's index.
    ///
    /// ``ancillaPool`` is a list of pre-initialised |0⟩ qubits that the
    /// recursive helper may borrow instead of allocating new ones.  Each
    /// borrowed qubit is restored to |0⟩ before the operation returns.
    /// Pass an empty array to fall back to ``use`` allocation.
    operation SparseOneHotSelect(
        data : Bool[][],
        address : Qubit[],
        target : Qubit[],
        useMeasurementAND : Bool,
        ancillaPool : Qubit[]
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
                    SparseOneHotSCS(tail, parts[0], most, target, useMeasurementAND, ancillaPool);
                }
                SparseOneHotSCS(tail, parts[1], most, target, useMeasurementAND, ancillaPool);
            } elif not rightEmpty {
                SparseOneHotSCS(tail, parts[1], most, target, useMeasurementAND, ancillaPool);
            } elif not leftEmpty {
                within { X(tail); } apply {
                    SparseOneHotSCS(tail, parts[0], most, target, useMeasurementAND, ancillaPool);
                }
            }
        }
    }

    /// Singly-controlled recursion for SparseOneHotSelect.
    ///
    /// Uses ``ancillaPool[0]`` as the helper qubit (must be |0⟩ on entry,
    /// restored on exit) and passes ``ancillaPool[1...]`` to recursive calls.
    /// Falls back to ``use helper = Qubit()`` when the pool is empty.
    operation SparseOneHotSCS(
        ctl : Qubit,
        data : Bool[][],
        address : Qubit[],
        target : Qubit[],
        useMeasurementAND : Bool,
        ancillaPool : Qubit[]
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
            let poolLen = Length(ancillaPool);

            if not leftEmpty and not rightEmpty {
                if poolLen > 0 {
                    let helper = ancillaPool[0];
                    let restPool = ancillaPool[1...];
                    if useMeasurementAND {
                        within { X(tail); } apply {
                            MeasurementBasedAND(ctl, tail, helper);
                        }
                        SparseOneHotSCS(helper, parts[0], most, target, true, restPool);
                        CNOT(ctl, helper);
                        SparseOneHotSCS(helper, parts[1], most, target, true, restPool);
                        Adjoint MeasurementBasedAND(ctl, tail, helper);
                    } else {
                        within { X(tail); } apply {
                            CCNOT(ctl, tail, helper);
                        }
                        SparseOneHotSCS(helper, parts[0], most, target, false, restPool);
                        CNOT(ctl, helper);
                        SparseOneHotSCS(helper, parts[1], most, target, false, restPool);
                        CCNOT(ctl, tail, helper);
                    }
                } else {
                    use helper = Qubit();
                    if useMeasurementAND {
                        within { X(tail); } apply {
                            MeasurementBasedAND(ctl, tail, helper);
                        }
                        SparseOneHotSCS(helper, parts[0], most, target, true, []);
                        CNOT(ctl, helper);
                        SparseOneHotSCS(helper, parts[1], most, target, true, []);
                        Adjoint MeasurementBasedAND(ctl, tail, helper);
                    } else {
                        within { X(tail); } apply {
                            CCNOT(ctl, tail, helper);
                        }
                        SparseOneHotSCS(helper, parts[0], most, target, false, []);
                        CNOT(ctl, helper);
                        SparseOneHotSCS(helper, parts[1], most, target, false, []);
                        CCNOT(ctl, tail, helper);
                    }
                }
            } elif not rightEmpty {
                if poolLen > 0 {
                    let helper = ancillaPool[0];
                    let restPool = ancillaPool[1...];
                    if useMeasurementAND {
                        MeasurementBasedAND(ctl, tail, helper);
                        SparseOneHotSCS(helper, parts[1], most, target, true, restPool);
                        Adjoint MeasurementBasedAND(ctl, tail, helper);
                    } else {
                        CCNOT(ctl, tail, helper);
                        SparseOneHotSCS(helper, parts[1], most, target, false, restPool);
                        CCNOT(ctl, tail, helper);
                    }
                } else {
                    use helper = Qubit();
                    if useMeasurementAND {
                        MeasurementBasedAND(ctl, tail, helper);
                        SparseOneHotSCS(helper, parts[1], most, target, true, []);
                        Adjoint MeasurementBasedAND(ctl, tail, helper);
                    } else {
                        CCNOT(ctl, tail, helper);
                        SparseOneHotSCS(helper, parts[1], most, target, false, []);
                        CCNOT(ctl, tail, helper);
                    }
                }
            } elif not leftEmpty {
                if poolLen > 0 {
                    let helper = ancillaPool[0];
                    let restPool = ancillaPool[1...];
                    if useMeasurementAND {
                        X(tail);
                        MeasurementBasedAND(ctl, tail, helper);
                        SparseOneHotSCS(helper, parts[0], most, target, true, restPool);
                        Adjoint MeasurementBasedAND(ctl, tail, helper);
                        X(tail);
                    } else {
                        X(tail);
                        CCNOT(ctl, tail, helper);
                        SparseOneHotSCS(helper, parts[0], most, target, false, restPool);
                        CCNOT(ctl, tail, helper);
                        X(tail);
                    }
                } else {
                    use helper = Qubit();
                    if useMeasurementAND {
                        X(tail);
                        MeasurementBasedAND(ctl, tail, helper);
                        SparseOneHotSCS(helper, parts[0], most, target, true, []);
                        Adjoint MeasurementBasedAND(ctl, tail, helper);
                        X(tail);
                    } else {
                        X(tail);
                        CCNOT(ctl, tail, helper);
                        SparseOneHotSCS(helper, parts[0], most, target, false, []);
                        CCNOT(ctl, tail, helper);
                        X(tail);
                    }
                }
            }
        }
    }

    /// Prepare a quantum state using GF2+X elimination followed by binary-encoding circuit synthesis.
    ///
    /// The procedure is:
    ///   1. Prepare the dense statevector on the reduced qubit subset.
    ///   2. Apply the binary-encoding operations (already reversed by Python).
    ///      SELECT gates borrow ancillas from ``params.ancillaPool`` (qubits that
    ///      are idle during binary encoding and start in |0⟩) to avoid allocating
    ///      extra qubits.
    ///   3. Apply the GF2+X expansion operations (CX / X).
    operation BinaryEncodingStatePreparation(
        params : BinaryEncodingStatePreparationParams,
        qs : Qubit[],
    ) : Unit {
        // Step 1: Dense state prep on reduced subspace
        ApplyDensePreparation(params.rowMap, params.stateVector, qs);
        // Step 2 & 3: Apply binary-encoding operations and GF2+X operations
        ApplyExpansion(params.binaryEncodingOps, params.gaussianEliminationOps, qs, params.ancillaPool);
    }

    /// Create a callable for the binary-encoding state preparation.
    function MakeBinaryEncodingStatePreparationOp(
        rowMap : Int[],
        stateVector : Double[],
        gaussianEliminationOps : MatrixCompressionOp[],
        binaryEncodingOps : MatrixCompressionOp[],
        numQubits : Int,
        ancillaPool : Int[],
    ) : Qubit[] => Unit {
        BinaryEncodingStatePreparation(new BinaryEncodingStatePreparationParams {
            rowMap = rowMap,
            stateVector = stateVector,
            gaussianEliminationOps = gaussianEliminationOps,
            binaryEncodingOps = binaryEncodingOps,
            numQubits = numQubits,
            ancillaPool = ancillaPool,
        }, _)
    }

    /// Top-level circuit entry point for binary-encoding state preparation.
    operation MakeBinaryEncodingStatePreparationCircuit(
        rowMap : Int[],
        stateVector : Double[],
        gaussianEliminationOps : MatrixCompressionOp[],
        binaryEncodingOps : MatrixCompressionOp[],
        numQubits : Int,
        ancillaPool : Int[],
    ) : Unit {
        use qs = Qubit[numQubits];
        BinaryEncodingStatePreparation(new BinaryEncodingStatePreparationParams {
            rowMap = rowMap,
            stateVector = stateVector,
            gaussianEliminationOps = gaussianEliminationOps,
            binaryEncodingOps = binaryEncodingOps,
            numQubits = numQubits,
            ancillaPool = ancillaPool,
        }, qs);
    }

    /// Applies the binary-encoding operations followed by GF2+X expansion operations.
    operation ApplyExpansion(
        binaryEncodingOps : MatrixCompressionOp[],
        gaussianEliminationOps : MatrixCompressionOp[],
        qs : Qubit[],
        ancillaPool : Int[],
    ) : Unit {
        let poolQubits = Subarray(ancillaPool, qs);
        for gate in binaryEncodingOps {
            ApplyMatrixCompressionOp(gate, qs, poolQubits);
        }
        for gate in gaussianEliminationOps {
            ApplyMatrixCompressionOp(gate, qs, []);
        }
    }

    /// Circuit entry point for the isometry stage of binary-encoding
    /// state preparation.
    /// Allocates qubits and delegates to ApplyExpansion.
    operation MakeBinaryEncodingExpansion(
        binaryEncodingOps : MatrixCompressionOp[],
        gaussianEliminationOps : MatrixCompressionOp[],
        numQubits : Int,
        ancillaPool : Int[],
    ) : Unit {
        use qs = Qubit[numQubits];
        ApplyExpansion(binaryEncodingOps, gaussianEliminationOps, qs, ancillaPool);
    }
}
