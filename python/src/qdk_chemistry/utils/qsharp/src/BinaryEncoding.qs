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
    import QDKChemistry.Utils.StatePreparation.StatePreparation;
    import QDKChemistry.Utils.StatePreparation.StatePreparationParams;

    /// A single gate produced by the matrix compression pipeline.
    ///
    /// ``kind`` identifies X, CX, SWAP, CCX, MCX, SELECT, and SELECT_AND as 0 through 6.
    /// ``qubits`` always contains qubit indices only.
    struct MatrixCompressionOp {
        kind : Int,
        qubits : Int[],
        controlState : Int,
        lookupData : Bool[][],
    }

    /// Apply a single matrix-compression gate to a qubit register.
    ///
    /// Little-endian control-state bits for a multi-controlled gate.
    function ControlStateBits(controlState : Int, numControls : Int) : Bool[] {
        mutable bits = [];
        for i in 0..numControls - 1 {
            set bits += [((controlState >>> i) &&& 1) == 1];
        }
        bits
    }

    /// Apply a single matrix-compression gate that is a plain unitary.
    ///
    /// Covers every gate except SELECT/SELECT_AND, which may borrow ancilla and measure and
    /// therefore cannot be adjoint- or control-generated. Callers that only ever emit plain
    /// gates (such as the GF2+X expansion) can use this to stay adjointable.
    operation ApplyAdjointableCompressionOp(gate : MatrixCompressionOp, qs : Qubit[]) : Unit is Adj + Ctl {
        if gate.kind == 0 {
            X(qs[gate.qubits[0]]);
        } elif gate.kind == 1 {
            CX(qs[gate.qubits[0]], qs[gate.qubits[1]]);
        } elif gate.kind == 2 {
            SWAP(qs[gate.qubits[0]], qs[gate.qubits[1]]);
        } elif gate.kind == 3 {
            CCNOT(qs[gate.qubits[0]], qs[gate.qubits[1]], qs[gate.qubits[2]]);
        } elif gate.kind == 4 {
            let numControls = Length(gate.qubits) - 1;
            let controlQubits = Subarray(gate.qubits[0..numControls - 1], qs);
            let ctrlStateBools = ControlStateBits(gate.controlState, numControls);
            ApplyControlledOnBitString(ctrlStateBools, X, controlQubits, qs[gate.qubits[numControls]]);
        } else {
            fail "Unsupported adjointable matrix-compression operation.";
        }
    }

    /// ``ancillaPool`` is a list of pre-initialised |0⟩ qubits that
    /// SparseOneHotSelect may borrow as helpers (avoids allocating new qubits).
    /// Pass an empty array when no pool is available (e.g. for GF2+X ops).
    operation ApplyMatrixCompressionOp(gate : MatrixCompressionOp, qs : Qubit[], ancillaPool : Qubit[]) : Unit {
        if gate.kind == 5 {
            let numAddr = gate.controlState;
            let selectedQubits = Subarray(gate.qubits, qs);
            let addrQubits = selectedQubits[...numAddr - 1];
            let targetQubits = selectedQubits[numAddr...];
            SparseOneHotSelect(gate.lookupData, addrQubits, targetQubits, false, ancillaPool);
        } elif gate.kind == 6 {
            let numAddr = gate.controlState;
            let selectedQubits = Subarray(gate.qubits, qs);
            let addrQubits = selectedQubits[...numAddr - 1];
            let targetQubits = selectedQubits[numAddr...];
            SparseOneHotSelect(gate.lookupData, addrQubits, targetQubits, true, ancillaPool);
        } else {
            ApplyAdjointableCompressionOp(gate, qs);
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

    /// Composes the dense preparation with binary-encoding expansion.
    /// The dense preparation is applied to the subregister specified by embeddingMap,
    /// then binary-encoding and GF2+X expansion operations are applied.
    ///
    /// The dense preparation is taken as *parameters* rather than as a callable: a callable
    /// that captures another callable cannot be resolved statically by the adaptive-profile
    /// code generator, which makes the composition unusable as a QPE `statePrep` argument.
    operation ComposeBinaryEncoding(
        denseParams : StatePreparationParams,
        embeddingMap : Int[],
        binaryEncodingOps : MatrixCompressionOp[],
        gaussianEliminationOps : MatrixCompressionOp[],
        ancillaPool : Int[],
        qs : Qubit[],
    ) : Unit {
        StatePreparation(denseParams, Subarray(embeddingMap, qs));
        ApplyExpansion(binaryEncodingOps, gaussianEliminationOps, qs, ancillaPool);
    }

    /// Returns a callable that applies binary-encoding composition.
    function MakeComposeBinaryEncodingOp(
        denseParams : StatePreparationParams,
        embeddingMap : Int[],
        binaryEncodingOps : MatrixCompressionOp[],
        gaussianEliminationOps : MatrixCompressionOp[],
        ancillaPool : Int[],
    ) : Qubit[] => Unit {
        ComposeBinaryEncoding(denseParams, embeddingMap, binaryEncodingOps, gaussianEliminationOps, ancillaPool, _)
    }

    /// Circuit entry point for binary-encoding composition.
    operation MakeComposeBinaryEncodingCircuit(
        denseParams : StatePreparationParams,
        embeddingMap : Int[],
        binaryEncodingOps : MatrixCompressionOp[],
        gaussianEliminationOps : MatrixCompressionOp[],
        numQubits : Int,
        ancillaPool : Int[],
    ) : Unit {
        use qs = Qubit[numQubits];
        ComposeBinaryEncoding(denseParams, embeddingMap, binaryEncodingOps, gaussianEliminationOps, ancillaPool, qs);
    }
}
