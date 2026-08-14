// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.BinaryEncoding {

    import Std.Arrays.MostAndTail;
    import Std.Arrays.Partitioned;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyPauliFromBitString;
    import Std.Convert.IntAsDouble;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Intrinsic.AND;
    import QDKChemistry.Utils.StatePreparation.StatePreparation;
    import QDKChemistry.Utils.StatePreparation.StatePreparationParams;

    /// A single gate produced by the matrix compression pipeline.
    ///
    /// ``kind`` names the gate: X, CX, SWAP, CCX, SELECT, or SELECT_AND.
    /// ``qubits`` always contains qubit indices only.
    struct MatrixCompressionOp {
        kind : String,
        qubits : Int[],
        controlState : Int,
        lookupData : Bool[][],
    }


    /// Apply a single matrix-compression gate that is a plain unitary.
    ///
    /// Covers every gate except SELECT/SELECT_AND, which may borrow ancilla and measure and
    /// therefore cannot be adjoint- or control-generated. Callers that only ever emit plain
    /// gates (such as the GF2+X expansion) can use this to stay adjointable.
    operation ApplyAdjointableCompressionOp(gate : MatrixCompressionOp, qs : Qubit[]) : Unit is Adj + Ctl {
        if gate.kind == "X" {
            X(qs[gate.qubits[0]]);
        } elif gate.kind == "CX" {
            CX(qs[gate.qubits[0]], qs[gate.qubits[1]]);
        } elif gate.kind == "SWAP" {
            SWAP(qs[gate.qubits[0]], qs[gate.qubits[1]]);
        } elif gate.kind == "CCX" {
            CCNOT(qs[gate.qubits[0]], qs[gate.qubits[1]], qs[gate.qubits[2]]);
        } else {
            fail $"Unsupported adjointable matrix-compression operation: {gate.kind}.";
        }
    }

    /// ``ancillaPool`` is a list of pre-initialised |0⟩ qubits that
    /// SparseOneHotSelect may borrow as helpers (avoids allocating new qubits).
    /// Pass an empty array when no pool is available (e.g. for GF2+X ops).
    operation ApplyMatrixCompressionOp(gate : MatrixCompressionOp, qs : Qubit[], ancillaPool : Qubit[]) : Unit is Adj {
        if gate.kind == "SELECT" or gate.kind == "SELECT_AND" {
            let numAddr = gate.controlState;
            let selectedQubits = Subarray(gate.qubits, qs);
            let addrQubits = selectedQubits[...numAddr - 1];
            let targetQubits = selectedQubits[numAddr...];
            SparseOneHotSelect(gate.lookupData, addrQubits, targetQubits, gate.kind == "SELECT_AND", ancillaPool);
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
    ) : Unit is Adj {
        let N = Length(data);

        if N == 0 or IsDataAllZeros(data) {
            // Nothing to apply
        } elif N == 1 {
            // Sole surviving row: write it straight onto the target bits.
            ApplyPauliFromBitString(PauliX, true, data[0], target);
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
    ) : Unit is Adj {
        let N = Length(data);

        if N == 0 or IsDataAllZeros(data) {
            // Skip empty branch
        } elif N == 1 {
            // Sole surviving row: write it onto the target bits, gated on this branch's control.
            Controlled ApplyPauliFromBitString([ctl], (PauliX, true, data[0], target));
        } else {
            let n = Ceiling(Lg(IntAsDouble(N)));
            let (most, tail) = MostAndTail(address[...n - 1]);
            let parts = Partitioned([2^(n - 1)], data);
            let leftEmpty = IsDataAllZeros(parts[0]);
            let rightEmpty = IsDataAllZeros(parts[1]);

            if leftEmpty and rightEmpty {
                // Both halves empty: nothing to recurse into.
            } elif Length(ancillaPool) > 0 {
                SparseOneHotSCSStep(
                    ctl,
                    tail,
                    ancillaPool[0],
                    parts,
                    most,
                    target,
                    leftEmpty,
                    rightEmpty,
                    useMeasurementAND,
                    ancillaPool[1...]
                );
            } else {
                use helper = Qubit();
                SparseOneHotSCSStep(ctl, tail, helper, parts, most, target, leftEmpty, rightEmpty, useMeasurementAND, []);
            }
        }
    }

    /// One recursion level of `SparseOneHotSCS`, given the branch ancilla to use.
    ///
    /// Split out so the pooled and freshly-allocated helper cases share a body; `use` scoping
    /// is the only reason the caller distinguishes them.
    operation SparseOneHotSCSStep(
        ctl : Qubit,
        tail : Qubit,
        helper : Qubit,
        parts : Bool[][][],
        most : Qubit[],
        target : Qubit[],
        leftEmpty : Bool,
        rightEmpty : Bool,
        useMeasurementAND : Bool,
        restPool : Qubit[]
    ) : Unit is Adj {
        // CCNOT is self-adjoint, so both strategies uncompute via `Adjoint andOp`.
        let andOp : (Qubit, Qubit, Qubit) => Unit is Adj = useMeasurementAND ? AND | CCNOT;

        if not leftEmpty and not rightEmpty {
            within { X(tail); } apply {
                andOp(ctl, tail, helper);
            }
            SparseOneHotSCS(helper, parts[0], most, target, useMeasurementAND, restPool);
            CNOT(ctl, helper);
            SparseOneHotSCS(helper, parts[1], most, target, useMeasurementAND, restPool);
            Adjoint andOp(ctl, tail, helper);
        } elif not rightEmpty {
            andOp(ctl, tail, helper);
            SparseOneHotSCS(helper, parts[1], most, target, useMeasurementAND, restPool);
            Adjoint andOp(ctl, tail, helper);
        } else {
            within { X(tail); } apply {
                andOp(ctl, tail, helper);
                SparseOneHotSCS(helper, parts[0], most, target, useMeasurementAND, restPool);
                Adjoint andOp(ctl, tail, helper);
            }
        }
    }

    /// Applies the binary-encoding operations followed by GF2+X expansion operations.
    operation ApplyExpansion(
        binaryEncodingOps : MatrixCompressionOp[],
        gaussianEliminationOps : MatrixCompressionOp[],
        qs : Qubit[],
        ancillaPool : Int[],
    ) : Unit is Adj {
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
    ) : Unit is Adj {
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
    ) : Qubit[] => Unit is Adj {
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
