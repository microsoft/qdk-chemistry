"""QIR to Qiskit QuantumCircuit Converter.

This module provides a converter from QIR (Quantum Intermediate Representation)
to Qiskit's QuantumCircuit using PyQIR's visitor/passes pattern.

The converter will raise an error if the QIR contains operations that
Qiskit cannot understand.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pyqir
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


class UnsupportedQIROperationError(Exception):
    """Raised when a QIR operation cannot be converted to Qiskit."""


class QirToQiskitConverter(pyqir.QirModuleVisitor):
    """Converts a QIR module to a Qiskit QuantumCircuit.

    This converter uses PyQIR's visitor pattern to traverse QIR and build
    an equivalent Qiskit circuit. It will raise UnsupportedQIROperationError
    if it encounters any operations that cannot be represented in Qiskit.
    """

    def __init__(self) -> None:
        super().__init__()
        self._circuit: QuantumCircuit = QuantumCircuit()

    def convert(self, qir: pyqir.Module) -> QuantumCircuit:
        """Convert a QIR module to a Qiskit QuantumCircuit.

        Args:
            qir: The QIR module to convert.

        Returns:
            A Qiskit QuantumCircuit representing the same quantum operations as the QIR module.

        Raises:
            UnsupportedQIROperationError: If the entry point has no basic blocks to walk.

        """
        # Get qubit/result counts from entry point function attributes
        entry_point = next(filter(pyqir.is_entry_point, qir.functions))
        num_qubits = pyqir.required_num_qubits(entry_point)
        num_results = pyqir.required_num_results(entry_point)

        # Create the circuit
        if num_qubits > 0 and num_results > 0:
            self._circuit = QuantumCircuit(
                QuantumRegister(num_qubits, "q"),
                ClassicalRegister(num_results, "c"),
            )
        elif num_qubits > 0:
            self._circuit = QuantumCircuit(QuantumRegister(num_qubits, "q"))
        else:
            self._circuit = QuantumCircuit()

        # Build the circuit
        basic_blocks = entry_point.basic_blocks
        if not basic_blocks:
            raise UnsupportedQIROperationError("QIR entry point has no function body.")
        self._walk_blocks(basic_blocks[0])
        return self._circuit

    # =========================================================================
    # Control flow
    # =========================================================================

    def _walk_blocks(self, block: "pyqir.BasicBlock") -> None:
        """Walk the control-flow graph from *block*, emitting conditional blocks as Qiskit if-blocks.

        The default pyqir visitor iterates basic blocks flatly and ignores branch
        terminators, which would silently emit conditional bodies unconditionally.

        Args:
            block: The entry basic block to start walking from.

        Raises:
            UnsupportedQIROperationError: If the control flow is not a forward-only chain of
                single-sided conditionals, or if a block is revisited (loop).

        """
        visited: set[str] = set()
        current: pyqir.BasicBlock | None = block
        while current is not None:
            if current.name in visited:
                raise UnsupportedQIROperationError("Loops in QIR are not supported in Qiskit QuantumCircuit.")
            visited.add(current.name)
            self._on_block(current)
            current = self._next_block(current)

    def _next_block(self, block: "pyqir.BasicBlock") -> "pyqir.BasicBlock | None":
        """Emit any conditional body reachable from *block* and return the block to continue with.

        Args:
            block: The basic block whose terminator is being resolved.

        Returns:
            The next basic block to visit, or ``None`` when the terminator returns.

        Raises:
            UnsupportedQIROperationError: If the branch is not a single-sided conditional.

        """
        terminator = block.terminator
        successors = terminator.successors
        if not successors:
            return None
        if len(successors) == 1:
            return terminator.operands[0]
        if len(successors) != 2:
            raise UnsupportedQIROperationError(f"Unsupported branch with {len(successors)} successors.")

        # LLVM stores conditional branch operands as [condition, false_target, true_target].
        condition, false_block, true_block = terminator.operands[0], terminator.operands[1], terminator.operands[2]
        clbit = self._condition_clbit(condition)
        if self._branches_to(true_block, false_block):
            self._emit_conditional_block(true_block, None, clbit)
            return false_block
        if self._branches_to(false_block, true_block):
            self._emit_conditional_block(None, false_block, clbit)
            return true_block
        join = self._common_successor(true_block, false_block)
        if join is None:
            raise UnsupportedQIROperationError(
                f"Unsupported control flow: branches to '{true_block.name}' and '{false_block.name}' do not reconverge."
            )
        self._emit_conditional_block(true_block, false_block, clbit)
        return join

    @staticmethod
    def _branches_to(block: "pyqir.BasicBlock", target: "pyqir.BasicBlock") -> bool:
        """Check whether *block* ends in an unconditional branch to *target*.

        Args:
            block: The candidate conditional body block.
            target: The expected join block.

        Returns:
            True if *block* falls through to *target*.

        """
        successors = block.terminator.successors
        return len(successors) == 1 and successors[0].name == target.name

    @staticmethod
    def _common_successor(a: "pyqir.BasicBlock", b: "pyqir.BasicBlock") -> "pyqir.BasicBlock | None":
        """Return the block both *a* and *b* unconditionally branch to, if any.

        Args:
            a: The true-branch block.
            b: The false-branch block.

        Returns:
            The shared join block, or ``None`` if the branches do not reconverge immediately.

        """
        a_successors, b_successors = a.terminator.successors, b.terminator.successors
        if len(a_successors) == 1 and len(b_successors) == 1 and a_successors[0].name == b_successors[0].name:
            return a_successors[0]
        return None

    def _condition_clbit(self, condition: pyqir.Value) -> int:
        """Resolve the classical bit index a branch condition reads.

        Args:
            condition: The QIR value used as the branch condition.

        Returns:
            The index of the classical bit holding the measurement result.

        Raises:
            UnsupportedQIROperationError: If the condition is not a measurement result read.

        """
        callee = getattr(getattr(condition, "callee", None), "name", "")
        if "read_result" not in callee:
            raise UnsupportedQIROperationError(
                f"Branch conditions must read a measurement result directly, got '{callee or condition}'."
            )
        return self._clbit(condition.args[0])

    def _emit_conditional_block(
        self,
        true_block: "pyqir.BasicBlock | None",
        false_block: "pyqir.BasicBlock | None",
        clbit: int,
    ) -> None:
        """Emit conditional branch bodies as a Qiskit if/else block guarded on a classical bit.

        Args:
            true_block: Block executed when the classical bit is 1, or ``None`` when empty.
            false_block: Block executed when the classical bit is 0, or ``None`` when empty.
            clbit: Index of the classical bit to test.

        """
        outer = self._circuit
        condition = (outer.clbits[clbit], 1)
        true_body = self._body_circuit(true_block)
        if false_block is None:
            outer.if_test(condition, true_body, outer.qubits, outer.clbits)
            return
        outer.if_else(condition, true_body, self._body_circuit(false_block), outer.qubits, outer.clbits)

    def _body_circuit(self, block: "pyqir.BasicBlock | None") -> QuantumCircuit:
        """Build a Qiskit circuit holding the operations of a conditional branch body.

        Args:
            block: The basic block to convert, or ``None`` for an empty body.

        Returns:
            A circuit over the same registers as the enclosing circuit.

        """
        outer = self._circuit
        body = QuantumCircuit(*outer.qregs, *outer.cregs)
        if block is None:
            return body
        self._circuit = body
        try:
            self._on_block(block)
        finally:
            self._circuit = outer
        return body

    def _qubit(self, q: pyqir.Value) -> int:
        """Get circuit qubit index for a QIR qubit value.

        Args:
            q: The QIR value representing a qubit.

        Returns:
            The index of the qubit in the Qiskit circuit.

        """
        # Use `qubit_id` on older pyqir instances
        if hasattr(pyqir, "qubit_id"):
            return pyqir.qubit_id(q)
        # Newer pyqir versions use the more generic `ptr_id` for both qubits and results
        return pyqir.ptr_id(q)

    def _clbit(self, r: pyqir.Value) -> int:
        """Get circuit classical bit index for a QIR result value.

        Args:
            r: The QIR value representing a result.

        Returns:
            The index of the classical bit in the Qiskit circuit.

        """
        # Use `result_id` on older pyqir instances
        if hasattr(pyqir, "result_id"):
            return pyqir.result_id(r)
        # Newer pyqir versions use the more generic `ptr_id` for both qubits and results
        return pyqir.ptr_id(r)

    def _angle(self, a: pyqir.Value) -> float:
        """Extract angle value from a QIR constant.

        Args:
            a: The QIR value representing an angle (e.g., for rotation gates).

        Returns:
            The angle as a float.

        """
        if hasattr(a, "value"):
            return float(a.value)
        ir_str = str(a)
        if "double" in ir_str:
            parts = ir_str.split()
            for i, part in enumerate(parts):
                if part == "double" and i + 1 < len(parts):
                    return float(parts[i + 1])
        raise UnsupportedQIROperationError(f"Cannot extract angle from: {a}")

    # =========================================================================
    # Single-qubit gates
    # =========================================================================

    def _on_qis_h(self, call: pyqir.Call, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a Hadamard gate to the target qubit.

        Args:
            call: The QIR call instruction for the H gate.
            target: The QIR value representing the target qubit.

        """
        self._circuit.h(self._qubit(target))

    def _on_qis_x(self, call: pyqir.Call, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a Pauli-X (NOT) gate to the target qubit.

        Args:
            call: The QIR call instruction for the X gate.
            target: The QIR value representing the target qubit.

        """
        self._circuit.x(self._qubit(target))

    def _on_qis_y(self, call: pyqir.Call, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a Pauli-Y gate to the target qubit.

        Args:
            call: The QIR call instruction for the Y gate.
            target: The QIR value representing the target qubit.

        """
        self._circuit.y(self._qubit(target))

    def _on_qis_z(self, call: pyqir.Call, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a Pauli-Z gate to the target qubit.

        Args:
            call: The QIR call instruction for the Z gate.
            target: The QIR value representing the target qubit.

        """
        self._circuit.z(self._qubit(target))

    def _on_qis_s(self, call: pyqir.Call, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply an S gate to the target qubit.

        Args:
            call: The QIR call instruction for the S gate.
            target: The QIR value representing the target qubit.

        """
        self._circuit.s(self._qubit(target))

    def _on_qis_s_adj(self, call: pyqir.Call, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply an S† (adjoint S) gate to the target qubit.

        Args:
            call: The QIR call instruction for the S† gate.
            target: The QIR value representing the target qubit.

        """
        self._circuit.sdg(self._qubit(target))

    def _on_qis_t(self, call: pyqir.Call, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a T gate to the target qubit.

        Args:
            call: The QIR call instruction for the T gate.
            target: The QIR value representing the target qubit.

        """
        self._circuit.t(self._qubit(target))

    def _on_qis_t_adj(self, call: pyqir.Call, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a T† (adjoint T) gate to the target qubit.

        Args:
            call: The QIR call instruction for the T† gate.
            target: The QIR value representing the target qubit.

        """
        self._circuit.tdg(self._qubit(target))

    # =========================================================================
    # Rotation gates
    # =========================================================================

    def _on_qis_rx(self, call: pyqir.Call, angle: pyqir.Value, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a RX gate to the target qubit.

        Args:
            call: The QIR call instruction for the RX gate.
            angle: The QIR value representing the rotation angle.
            target: The QIR value representing the target qubit.

        """
        self._circuit.rx(self._angle(angle), self._qubit(target))

    def _on_qis_ry(self, call: pyqir.Call, angle: pyqir.Value, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a RY gate to the target qubit.

        Args:
            call: The QIR call instruction for the RY gate.
            angle: The QIR value representing the rotation angle.
            target: The QIR value representing the target qubit.

        """
        self._circuit.ry(self._angle(angle), self._qubit(target))

    def _on_qis_rz(self, call: pyqir.Call, angle: pyqir.Value, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a RZ gate to the target qubit.

        Args:
            call: The QIR call instruction for the RZ gate.
            angle: The QIR value representing the rotation angle.
            target: The QIR value representing the target qubit.

        """
        self._circuit.rz(self._angle(angle), self._qubit(target))

    # =========================================================================
    # Two-qubit gates
    # =========================================================================

    def _on_qis_cx(self, call: pyqir.Call, ctrl: pyqir.Value, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a CX (CNOT) gate to the target qubit.

        Args:
            call: The QIR call instruction for the CX gate.
            ctrl: The QIR value representing the control qubit.
            target: The QIR value representing the target qubit.

        """
        self._circuit.cx(self._qubit(ctrl), self._qubit(target))

    def _on_qis_cy(self, call: pyqir.Call, ctrl: pyqir.Value, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a CY gate to the target qubit.

        Args:
            call: The QIR call instruction for the CY gate.
            ctrl: The QIR value representing the control qubit.
            target: The QIR value representing the target qubit.

        """
        self._circuit.cy(self._qubit(ctrl), self._qubit(target))

    def _on_qis_cz(self, call: pyqir.Call, ctrl: pyqir.Value, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a CZ gate to the target qubit.

        Args:
            call: The QIR call instruction for the CZ gate.
            ctrl: The QIR value representing the control qubit.
            target: The QIR value representing the target qubit.

        """
        self._circuit.cz(self._qubit(ctrl), self._qubit(target))

    def _on_qis_swap(self, call: pyqir.Call, t1: pyqir.Value, t2: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a SWAP gate to the target qubits.

        Args:
            call: The QIR call instruction for the SWAP gate.
            t1: The QIR value representing the first target qubit.
            t2: The QIR value representing the second target qubit.

        """
        self._circuit.swap(self._qubit(t1), self._qubit(t2))

    def _on_qis_rxx(self, call: pyqir.Call, angle: pyqir.Value, t1: pyqir.Value, t2: pyqir.Value) -> None:  # noqa: ARG002
        """Apply an RXX gate to the target qubits.

        Args:
            call: The QIR call instruction for the RXX gate.
            angle: The QIR value representing the rotation angle.
            t1: The QIR value representing the first target qubit.
            t2: The QIR value representing the second target qubit.

        """
        self._circuit.rxx(self._angle(angle), self._qubit(t1), self._qubit(t2))

    def _on_qis_ryy(self, call: pyqir.Call, angle: pyqir.Value, t1: pyqir.Value, t2: pyqir.Value) -> None:  # noqa: ARG002
        """Apply an RYY gate to the target qubits.

        Args:
            call: The QIR call instruction for the RYY gate.
            angle: The QIR value representing the rotation angle.
            t1: The QIR value representing the first target qubit.
            t2: The QIR value representing the second target qubit.

        """
        self._circuit.ryy(self._angle(angle), self._qubit(t1), self._qubit(t2))

    def _on_qis_rzz(self, call: pyqir.Call, angle: pyqir.Value, t1: pyqir.Value, t2: pyqir.Value) -> None:  # noqa: ARG002
        """Apply an RZZ gate to the target qubits.

        Args:
            call: The QIR call instruction for the RZZ gate.
            angle: The QIR value representing the rotation angle.
            t1: The QIR value representing the first target qubit.
            t2: The QIR value representing the second target qubit.

        """
        self._circuit.rzz(self._angle(angle), self._qubit(t1), self._qubit(t2))

    # =========================================================================
    # Three-qubit gates
    # =========================================================================

    def _on_qis_ccx(self, call: pyqir.Call, c1: pyqir.Value, c2: pyqir.Value, target: pyqir.Value) -> None:  # noqa: ARG002
        """Apply a CCX (Toffoli) gate to the target qubit.

        Args:
            call: The QIR call instruction for the CCX gate.
            c1: The QIR value representing the first control qubit.
            c2: The QIR value representing the second control qubit.
            target: The QIR value representing the target qubit.

        """
        self._circuit.ccx(self._qubit(c1), self._qubit(c2), self._qubit(target))

    # =========================================================================
    # Measurement and reset
    # =========================================================================

    def _on_qis_m(self, call: pyqir.Call, target: pyqir.Value, result: pyqir.Value) -> None:  # noqa: ARG002
        """Measure the target qubit and store the result in the classical bit.

        Args:
            call: The QIR call instruction for the measurement.
            target: The QIR value representing the target qubit.
            result: The QIR value representing the classical bit to store the measurement result.

        """
        self._circuit.measure(self._qubit(target), self._clbit(result))

    def _on_qis_mz(self, call: pyqir.Call, target: pyqir.Value, result: pyqir.Value) -> None:  # noqa: ARG002
        """Measure the target qubit in the Z basis and store the result in the classical bit.

        Args:
            call: The QIR call instruction for the measurement in the Z basis.
            target: The QIR value representing the target qubit.
            result: The QIR value representing the classical bit to store the measurement result.

        """
        self._circuit.measure(self._qubit(target), self._clbit(result))

    def _on_qis_mresetz(self, call: pyqir.Call, target: pyqir.Value, result: pyqir.Value) -> None:  # noqa: ARG002
        """Measure the target qubit in the Z basis, store the result in the classical bit, and reset the qubit.

        Args:
            call: The QIR call instruction for the measurement and reset in the Z basis.
            target: The QIR value representing the target qubit.
            result: The QIR value representing the classical bit to store the measurement result.

        """
        self._circuit.measure(self._qubit(target), self._clbit(result))
        self._circuit.reset(self._qubit(target))

    def _on_qis_reset(self, call: pyqir.Call, target: pyqir.Value) -> None:  # noqa: ARG002
        """Reset the target qubit to the |0> state.

        Args:
            call: The QIR call instruction for the reset operation.
            target: The QIR value representing the target qubit.

        """
        self._circuit.reset(self._qubit(target))

    # =========================================================================
    # Unsupported operations
    # =========================================================================

    def _on_qis_read_result(self, call: pyqir.Call, result: pyqir.Value) -> None:
        """Ignore a result read; the value is consumed by the branch terminator.

        Args:
            call: The QIR call instruction reading the result.
            result: The QIR value representing the measurement result.

        """


def qir_ir_to_qiskit(ir: str) -> QuantumCircuit:
    """Convert QIR LLVM IR text to a Qiskit QuantumCircuit."""
    return QirToQiskitConverter().convert(pyqir.Module.from_ir(pyqir.Context(), ir))
