"""Utility functions for QIR visitor passes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pyqir

__all__ = ["QirBitstringOrderVisitor", "get_qir_result_qubit_order"]


class QirBitstringOrderVisitor(pyqir.QirModuleVisitor):
    """Extract the output bitstring qubit ordering from a QIR module.

    The visitor pass tracks:

    1. **Measurement mapping**: ``__quantum__qis__m__body`` calls that bind a
       ``result_id`` to the ``qubit_id`` that was measured.
    2. **Record order**: the sequence of ``result_id`` values passed to
       ``__quantum__rt__result_record_output``, which defines the bitstring
       position order.
    """

    def __init__(self) -> None:
        """Initialize the visitor with empty state."""
        super().__init__()
        self.result_to_qubit: dict[int, int] = {}
        self.result_order: list[int] = []

    def _on_qis_m(self, call: pyqir.Call, target: pyqir.Value, result: pyqir.Value) -> None:  # noqa: ARG002
        """Track the mapping from measurement result ID to qubit ID."""
        qid = pyqir.qubit_id(target)
        rid = pyqir.result_id(result)
        if qid is not None and rid is not None:
            self.result_to_qubit[rid] = qid

    def _on_rt_result_record_output(self, call: pyqir.Call, result: pyqir.Value, target: pyqir.Value) -> None:  # noqa: ARG002
        """Track the order of result IDs in the output record."""
        rid = pyqir.result_id(result)
        if rid is not None:
            self.result_order.append(rid)


def get_qir_result_qubit_order(qir: str) -> list[int]:
    """Return the qubit ordering of the output bitstring produced by QIR.

    Args:
        qir: QIR as LLVM-IR text.

    Returns:
        List of qubit IDs in bitstring output order.
        ``qubit_order[i]`` is the qubit whose measurement result is at
        position *i* (i.e. character *i* of the bitstring / element *i* of
        the ``[Zero, One, ...]`` list).

    Example:
        qubit_order = get_qir_result_qubit_order(qir_ir_string)
        # qubit_order == [1, 0]  means position 0 is qubit 1, position 1 is qubit 0

    """
    context = pyqir.Context()
    module = pyqir.Module.from_ir(context, qir)
    visitor = QirBitstringOrderVisitor()
    visitor.run(module)
    qubit_order: list[int] = []
    for rid in visitor.result_order:
        if rid not in visitor.result_to_qubit:
            raise ValueError(
                f"Result ID {rid} appears in result_record_output but has no corresponding measurement instruction."
            )
        qubit_order.append(visitor.result_to_qubit[rid])
    return qubit_order
