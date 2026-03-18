"""Tests for QIR utility functions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pyqir

from qdk_chemistry.utils.qir import QirBitstringOrderVisitor, get_qir_result_qubit_order


class TestGetQirResultQubitOrder:
    """Tests for get_qir_result_qubit_order."""

    def test_sequential_record_from_ll(self, test_data_files_path):
        """5-qubit sequential: record [r0,r1,r2,r3,r4] → qubit order [0,1,2,3,4]."""
        qir = (test_data_files_path / "test_qir_sequential_record.ll").read_text()
        order = get_qir_result_qubit_order(qir)
        assert order == [0, 1, 2, 3, 4]

    def test_reversed_record_from_ll(self, test_data_files_path):
        """5-qubit reversed: record [r4,r3,r2,r1,r0] → qubit order [4,3,2,1,0]."""
        qir = (test_data_files_path / "test_qir_reversed_record.ll").read_text()
        order = get_qir_result_qubit_order(qir)
        assert order == [4, 3, 2, 1, 0]

    def test_circuit_qir_from_ll(self, test_data_files_path):
        """2-qubit reversed from test_circuit_qir.ll → qubit order [1, 0]."""
        qir = (test_data_files_path / "test_circuit_qir.ll").read_text()
        order = get_qir_result_qubit_order(qir)
        assert len(order) == 2
        assert order == [1, 0]


class TestQirBitstringOrderVisitor:
    """Tests for the QirBitstringOrderVisitor class."""

    def test_visitor_initial_state(self):
        """Test that a newly created visitor has empty state."""
        visitor = QirBitstringOrderVisitor()
        assert visitor.result_to_qubit == {}
        assert visitor.result_order == []

    def test_visitor_sequential(self, test_data_files_path):
        """Visitor on sequential .ll: result_to_qubit is identity, order is [0..4]."""
        qir = (test_data_files_path / "test_qir_sequential_record.ll").read_text()
        context = pyqir.Context()
        module = pyqir.Module.from_ir(context, qir)
        visitor = QirBitstringOrderVisitor()
        visitor.run(module)
        assert visitor.result_to_qubit == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        assert visitor.result_order == [0, 1, 2, 3, 4]

    def test_visitor_reversed(self, test_data_files_path):
        """Visitor on reversed .ll: result_to_qubit is identity, order is [4..0]."""
        qir = (test_data_files_path / "test_qir_reversed_record.ll").read_text()
        context = pyqir.Context()
        module = pyqir.Module.from_ir(context, qir)
        visitor = QirBitstringOrderVisitor()
        visitor.run(module)
        assert visitor.result_to_qubit == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        assert visitor.result_order == [4, 3, 2, 1, 0]
        for rid in visitor.result_order:
            assert rid in visitor.result_to_qubit
