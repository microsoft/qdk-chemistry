"""Tests for the alias sampling and QROM state preparation algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.state_preparation.alias_sampling import AliasSamplingStatePreparation
from qdk_chemistry.algorithms.state_preparation.qrom_state_prep import QROMStatePreparation


class TestAliasSamplingStatePreparation:
    """Tests for the alias sampling state preparation algorithm."""

    def test_name(self):
        """Test algorithm name."""
        prep = AliasSamplingStatePreparation()
        assert prep.name() == "alias_sampling"

    def test_type_name(self):
        """Test algorithm type name."""
        prep = AliasSamplingStatePreparation()
        assert prep.type_name() == "state_prep"

    def test_bits_precision_default(self):
        """Test default bits precision is 10."""
        prep = AliasSamplingStatePreparation()
        assert prep.bits_precision == 10

    def test_bits_precision_custom(self):
        """Test custom bits precision."""
        prep = AliasSamplingStatePreparation(bits_precision=7)
        assert prep.bits_precision == 7

    def test_prepare_from_statevector_returns_circuit(self):
        """Test that prepare_from_statevector returns a Circuit with ops set."""
        prep = AliasSamplingStatePreparation(bits_precision=4)
        statevector = np.array([0.5, 0.3, 0.7, 0.1])
        circuit = prep.prepare_from_statevector(
            statevector=statevector,
            num_qubits=2,
            qubit_indices=[0, 1],
        )
        assert circuit is not None
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_run_impl_raises(self):
        """Test that _run_impl raises NotImplementedError."""
        prep = AliasSamplingStatePreparation()
        with pytest.raises(NotImplementedError, match="prepare_from_statevector"):
            prep.run(None)

    def test_registered_in_registry(self):
        """Test that alias_sampling is registered in the algorithm registry."""
        prep = registry.create("state_prep", "alias_sampling")
        assert isinstance(prep, AliasSamplingStatePreparation)


class TestQROMStatePreparation:
    """Tests for the QROM-based state preparation algorithm."""

    def test_name(self):
        """Test algorithm name."""
        prep = QROMStatePreparation()
        assert prep.name() == "qrom_state_prep"

    def test_type_name(self):
        """Test algorithm type name."""
        prep = QROMStatePreparation()
        assert prep.type_name() == "state_prep"

    def test_rotation_bit_precision_default(self):
        """Test default rotation bit precision is 10."""
        prep = QROMStatePreparation()
        assert prep.rotation_bit_precision == 10

    def test_rotation_bit_precision_custom(self):
        """Test custom rotation bit precision."""
        prep = QROMStatePreparation(rotation_bit_precision=8)
        assert prep.rotation_bit_precision == 8

    def test_prepare_from_statevector_returns_circuit(self):
        """Test that prepare_from_statevector returns a Circuit with ops set."""
        prep = QROMStatePreparation(rotation_bit_precision=4)
        statevector = np.array([0.5, 0.3, 0.7, 0.1])
        circuit = prep.prepare_from_statevector(
            statevector=statevector,
            num_qubits=2,
            qubit_indices=[0, 1],
        )
        assert circuit is not None
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_run_impl_raises(self):
        """Test that _run_impl raises NotImplementedError."""
        prep = QROMStatePreparation()
        with pytest.raises(NotImplementedError, match="prepare_from_statevector"):
            prep.run(None)

    def test_registered_in_registry(self):
        """Test that qrom_state_prep is registered in the algorithm registry."""
        prep = registry.create("state_prep", "qrom_state_prep")
        assert isinstance(prep, QROMStatePreparation)
