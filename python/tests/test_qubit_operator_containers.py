"""Tests for the qubit operator wrapper and representation containers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.data import (
    PauliLCUContainer,
    QubitOperator,
    QubitOperatorContainer,
    RotatedPaulis,
    SOSSAContainer,
)


def test_qubit_operator_wraps_pauli_lcu_container() -> None:
    """Shared metadata and the compatibility Pauli API delegate to the container."""
    container = PauliLCUContainer(["XI", "ZZ"], np.array([0.5, -0.25]), "jordan-wigner", "blocked")
    operator = QubitOperator(container)

    assert isinstance(container, QubitOperatorContainer)
    assert operator.get_container() is container
    assert operator.get_container_type() == "pauli_lcu"
    assert operator.num_qubits == 2
    assert operator.encoding == "jordan-wigner"
    assert operator.fermion_mode_order == "blocked"
    assert operator.pauli_strings is container.pauli_strings
    assert operator.coefficients is container.coefficients
    assert operator.schatten_norm == container.schatten_norm
    np.testing.assert_allclose(operator.to_matrix(), container.to_matrix())
    assert operator.is_hermitian()
    assert operator.equiv(QubitOperator(PauliLCUContainer(["ZZ", "XI"], np.array([-0.25, 0.5]))))

    scaled = 2 * operator
    added = operator + operator
    assert isinstance(scaled, QubitOperator)
    assert isinstance(added, QubitOperator)
    np.testing.assert_allclose(scaled.coefficients, np.array([1.0, -0.5]))
    assert added.pauli_strings == ["XI", "ZZ", "XI", "ZZ"]

    restored = QubitOperator.from_json(operator.to_json())
    assert restored.get_container_type() == "pauli_lcu"
    assert restored.content_hash() == operator.content_hash()


def test_qubit_operator_rejects_legacy_constructor() -> None:
    """The removed direct Pauli constructor is intentionally unsupported."""
    with pytest.raises(TypeError, match="QubitOperator requires a QubitOperatorContainer"):
        QubitOperator(["X"])  # type: ignore[arg-type]


def test_sos_container_stores_block_encoding_data() -> None:
    """The SOS container stores per-block Givens angles, LCU coefficients, and Pauli terms."""
    container = SOSSAContainer(
        num_spatial_orbitals=2,
        energy_shift=-1.5,
        num_ranks=1,
        num_bases=1,
        num_copies=1,
        one_body=RotatedPaulis(np.array([[0.1], [0.2]]), np.array([[0.2, 0.2j], [0.3, -0.3j]]), ("X", "Y")),
        num_positive_one_body_terms=1,
        two_body=RotatedPaulis(np.array([[0.3]]), np.array([[0.3, 0.7]]), ("Z",)),
        encoding="jordan-wigner",
        fermion_mode_order="blocked",
    )
    operator = QubitOperator(container)

    assert operator.get_container_type() == "sossa"
    assert container.num_positive_one_body_terms == 1
    assert container.one_body.paulis == ("X", "Y")
    assert container.two_body.paulis == ("Z",)
    np.testing.assert_allclose(container.one_body.coeffs, np.array([[0.2, 0.2j], [0.3, -0.3j]]))
    restored = QubitOperator.from_json(operator.to_json()).get_container()
    assert restored.type == "sossa"
    np.testing.assert_allclose(restored.two_body.coeffs, container.two_body.coeffs)
    np.testing.assert_allclose(restored.two_body.angles, container.two_body.angles)
    np.testing.assert_allclose(restored.one_body.coeffs, container.one_body.coeffs)
    assert restored.num_positive_one_body_terms == 1
