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
    RotatedPauliContainer,
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


def test_rotated_pauli_container_is_a_qubit_operator_container() -> None:
    """Rotated-Pauli data is represented by its own container."""
    container = RotatedPauliContainer(
        ["IIIX"],
        np.array([0.5]),
        [np.array([1.0, 0.0])],
        4,
        encoding="jordan-wigner",
        fermion_mode_order="blocked",
    )
    operator = QubitOperator(container)

    assert operator.get_container_type() == "rotated_pauli"
    assert container.num_qubits == 4
    np.testing.assert_allclose(container.coefficients, np.array([0.5]))
    assert QubitOperator.from_json(operator.to_json()).get_container_type() == "rotated_pauli"

    for access in (
        lambda: operator.pauli_strings,
        lambda: operator.to_matrix(),
        lambda: operator.is_hermitian(),
        lambda: 2 * operator,
    ):
        with pytest.raises(TypeError, match="only available for Pauli-LCU operators"):
            access()


def test_sos_container_stores_block_encoding_data() -> None:
    """The SOS container splits the generators into d1/q1/sf rotated-Pauli combinations."""
    d1 = RotatedPauliContainer(["IZ"], np.array([0.4]), [np.array([0.1])], 4, "jordan-wigner", "blocked")
    q1 = RotatedPauliContainer(["ZI"], np.array([0.6]), [np.array([0.2])], 4, "jordan-wigner", "blocked")
    sf = RotatedPauliContainer(["ZZ"], np.array([0.5]), [np.array([0.3])], 4, "jordan-wigner", "blocked")
    container = SOSSAContainer(
        num_spatial_orbitals=2,
        num_qubits=4,
        energy_shift=-1.5,
        num_ranks=1,
        num_bases=1,
        num_copies=1,
        d1=d1,
        q1=q1,
        sf=sf,
        inner_coefficients=np.array([[1.0, 0.0], [1.0, 0.0], [0.3, 0.7]]),
        encoding="jordan-wigner",
        fermion_mode_order="blocked",
    )
    operator = QubitOperator(container)

    assert operator.get_container_type() == "sossa"
    assert container.num_positive_one_body_terms == 1
    np.testing.assert_allclose(container.d1.coefficients, np.array([0.4]))
    restored = QubitOperator.from_json(operator.to_json()).get_container()
    assert restored.type == "sossa"
    np.testing.assert_allclose(restored.inner_coefficients, container.inner_coefficients)
    np.testing.assert_allclose(restored.sf.rotations[0], container.sf.rotations[0])
