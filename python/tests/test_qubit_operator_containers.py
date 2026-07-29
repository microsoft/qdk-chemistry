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
from qdk_chemistry.data.sossa_qubit_operator import (
    RotatedMode,
    RotatedPauliTerm,
    SpinPolicy,
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
    mode = RotatedMode(np.array([1.0, 0.0]), spin=0)
    container = RotatedPauliContainer(
        [RotatedPauliTerm(0.5, {0: 1}, mode)],
        num_qubits=4,
        encoding="jordan-wigner",
        fermion_mode_order="blocked",
    )
    operator = QubitOperator(container)

    assert operator.get_container_type() == "rotated_pauli"
    assert container.lcu_normalization == pytest.approx(0.5)
    assert QubitOperator.from_json(operator.to_json()).get_container_type() == "rotated_pauli"

    for access in (
        lambda: operator.pauli_strings,
        lambda: operator.to_matrix(),
        lambda: operator.is_hermitian(),
        lambda: 2 * operator,
    ):
        with pytest.raises(TypeError, match="only available for Pauli-LCU operators"):
            access()


def test_sos_container_nests_qubit_operators() -> None:
    """SOS generators are wrapped Pauli representations carrying spin-policy metadata."""
    mode = RotatedMode(np.array([1.0, 0.0]), spin=0)
    nested = QubitOperator(
        RotatedPauliContainer(
            [RotatedPauliTerm(0.5, {0: 1}, mode)],
            num_qubits=4,
            encoding="jordan-wigner",
            fermion_mode_order="blocked",
            spin_policy=SpinPolicy.Specific,
            source_index=(0,),
        )
    )
    container = SOSSAContainer(2, 4, 0.0, [nested], "jordan-wigner", "blocked")
    operator = QubitOperator(container)

    assert operator.get_container_type() == "sossa"
    assert container.generators[0] is nested
    assert container.normalization == pytest.approx(0.125)
    restored = QubitOperator.from_json(operator.to_json())
    assert restored.get_container_type() == "sossa"
    assert isinstance(restored.get_container().generators[0], QubitOperator)
