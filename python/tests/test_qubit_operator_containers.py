"""Tests for the qubit operator wrapper and representation containers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import h5py
import numpy as np
import pytest

from qdk_chemistry.data import QubitOperator
from qdk_chemistry.data.qubit_operator.containers.base import QubitOperatorContainer
from qdk_chemistry.data.qubit_operator.containers.pauli_lcu import PauliLCUContainer
from qdk_chemistry.data.qubit_operator.containers.sossa import RotatedPaulis, SOSSAContainer


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


def test_qubit_operator_requires_coefficients_with_pauli_strings() -> None:
    """Pauli strings alone are neither a container nor a complete legacy call."""
    with pytest.raises(TypeError, match="QubitOperator requires a QubitOperatorContainer"):
        QubitOperator(["X"])  # type: ignore[arg-type]


def test_qubit_operator_still_accepts_the_legacy_positional_constructor() -> None:
    """``QubitOperator(pauli_strings, coefficients)`` is a shipped signature and still works."""
    operator = QubitOperator(["XI", "ZZ"], np.array([0.5, -0.25]))

    assert operator.get_container_type() == "pauli_lcu"
    assert operator.pauli_strings == ["XI", "ZZ"]
    np.testing.assert_allclose(operator.coefficients, np.array([0.5, -0.25]))


def test_qubit_operator_reads_documents_written_before_container_dispatch() -> None:
    """A document with no ``container_type`` loads as a Pauli LCU operator.

    Releases through 2.1.0 wrote no ``container_type`` and this PR did not bump
    ``_serialization_version``, so the version guard cannot tell those documents apart
    from current ones. Defaulting the missing key is what keeps them readable.
    """
    current = QubitOperator(PauliLCUContainer(["XI", "ZZ"], np.array([0.5, -0.25]), "jordan-wigner")).to_json()
    legacy = {key: value for key, value in current.items() if key != "container_type"}
    assert "container_type" not in legacy

    restored = QubitOperator.from_json(legacy)

    assert restored.get_container_type() == "pauli_lcu"
    assert restored.pauli_strings == ["XI", "ZZ"]
    np.testing.assert_allclose(restored.coefficients, np.array([0.5, -0.25]))


def test_qubit_operator_reads_hdf5_groups_written_before_container_dispatch(tmp_path) -> None:
    """The same missing-``container_type`` default applies to HDF5 groups."""
    operator = QubitOperator(PauliLCUContainer(["XI", "ZZ"], np.array([0.5, -0.25]), "jordan-wigner"))
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("operator")
        operator.to_hdf5(group)
        del group.attrs["container_type"]

    with h5py.File(path, "r") as handle:
        restored = QubitOperator.from_hdf5(handle["operator"])

    assert restored.get_container_type() == "pauli_lcu"
    assert restored.pauli_strings == ["XI", "ZZ"]


def test_sos_container_json_roundtrip_preserves_complex_coefficients() -> None:
    """Complex LCU coefficients and Givens angles survive a JSON round-trip.

    The SOS generators carry the D1/Q1 ``+/-i`` sign in the imaginary part, so a
    serializer that silently drops it would still produce a well-formed container
    while flipping particle generators into hole generators.
    """
    one_body_coeffs = np.array([[0.2, 0.2j], [0.3, -0.3j]])
    container = SOSSAContainer(
        num_spatial_orbitals=2,
        energy_shift=-1.5,
        num_ranks=1,
        num_bases=1,
        num_copies=1,
        one_body=RotatedPaulis(np.array([[0.1], [0.2]]), one_body_coeffs, ("X", "Y")),
        num_positive_one_body_terms=1,
        two_body=RotatedPaulis(np.array([[0.3]]), np.array([[0.3, 0.7]]), ("Z",)),
        encoding="jordan-wigner",
        fermion_mode_order="blocked",
    )

    restored = QubitOperator.from_json(QubitOperator(container).to_json()).get_container()

    np.testing.assert_allclose(restored.one_body.coeffs, one_body_coeffs)
    np.testing.assert_allclose(restored.one_body.angles, container.one_body.angles)
    np.testing.assert_allclose(restored.two_body.coeffs, container.two_body.coeffs)
    np.testing.assert_allclose(restored.two_body.angles, container.two_body.angles)
    assert restored.num_positive_one_body_terms == 1
    assert restored.energy_shift == pytest.approx(-1.5)
