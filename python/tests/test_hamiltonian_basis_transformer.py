"""Tests for the Hamiltonian basis-transformer Python API."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import QdkHamiltonianBasisTransformer, available, create, inspect_settings
from qdk_chemistry.data import CholeskyHamiltonianContainer, Hamiltonian, Orbitals, SettingsAreLocked
from qdk_chemistry.data.symmetry import spin_index_set

from .test_helpers import create_test_basis_set, create_test_hamiltonian


def test_qdk_transformer_registry():
    """The concrete transformer is registered with its native run binding."""
    assert "qdk" in available("hamiltonian_basis_transformer")
    transformer = create("hamiltonian_basis_transformer")
    assert isinstance(transformer, QdkHamiltonianBasisTransformer)
    assert "run" in QdkHamiltonianBasisTransformer.__dict__
    assert transformer.name() == "qdk"
    assert transformer.settings().get("validation_tolerance") == pytest.approx(1.0e-10)
    assert inspect_settings("hamiltonian_basis_transformer", "qdk")[0][0] == "validation_tolerance"


def test_qdk_transformer_rejects_non_cholesky_hamiltonian():
    """The QDK implementation rejects unsupported Hamiltonian containers."""
    source = create_test_hamiltonian(2)
    transformer = create("hamiltonian_basis_transformer")
    assert transformer.hash(source, source.get_orbitals())

    with pytest.raises(ValueError, match="requires a Cholesky Hamiltonian"):
        transformer.run(source, source.get_orbitals())

    with pytest.raises(SettingsAreLocked):
        transformer.settings().set("validation_tolerance", 1.0e-9)


@pytest.mark.parametrize("source_null_eigenvalue", [0.0, 1.0e-30])
def test_qdk_transformer_rejects_target_metric_null_mode_amplification(source_null_eigenvalue):
    """Differences in a source-null AO mode cannot become large in the target metric."""
    angle = 0.3
    source_coefficients = np.eye(3)
    target_coefficients = source_coefficients.copy()
    target_coefficients[:, 0] = [np.cos(angle), np.sin(angle), 1.0e6]
    target_coefficients[:, 1] = [-np.sin(angle), np.cos(angle), 0.0]
    source_overlap = np.diag([1.0, 1.0, source_null_eigenvalue])
    target_overlap = source_overlap.copy()
    target_overlap[2, 2] = 5.0e-11
    basis_set = create_test_basis_set(3, "test-python-target-metric-transform")
    active_indices = spin_index_set(3, [0, 1], [0, 1])
    inactive_indices = spin_index_set(3, [2], [2])
    source_orbitals = Orbitals(
        source_coefficients,
        None,
        source_overlap,
        basis_set,
        active_indices,
        inactive_indices,
    )
    target_orbitals = Orbitals(
        target_coefficients,
        None,
        target_overlap,
        basis_set,
        active_indices,
        inactive_indices,
    )
    source = Hamiltonian(
        CholeskyHamiltonianContainer(
            np.eye(2),
            np.ones((4, 1)),
            source_orbitals,
            0.0,
            np.empty((0, 0)),
        )
    )

    with pytest.raises(ValueError, match="target AO metric"):
        create("hamiltonian_basis_transformer").run(source, target_orbitals)


def test_qdk_transformer_runs_successfully():
    """The transformer rotates every supported Hamiltonian component."""
    angle = 0.3
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    basis_set = create_test_basis_set(2, "test-python-basis-transform")
    active_indices = spin_index_set(2, [0, 1], [0, 1])
    source_orbitals = Orbitals(
        np.eye(2),
        None,
        np.eye(2),
        basis_set,
        active_indices,
        spin_index_set(2, [], []),
    )
    target_orbitals = Orbitals(
        rotation,
        None,
        np.eye(2),
        basis_set,
        active_indices,
        spin_index_set(2, [], []),
    )

    one_body = np.array([[1.2, -0.3], [-0.3, 0.7]])
    factor_0 = np.array([[0.9, 0.2], [0.2, 0.4]])
    factor_1 = np.array([[0.1, -0.5], [-0.5, 0.8]])
    factors = (factor_0, factor_1)
    three_center = np.column_stack([factor.reshape(-1, order="F") for factor in factors])
    inactive_fock = np.array([[2.0, 0.1], [0.1, 1.7]])
    source = Hamiltonian(
        CholeskyHamiltonianContainer(
            one_body,
            three_center,
            source_orbitals,
            1.25,
            inactive_fock,
        )
    )

    transformed = create("hamiltonian_basis_transformer").run(source, target_orbitals)

    np.testing.assert_allclose(
        transformed.get_one_body_integrals()[0],
        rotation.T @ one_body @ rotation,
        atol=1.0e-13,
    )
    transformed_factors = [rotation.T @ factor @ rotation for factor in factors]
    for p in range(2):
        for q in range(2):
            for r in range(2):
                for s in range(2):
                    expected = sum(factor[p, q] * factor[r, s] for factor in transformed_factors)
                    assert transformed.get_two_body_element(p, q, r, s) == pytest.approx(expected, abs=1.0e-13)
    np.testing.assert_allclose(
        transformed.get_inactive_fock_matrix()[0],
        rotation.T @ inactive_fock @ rotation,
        atol=1.0e-13,
    )
    assert transformed.get_core_energy() == pytest.approx(1.25)
    assert transformed.get_orbitals() is target_orbitals
