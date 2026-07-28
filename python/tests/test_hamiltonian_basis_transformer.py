"""Tests for the Hamiltonian basis-transformer Python API."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.algorithms import QdkHamiltonianBasisTransformer, available, create, inspect_settings

from .test_helpers import create_test_hamiltonian


def test_qdk_transformer_registry():
    assert "qdk" in available("hamiltonian_basis_transformer")
    transformer = create("hamiltonian_basis_transformer")
    assert isinstance(transformer, QdkHamiltonianBasisTransformer)
    assert transformer.name() == "qdk"
    assert transformer.settings().get("validation_tolerance") == pytest.approx(1.0e-10)
    assert inspect_settings("hamiltonian_basis_transformer", "qdk")[0][0] == "validation_tolerance"


def test_qdk_transformer_rejects_non_cholesky_hamiltonian():
    source = create_test_hamiltonian(2)
    transformer = create("hamiltonian_basis_transformer")
    assert transformer.hash(source, source.get_orbitals())

    with pytest.raises(ValueError, match="requires a Cholesky Hamiltonian"):
        transformer.run(source, source.get_orbitals())
