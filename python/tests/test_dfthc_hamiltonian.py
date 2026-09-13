"""Tests for DFTHC Hamiltonian construction and persistence."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from qdk_chemistry.data import DFTHCHamiltonianContainer, Hamiltonian, HamiltonianType, ModelOrbitals

from .test_helpers import create_random_factorized_hamiltonian


def test_ordinary_df_overload_matches_general() -> None:
    """Omitting WB preserves the factors, conventional integrals, and metadata."""
    one_body = np.array([[1.0, 0.3], [0.3, 1.0]])
    u = np.array([0.8, 0.6, -0.6, 0.8, 1.0, 0.0, 0.0, 1.0])
    w = np.array([0.5, -0.3, 0.25, 0.6])
    orbitals = ModelOrbitals(2)
    inactive_fock = np.array([[0.5, 0.1], [0.2, 0.6]])
    general = DFTHCHamiltonianContainer(
        one_body, u, w, np.zeros((2, 1)), orbitals, 1.5, inactive_fock, HamiltonianType.NonHermitian
    )
    ordinary = DFTHCHamiltonianContainer(
        one_body_integrals=one_body,
        u_matrices=u,
        w_matrices=w,
        orbitals=orbitals,
        core_energy=1.5,
        inactive_fock_matrix=inactive_fock,
        type=HamiltonianType.NonHermitian,
    )

    assert ordinary.get_num_orbitals() == 2
    assert ordinary.get_num_ranks() == 2
    assert ordinary.get_num_bases() == 2
    assert ordinary.get_num_copies() == 1
    np.testing.assert_array_equal(ordinary.get_u_matrices(), u)
    np.testing.assert_array_equal(ordinary.get_w_matrices(), w)
    np.testing.assert_array_equal(ordinary.get_wb_matrix(), np.zeros((2, 1)))
    np.testing.assert_array_equal(ordinary.get_one_body_integrals()[0], one_body)
    np.testing.assert_array_equal(ordinary.get_inactive_fock_matrix()[0], inactive_fock)
    assert ordinary.get_orbitals() is orbitals
    assert ordinary.get_core_energy() == 1.5
    assert ordinary.get_type() == HamiltonianType.NonHermitian
    np.testing.assert_array_equal(ordinary.reconstruct_two_body_integrals(), general.reconstruct_two_body_integrals())
    np.testing.assert_array_equal(ordinary.get_h1_prime(), general.get_h1_prime())
    assert ordinary.get_lambda() == general.get_lambda()
    assert Hamiltonian(ordinary).content_hash() == Hamiltonian(general).content_hash()


@pytest.mark.parametrize(
    ("num_orbitals", "u_size", "w_size", "match"),
    [
        (0, 0, 0, "rank"),
        (2, 3, 2, "U matrices size"),
        (2, 4, 3, "W matrices size"),
        (2, 6, 3, "Ordinary DF"),
        (2, 0, 2, "basis"),
        (2, 4, 0, "rank"),
        (2, 4, 1, "rank"),
    ],
)
def test_ordinary_df_overload_rejects_invalid_shapes(num_orbitals: int, u_size: int, w_size: int, match: str) -> None:
    """Inconsistent ordinary-DF dimensions fail without unsafe dimension inference."""
    with pytest.raises(ValueError, match=match):
        DFTHCHamiltonianContainer(
            np.eye(num_orbitals),
            np.full(u_size, 1.0 / np.sqrt(2)),
            np.ones(w_size),
            ModelOrbitals(2),
            0.0,
            np.eye(0),
        )


@pytest.mark.parametrize(
    ("num_ranks", "num_bases", "num_copies"),
    [(2, 2, 1), (1, 3, 2), (2, 2, 2)],
    ids=["ordinary_df", "shared_basis", "general_dfthc"],
)
def test_json_hdf5_roundtrips(num_ranks: int, num_bases: int, num_copies: int, tmp_path: Path) -> None:
    """Hamiltonian dispatch preserves DF, shared-basis, and general DFTHC data."""
    source = create_random_factorized_hamiltonian(
        num_orbitals=2, num_ranks=num_ranks, num_bases=num_bases, num_copies=num_copies
    )
    one_body = source.get_one_body_integrals()[0]
    w = source.get_w_matrices()
    orbitals = source.get_orbitals()
    inactive_fock = np.array([[0.5, 0.1], [0.2, 0.6]])
    if num_copies == 1:
        u = np.array([0.8, 0.6, -0.6, 0.8, 1.0, 0.0, 0.0, 1.0])
        container = DFTHCHamiltonianContainer(
            one_body, u, w, orbitals, 1.5, inactive_fock, HamiltonianType.NonHermitian
        )
    else:
        container = DFTHCHamiltonianContainer(
            one_body,
            source.get_u_matrices(),
            w,
            source.get_wb_matrix(),
            orbitals,
            1.5,
            inactive_fock,
            HamiltonianType.NonHermitian,
        )
    hamiltonian = Hamiltonian(container)
    serialized = hamiltonian.to_json()
    payload = json.loads(serialized)
    assert payload["container"]["container_type"] == "factorized"
    assert payload["container"]["version"] == "0.2.0"
    np.testing.assert_array_equal(payload["container"]["one_body_integrals"], one_body)

    filename = str(tmp_path / "dfthc.hamiltonian.h5")
    hamiltonian.to_hdf5_file(filename)
    with h5py.File(filename, "r") as handle:
        assert handle["container"].attrs["container_type"] == "factorized"
        assert handle["container"].attrs["version"] == "0.2.0"
        np.testing.assert_array_equal(handle["container/one_body_integrals"][:], one_body)

    reference = hamiltonian.get_container()
    for restored in (Hamiltonian.from_json(serialized), Hamiltonian.from_hdf5_file(filename)):
        restored_container = restored.get_container()
        assert isinstance(restored_container, DFTHCHamiltonianContainer)
        assert restored.content_hash() == hamiltonian.content_hash()
        assert restored_container.get_num_ranks() == num_ranks
        assert restored_container.get_num_bases() == num_bases
        assert restored_container.get_num_copies() == num_copies
        np.testing.assert_array_equal(restored_container.get_u_matrices(), reference.get_u_matrices())
        np.testing.assert_array_equal(restored_container.get_w_matrices(), reference.get_w_matrices())
        np.testing.assert_array_equal(restored_container.get_wb_matrix(), reference.get_wb_matrix())
        assert restored.get_orbitals().content_hash() == orbitals.content_hash()
        assert restored.get_core_energy() == 1.5
        assert restored.get_type() == HamiltonianType.NonHermitian
        np.testing.assert_array_equal(restored.get_inactive_fock_matrix()[0], inactive_fock)
        np.testing.assert_array_equal(restored.get_one_body_integrals(), hamiltonian.get_one_body_integrals())
        np.testing.assert_array_equal(restored.get_two_body_integrals(), hamiltonian.get_two_body_integrals())
        np.testing.assert_array_equal(restored_container.get_h1_prime(), reference.get_h1_prime())
        assert restored_container.get_lambda() == reference.get_lambda()


def test_existing_factorized_fixture_dispatches_to_dfthc() -> None:
    """Existing factorized wire data loads without rewriting its tag or fields."""
    filename = Path(__file__).parent / "test_data" / "h2_dfthc_r2_b2_c1.hamiltonian.json"
    hamiltonian = Hamiltonian.from_json(filename.read_text())
    assert isinstance(hamiltonian.get_container(), DFTHCHamiltonianContainer)
    np.testing.assert_array_equal(
        hamiltonian.get_one_body_integrals()[0], json.loads(filename.read_text())["container"]["one_body_integrals"]
    )
