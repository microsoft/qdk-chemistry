"""Container-aware QDK qubit mapper tests."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections import defaultdict

import numpy as np
import pytest
from scipy import sparse

import qdk_chemistry.algorithms.qubit_mapper.qdk_qubit_mapper as qdk_mapper_module
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    CholeskyHamiltonianContainer,
    Hamiltonian,
    LatticeGraph,
    MajoranaMapping,
    Orbitals,
    SparseHamiltonianContainer,
)
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian

from .test_helpers import create_test_basis_set, create_test_orbitals


def _dense_hamiltonian(one_body: np.ndarray, two_body: np.ndarray) -> Hamiltonian:
    orbitals = create_test_orbitals(one_body.shape[0])
    return Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 0.0, np.eye(0)))


def _dense_copy(hamiltonian: Hamiltonian) -> Hamiltonian:
    h1_alpha, _ = hamiltonian.get_one_body_integrals()
    eri_aaaa, _, _ = hamiltonian.get_two_body_integrals()
    return _dense_hamiltonian(h1_alpha, np.array(eri_aaaa))


def _unrestricted_orbitals(n_orbitals: int) -> Orbitals:
    coeffs_alpha = np.eye(n_orbitals)
    coeffs_beta = np.eye(n_orbitals)
    coeffs_beta[0, 0] = 0.9
    return Orbitals(coeffs_alpha, coeffs_beta, None, None, None, create_test_basis_set(n_orbitals))


def _set_coulomb_symmetric(two_body: np.ndarray, n_orbitals: int, indices, value: float) -> None:
    p, q, r, s = indices
    keys = set()
    for left in {(p, q), (q, p)}:
        for right in {(r, s), (s, r)}:
            keys.add((*left, *right))
            keys.add((*right, *left))

    for a, b, c, d in keys:
        two_body[((a * n_orbitals + b) * n_orbitals + c) * n_orbitals + d] = value


def _terms_by_pauli(qh) -> dict[str, complex]:
    assert len(qh.pauli_strings) == len(qh.coefficients)
    terms: defaultdict[str, complex] = defaultdict(complex)
    for pauli_string, coefficient in zip(qh.pauli_strings, qh.coefficients, strict=True):
        terms[pauli_string] += coefficient
    return dict(sorted(terms.items()))


def _assert_equivalent(left, right, atol: float = 1e-12) -> None:
    left_terms = _terms_by_pauli(left)
    right_terms = _terms_by_pauli(right)
    assert left_terms.keys() == right_terms.keys()
    for pauli_string, coefficient in left_terms.items():
        assert coefficient == pytest.approx(right_terms[pauli_string], abs=atol)


MAPPING_FACTORIES = [
    MajoranaMapping.jordan_wigner,
    MajoranaMapping.bravyi_kitaev,
    MajoranaMapping.parity,
]


def test_qdk_mapper_passes_hamiltonian_to_native_overload(monkeypatch) -> None:
    """The Python mapper should let the native layer choose the container path."""
    hamiltonian = _dense_hamiltonian(np.array([[1.0]]), np.zeros(1))
    mapping = MajoranaMapping.jordan_wigner(num_modes=2)
    seen = {}

    def fake_majorana_map_hamiltonian(mapping_arg, hamiltonian_arg, threshold, integral_threshold):
        seen["mapping"] = mapping_arg
        seen["hamiltonian"] = hamiltonian_arg
        seen["threshold"] = threshold
        seen["integral_threshold"] = integral_threshold
        return [[]], [1.0]

    monkeypatch.setattr(qdk_mapper_module, "majorana_map_hamiltonian", fake_majorana_map_hamiltonian)

    result = qdk_mapper_module.QdkQubitMapper(threshold=1e-9, integral_threshold=1e-8).run(hamiltonian, mapping)

    assert result.pauli_strings == ["II"]
    assert seen == {
        "mapping": mapping,
        "hamiltonian": hamiltonian,
        "threshold": 1e-9,
        "integral_threshold": 1e-8,
    }


@pytest.mark.parametrize(
    "mapping_factory",
    MAPPING_FACTORIES,
)
def test_sparse_container_canonicalizes_symmetry_related_two_body_keys(mapping_factory) -> None:
    """Sparse ERIs may be stored at a symmetry-related key, not the canonical dense key."""
    n_orbitals = 2
    one_body = np.zeros((n_orbitals, n_orbitals))
    value = 0.7

    dense_two_body = np.zeros(n_orbitals**4)
    _set_coulomb_symmetric(dense_two_body, n_orbitals, (0, 1, 0, 1), value)
    dense = _dense_hamiltonian(one_body, dense_two_body)

    sparse_container = SparseHamiltonianContainer(
        sparse.csc_matrix(one_body),
        {(1, 0, 0, 1): value},
    )
    sparse_hamiltonian = Hamiltonian(sparse_container)

    mapping = mapping_factory(num_modes=2 * n_orbitals)
    mapper = create("qubit_mapper", "qdk")

    _assert_equivalent(
        mapper.run(sparse_hamiltonian, mapping),
        mapper.run(dense, mapping),
    )


@pytest.mark.parametrize("mapping_factory", MAPPING_FACTORIES)
def test_sparse_model_hamiltonian_maps_like_dense_materialization(mapping_factory) -> None:
    """A real sparse model Hamiltonian should match its dense materialized path."""
    sparse_hamiltonian = create_hubbard_hamiltonian(
        LatticeGraph.chain(4),
        epsilon=-0.25,
        t=0.8,
        U=0.4,
    )
    dense_hamiltonian = _dense_copy(sparse_hamiltonian)

    mapping = mapping_factory(num_modes=8)
    mapper = create("qubit_mapper", "qdk")

    _assert_equivalent(
        mapper.run(sparse_hamiltonian, mapping),
        mapper.run(dense_hamiltonian, mapping),
    )


@pytest.mark.parametrize("mapping_factory", MAPPING_FACTORIES)
def test_cholesky_container_maps_like_dense_reconstruction(mapping_factory) -> None:
    """The Cholesky path should stream rows while preserving dense-path semantics."""
    n_orbitals = 2
    one_body = np.array([[0.4, -0.1], [-0.1, 0.2]])
    three_center = np.array(
        [
            [0.7, 0.0],
            [0.2, -0.3],
            [0.2, -0.3],
            [0.5, 0.4],
        ]
    )
    orbitals = create_test_orbitals(n_orbitals)
    cholesky = Hamiltonian(CholeskyHamiltonianContainer(one_body, three_center, orbitals, 0.0, np.eye(0)))

    dense_two_body = (three_center @ three_center.T).reshape(n_orbitals**4)
    dense = _dense_hamiltonian(one_body, dense_two_body)

    mapping = mapping_factory(num_modes=2 * n_orbitals)
    mapper = create("qubit_mapper", "qdk")

    _assert_equivalent(
        mapper.run(cholesky, mapping),
        mapper.run(dense, mapping),
    )


@pytest.mark.parametrize("mapping_factory", MAPPING_FACTORIES)
def test_unrestricted_cholesky_container_maps_like_dense_reconstruction(mapping_factory) -> None:
    """The Cholesky row path should preserve separate alpha/beta spin channels."""
    n_orbitals = 2
    one_body_alpha = np.array([[0.4, -0.1], [-0.1, 0.2]])
    one_body_beta = np.array([[0.5, 0.03], [0.03, 0.1]])
    three_center_alpha = np.array(
        [
            [0.7, 0.0],
            [0.2, -0.3],
            [0.1, 0.4],
            [0.5, 0.2],
        ]
    )
    three_center_beta = np.array(
        [
            [0.6, 0.1],
            [-0.2, 0.2],
            [0.3, -0.1],
            [0.4, 0.5],
        ]
    )
    orbitals = _unrestricted_orbitals(n_orbitals)
    cholesky = Hamiltonian(
        CholeskyHamiltonianContainer(
            one_body_alpha,
            one_body_beta,
            three_center_alpha,
            three_center_beta,
            orbitals,
            0.0,
            np.eye(0),
            np.eye(0),
        )
    )

    dense = Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(
            one_body_alpha,
            one_body_beta,
            (three_center_alpha @ three_center_alpha.T).reshape(n_orbitals**4),
            (three_center_alpha @ three_center_beta.T).reshape(n_orbitals**4),
            (three_center_beta @ three_center_beta.T).reshape(n_orbitals**4),
            orbitals,
            0.0,
            np.eye(0),
            np.eye(0),
        )
    )

    mapping = mapping_factory(num_modes=2 * n_orbitals)
    mapper = create("qubit_mapper", "qdk")

    _assert_equivalent(
        mapper.run(cholesky, mapping),
        mapper.run(dense, mapping),
    )
