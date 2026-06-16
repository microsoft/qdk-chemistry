"""Equivalence tests for QDK qubit mapper sparse and Cholesky fast paths."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Callable

import numpy as np
import pytest
import scipy.sparse as sp

from qdk_chemistry._core.data import majorana_map_hamiltonian, sparse_pauli_word_to_label
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    CholeskyHamiltonianContainer,
    Hamiltonian,
    LatticeGraph,
    MajoranaMapping,
    Orbitals,
    QubitHamiltonian,
    SparseHamiltonianContainer,
    Structure,
)
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.utils.model_hamiltonians import (
    create_hubbard_hamiltonian,
    create_ppp_hamiltonian,
)

from .test_helpers import create_test_basis_set, create_test_orbitals

MappingFactory = Callable[[int], MajoranaMapping]

try:
    import pyscf  # noqa: F401
    import qdk_chemistry.plugins.pyscf as pyscf_plugin

    pyscf_plugin.load()
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


@pytest.fixture(
    params=[
        MajoranaMapping.jordan_wigner,
        MajoranaMapping.bravyi_kitaev,
        MajoranaMapping.parity,
    ],
    ids=["jw", "bk", "parity"],
)
def mapping_factory(request: pytest.FixtureRequest) -> MappingFactory:
    """Return a standard untapered Majorana mapping factory."""
    return request.param


def _term_dict(hamiltonian):
    terms: dict[str, complex] = {}
    for label, coeff in zip(hamiltonian.pauli_strings, hamiltonian.coefficients, strict=True):
        terms[label] = terms.get(label, 0.0j) + complex(coeff)
    return {label: coeff for label, coeff in terms.items() if abs(coeff) >= 1e-12}


def _assert_term_equivalent(actual, expected, *, atol: float = 1e-12) -> None:
    actual_terms = _term_dict(actual)
    expected_terms = _term_dict(expected)
    missing = sorted(set(expected_terms) - set(actual_terms))
    extra = sorted(set(actual_terms) - set(expected_terms))
    if missing:
        pytest.fail(f"Missing Pauli terms: {missing[:8]}")
    if extra:
        pytest.fail(f"Unexpected Pauli terms: {extra[:8]}")
    for label in sorted(expected_terms):
        if actual_terms[label] != pytest.approx(expected_terms[label], abs=atol):
            pytest.fail(
                f"Mismatched coefficient for {label}: "
                f"actual={actual_terms[label]!r}, expected={expected_terms[label]!r}"
            )


def _dense_reference(hamiltonian: Hamiltonian) -> Hamiltonian:
    h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
    h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()
    if hamiltonian.has_inactive_fock_matrix():
        fock_alpha, fock_beta = hamiltonian.get_inactive_fock_matrix()
    else:
        fock_alpha = np.eye(0)
        fock_beta = np.eye(0)
    orbitals = hamiltonian.get_orbitals()
    core_energy = hamiltonian.get_core_energy()

    if orbitals.is_restricted():
        container = CanonicalFourCenterHamiltonianContainer(
            np.array(h1_alpha, copy=True),
            np.array(h2_aaaa, copy=True),
            orbitals,
            core_energy,
            np.array(fock_alpha, copy=True),
        )
    else:
        container = CanonicalFourCenterHamiltonianContainer(
            np.array(h1_alpha, copy=True),
            np.array(h1_beta, copy=True),
            np.array(h2_aaaa, copy=True),
            np.array(h2_aabb, copy=True),
            np.array(h2_bbbb, copy=True),
            orbitals,
            core_energy,
            np.array(fock_alpha, copy=True),
            np.array(fock_beta, copy=True),
        )
    return Hamiltonian(container)


def _native_result_to_qubit_hamiltonian(mapping: MajoranaMapping, words, coefficients) -> QubitHamiltonian:
    pauli_strings = [sparse_pauli_word_to_label(word, mapping.num_qubits) for word in words]
    return QubitHamiltonian(
        pauli_strings=pauli_strings,
        coefficients=np.array(coefficients, dtype=complex),
        encoding=mapping.base_encoding,
        fermion_mode_order=FermionModeOrder.BLOCKED,
    )


def _dense_overload_result_for_sparse(
    hamiltonian: Hamiltonian,
    mapping: MajoranaMapping,
    *,
    threshold: float = 1e-12,
    integral_threshold: float = 1e-12,
) -> QubitHamiltonian:
    h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
    h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()
    n_spatial = h1_alpha.shape[0]
    spin_symmetric = hamiltonian.get_orbitals().is_restricted()

    h1_alpha_flat = np.ascontiguousarray(h1_alpha).ravel()
    h1_beta_flat = h1_alpha_flat if spin_symmetric else np.ascontiguousarray(h1_beta).ravel()
    h2_aaaa_flat = np.ascontiguousarray(h2_aaaa).ravel()
    h2_aabb_flat = h2_aaaa_flat if spin_symmetric else np.ascontiguousarray(h2_aabb).ravel()
    h2_bbbb_flat = h2_aaaa_flat if spin_symmetric else np.ascontiguousarray(h2_bbbb).ravel()

    words, coefficients = majorana_map_hamiltonian(
        mapping,
        0.0,
        h1_alpha_flat,
        h1_beta_flat,
        h2_aaaa_flat,
        h2_aabb_flat,
        h2_bbbb_flat,
        n_spatial,
        spin_symmetric,
        threshold,
        integral_threshold,
    )
    return _native_result_to_qubit_hamiltonian(mapping, words, coefficients)


def _symmetric_three_center(n_orbitals: int, n_aux: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = np.zeros((n_orbitals * n_orbitals, n_aux))
    for p in range(n_orbitals):
        for q in range(p, n_orbitals):
            row = rng.standard_normal(n_aux) * 0.2
            factors[p * n_orbitals + q] = row
            factors[q * n_orbitals + p] = row
    return factors


def _symmetric_one_body(n_orbitals: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_orbitals, n_orbitals)) * 0.1
    return (raw + raw.T) / 2 + np.diag(np.linspace(0.8, -0.4, n_orbitals))


def _four_center_from_cholesky(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(left @ right.T).ravel()


def _restricted_cholesky_pair(n_orbitals: int, n_aux: int, seed: int) -> tuple[Hamiltonian, Hamiltonian]:
    h1 = _symmetric_one_body(n_orbitals, seed)
    three_center = _symmetric_three_center(n_orbitals, n_aux, seed + 100)
    orbitals = create_test_orbitals(n_orbitals)
    candidate = Hamiltonian(
        CholeskyHamiltonianContainer(
            np.array(h1, copy=True),
            np.array(three_center, copy=True),
            orbitals,
            0.0,
            np.eye(0),
        )
    )
    dense = Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(
            np.array(h1, copy=True),
            _four_center_from_cholesky(three_center, three_center),
            orbitals,
            0.0,
            np.eye(0),
        )
    )
    return candidate, dense


def _unrestricted_cholesky_pair(n_orbitals: int, n_aux: int, seed: int) -> tuple[Hamiltonian, Hamiltonian]:
    rng = np.random.default_rng(seed)
    coeffs_alpha = np.eye(n_orbitals)
    coeffs_beta = np.eye(n_orbitals) + rng.standard_normal((n_orbitals, n_orbitals)) * 0.05
    orbitals = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, create_test_basis_set(n_orbitals))
    h1_alpha = _symmetric_one_body(n_orbitals, seed)
    h1_beta = _symmetric_one_body(n_orbitals, seed + 1)
    three_center_alpha = _symmetric_three_center(n_orbitals, n_aux, seed + 100)
    three_center_beta = _symmetric_three_center(n_orbitals, n_aux, seed + 200)
    candidate = Hamiltonian(
        CholeskyHamiltonianContainer(
            np.array(h1_alpha, copy=True),
            np.array(h1_beta, copy=True),
            np.array(three_center_alpha, copy=True),
            np.array(three_center_beta, copy=True),
            orbitals,
            0.0,
            np.eye(0),
            np.eye(0),
        )
    )
    dense = Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(
            np.array(h1_alpha, copy=True),
            np.array(h1_beta, copy=True),
            _four_center_from_cholesky(three_center_alpha, three_center_alpha),
            _four_center_from_cholesky(three_center_alpha, three_center_beta),
            _four_center_from_cholesky(three_center_beta, three_center_beta),
            orbitals,
            0.0,
            np.eye(0),
            np.eye(0),
        )
    )
    return candidate, dense


def _sparse_hamiltonian(
    one_body: np.ndarray, two_body_entries: dict[tuple[int, int, int, int], float]
) -> Hamiltonian:
    return Hamiltonian(
        SparseHamiltonianContainer(
            sp.csr_matrix(one_body),
            two_body_entries,
            0.0,
        )
    )


def _molecular_hamiltonian_pair(
    structure: Structure,
    charge: int,
    spin_multiplicity: int,
    basis: str,
) -> tuple[Hamiltonian, Hamiltonian]:
    scf_solver = create("scf_solver", "pyscf")
    scf_solver.settings()["method"] = "hf"
    scf_solver.settings()["scf_type"] = "restricted"
    scf_solver.settings()["convergence_threshold"] = 1e-8
    _, wavefunction = scf_solver.run(structure, charge, spin_multiplicity, basis)
    orbitals = wavefunction.get_orbitals()

    cholesky = create("hamiltonian_constructor", "qdk_cholesky").run(orbitals)
    dense = _dense_reference(cholesky)
    return cholesky, dense


@pytest.mark.parametrize(
    ("n_orbitals", "n_aux", "seed"),
    [
        (2, 2, 11),
        (3, 3, 12),
        (4, 3, 13),
    ],
)
def test_cholesky_fast_path_matches_dense_restricted(
    mapping_factory: MappingFactory, n_orbitals: int, n_aux: int, seed: int
) -> None:
    """Restricted Cholesky-backed Hamiltonians map exactly like dense references."""
    hamiltonian, dense = _restricted_cholesky_pair(n_orbitals, n_aux, seed)
    mapping = mapping_factory(2 * n_orbitals)
    mapper = create("qubit_mapper", "qdk")

    _assert_term_equivalent(mapper.run(hamiltonian, mapping), mapper.run(dense, mapping))


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
@pytest.mark.parametrize(
    ("structure", "charge", "spin_multiplicity", "basis"),
    [
        pytest.param(
            Structure(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]), ["H", "H"]),
            0,
            1,
            "sto-3g",
            id="h2-sto-3g",
        ),
        pytest.param(
            Structure(
                np.array(
                    [
                        [0.000000, 0.000000, 0.000000],
                        [0.000000, 1.430429, 1.107157],
                        [0.000000, -1.430429, 1.107157],
                    ]
                ),
                ["O", "H", "H"],
            ),
            0,
            1,
            "sto-3g",
            id="h2o-sto-3g",
        ),
        pytest.param(
            Structure(np.array([[0.0, 0.0, -1.05], [0.0, 0.0, 1.05]]), ["N", "N"]),
            0,
            1,
            "sto-3g",
            id="n2-sto-3g",
        ),
    ],
)
def test_cholesky_fast_path_matches_dense_for_molecular_systems(
    mapping_factory: MappingFactory,
    structure: Structure,
    charge: int,
    spin_multiplicity: int,
    basis: str,
) -> None:
    """Real molecular Cholesky fast paths match dense references from the same factors."""
    hamiltonian, dense = _molecular_hamiltonian_pair(structure, charge, spin_multiplicity, basis)
    h1_alpha, _ = hamiltonian.get_one_body_integrals()
    mapping = mapping_factory(2 * h1_alpha.shape[0])
    mapper = create("qubit_mapper", "qdk")

    _assert_term_equivalent(mapper.run(hamiltonian, mapping), mapper.run(dense, mapping))


@pytest.mark.parametrize(
    ("n_orbitals", "n_aux", "seed"),
    [
        (2, 2, 21),
        (3, 3, 22),
    ],
)
def test_cholesky_fast_path_matches_dense_unrestricted(
    mapping_factory: MappingFactory, n_orbitals: int, n_aux: int, seed: int
) -> None:
    """Unrestricted Cholesky-backed Hamiltonians map exactly like dense references."""
    hamiltonian, dense = _unrestricted_cholesky_pair(n_orbitals, n_aux, seed)
    mapping = mapping_factory(2 * n_orbitals)
    mapper = create("qubit_mapper", "qdk")

    _assert_term_equivalent(mapper.run(hamiltonian, mapping), mapper.run(dense, mapping))


def test_cholesky_rejects_invalid_factor_shape() -> None:
    """Cholesky containers reject factors that cannot represent n_spatial^2 rows."""
    n_orbitals = 2
    orbitals = create_test_orbitals(n_orbitals)

    with pytest.raises(ValueError, match="n_active_alpha\\^2"):
        CholeskyHamiltonianContainer(
            np.eye(n_orbitals),
            np.ones((n_orbitals * n_orbitals - 1, 2)),
            orbitals,
            0.0,
            np.eye(0),
        )


def test_cholesky_rejects_mismatched_unrestricted_aux_dimensions() -> None:
    """Unrestricted alpha/beta Cholesky factors must share the auxiliary dimension."""
    n_orbitals = 2
    coeffs_alpha = np.eye(n_orbitals)
    coeffs_beta = np.eye(n_orbitals) + 0.1 * np.triu(np.ones((n_orbitals, n_orbitals)), 1)
    orbitals = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, create_test_basis_set(n_orbitals))

    with pytest.raises(ValueError, match="Beta three-center cols"):
        CholeskyHamiltonianContainer(
            np.eye(n_orbitals),
            np.eye(n_orbitals),
            np.ones((n_orbitals * n_orbitals, 2)),
            np.ones((n_orbitals * n_orbitals, 3)),
            orbitals,
            0.0,
            np.eye(0),
            np.eye(0),
        )


def _ppp_chain_hamiltonian() -> Hamiltonian:
    n_sites = 4
    lattice = LatticeGraph.chain(n_sites)
    epsilon = np.zeros(n_sites)
    hopping = np.ones((n_sites, n_sites))
    onsite = np.full(n_sites, 0.7)
    intersite = np.full((n_sites, n_sites), 0.2)
    np.fill_diagonal(intersite, 0.0)
    charge = np.ones(n_sites)
    return create_ppp_hamiltonian(lattice, epsilon=epsilon, t=hopping, U=onsite, V=intersite, z=charge)


@pytest.mark.parametrize(
    "hamiltonian_factory",
    [
        pytest.param(
            lambda: create_hubbard_hamiltonian(LatticeGraph.square(2, 2), epsilon=-0.5, t=1.0, U=0.3),
            id="hubbard-2x2",
        ),
        pytest.param(
            lambda: create_hubbard_hamiltonian(LatticeGraph.chain(4), epsilon=-0.5, t=1.0, U=0.4),
            id="hubbard-1x4",
        ),
        pytest.param(_ppp_chain_hamiltonian, id="ppp-chain"),
    ],
)
def test_sparse_fast_path_matches_dense(mapping_factory: MappingFactory, hamiltonian_factory) -> None:
    """Sparse model Hamiltonians map exactly like dense references."""
    hamiltonian = hamiltonian_factory()
    dense_source = hamiltonian_factory()
    h1_alpha, _ = hamiltonian.get_one_body_integrals()
    dense = _dense_reference(dense_source)
    mapping = mapping_factory(2 * h1_alpha.shape[0])
    mapper = create("qubit_mapper", "qdk")

    _assert_term_equivalent(mapper.run(hamiltonian, mapping), mapper.run(dense, mapping))


@pytest.mark.parametrize(
    "stored_key",
    [
        (0, 1, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
    ],
)
def test_sparse_fast_path_matches_raw_dense_overload_for_sparse_key(
    mapping_factory: MappingFactory, stored_key: tuple[int, int, int, int]
) -> None:
    one_body = np.array([[0.2, -0.7], [0.3, -0.1]])
    two_body = {stored_key: 0.6}
    hamiltonian = _sparse_hamiltonian(one_body, two_body)
    mapping = mapping_factory(4)
    mapper = create("qubit_mapper", "qdk")

    _assert_term_equivalent(mapper.run(hamiltonian, mapping), _dense_overload_result_for_sparse(hamiltonian, mapping))


def test_sparse_fast_path_keeps_symmetry_alias_coordinates_independent(mapping_factory: MappingFactory) -> None:
    one_body = np.array([[0.2, -0.7], [0.3, -0.1]])
    two_body = {(0, 1, 0, 1): 0.6, (1, 0, 1, 0): 0.7}
    hamiltonian = _sparse_hamiltonian(one_body, two_body)
    mapping = mapping_factory(4)
    mapper = create("qubit_mapper", "qdk")

    _assert_term_equivalent(mapper.run(hamiltonian, mapping), _dense_overload_result_for_sparse(hamiltonian, mapping))


def test_sparse_container_rejects_out_of_bounds_two_body_key() -> None:
    with pytest.raises(ValueError):
        _sparse_hamiltonian(np.eye(2), {(0, 0, 0, 2): 0.5})
