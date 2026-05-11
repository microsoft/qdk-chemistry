"""Test OpenFermion Qubit Mapper and conversion utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pytest

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
)
from .test_helpers import (
    create_nontrivial_test_hamiltonian,
    create_test_basis_set,
)

OPENFERMION_AVAILABLE = importlib.util.find_spec("openfermion") is not None

if OPENFERMION_AVAILABLE:
    import openfermion as of

    from qdk_chemistry.algorithms import QubitMapper, available, create
    from qdk_chemistry.data import (
        CanonicalFourCenterHamiltonianContainer,
        Hamiltonian,
        MajoranaMapping,
        Orbitals,
        Symmetries,
    )
    from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
    from qdk_chemistry.plugins.openfermion.conversion import (
        hamiltonian_to_fermion_operator,
        hamiltonian_to_interaction_operator,
        qubit_hamiltonian_to_qubit_operator,
        qubit_operator_to_qubit_hamiltonian,
    )

    _ENCODING_TO_MAPPING = {
        "jordan-wigner": MajoranaMapping.jordan_wigner,
        "bravyi-kitaev": MajoranaMapping.bravyi_kitaev,
    }

if TYPE_CHECKING:
    from qdk_chemistry.data import QubitHamiltonian

pytestmark = pytest.mark.skipif(not OPENFERMION_AVAILABLE, reason="OpenFermion not available")


def _num_spin_orbitals(hamiltonian: Hamiltonian) -> int:
    """Return the number of spin-orbitals in *hamiltonian*."""
    return 2 * hamiltonian.get_one_body_integrals()[0].shape[0]


def _assert_pauli_ops_equal(actual: QubitHamiltonian, expected: QubitHamiltonian) -> None:
    """Assert two QubitHamiltonians have identical Pauli terms and coefficients."""
    assert actual.equiv(expected, atol=float_comparison_absolute_tolerance), (
        f"Pauli operators differ.\n  actual:   {actual.pauli_strings}\n  expected: {expected.pauli_strings}"
    )


# -------------------------------------------------------------------------------------
# Plugin registration
# -------------------------------------------------------------------------------------


def test_openfermion_plugin_registered():
    """Test that the OpenFermion qubit mapper is registered in the algorithm registry."""
    assert "openfermion" in available("qubit_mapper")


def test_openfermion_mapper_create():
    """Test creating an OpenFermionQubitMapper via the registry."""
    mapper = create("qubit_mapper", "openfermion")
    assert isinstance(mapper, QubitMapper)
    assert mapper.name() == "openfermion"


# -------------------------------------------------------------------------------------
# Standard encodings: verify exact match against QDK native (JW, BK)
# -------------------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ["jordan-wigner", "bravyi-kitaev"])
def test_openfermion_matches_qdk_native(encoding):
    """OpenFermion and QDK native mappers produce identical Pauli terms."""
    hamiltonian = create_nontrivial_test_hamiltonian()
    n = _num_spin_orbitals(hamiltonian)
    mapping = _ENCODING_TO_MAPPING[encoding](n)

    of_qh = create("qubit_mapper", "openfermion").run(hamiltonian, mapping)
    qdk_qh = create("qubit_mapper", "qdk").run(hamiltonian, mapping)

    _assert_pauli_ops_equal(of_qh, qdk_qh)


def test_openfermion_bk_tree_encoding():
    """BK-tree encoding produces the correct QubitHamiltonian (no QDK native equivalent).

    Builds an independent reference by re-indexing the interleaved
    InteractionOperator to blocked ordering, applying the BK-tree transform,
    and removing core energy.
    """
    hamiltonian = create_nontrivial_test_hamiltonian()
    n = _num_spin_orbitals(hamiltonian)
    mapping = MajoranaMapping(table=list(MajoranaMapping.jordan_wigner(n).table), name="bravyi-kitaev-tree")

    qh = create("qubit_mapper", "openfermion").run(hamiltonian, mapping)

    # Build reference independently: blocked InteractionOperator → FermionOp → BK-tree
    iop = hamiltonian_to_interaction_operator(hamiltonian)
    n_so = iop.n_qubits
    n_spatial = n_so // 2
    idx = np.array([2 * j if j < n_spatial else 2 * (j - n_spatial) + 1 for j in range(n_so)])
    h1_blocked = iop.one_body_tensor[np.ix_(idx, idx)]
    h2_blocked = iop.two_body_tensor[np.ix_(idx, idx, idx, idx)]
    iop_blocked = of.InteractionOperator(iop.constant, h1_blocked, h2_blocked)
    fop_blocked = of.transforms.get_fermion_operator(iop_blocked)

    ref_qop = of.transforms.bravyi_kitaev_tree(fop_blocked)
    ref_qop.compress()

    # Remove core energy
    core_energy = hamiltonian.get_core_energy()
    if abs(core_energy) > 1e-15:
        ref_qop -= core_energy * of.QubitOperator(())
        ref_qop.compress()

    ref_qh = qubit_operator_to_qubit_hamiltonian(ref_qop, encoding="bravyi-kitaev-tree")

    _assert_pauli_ops_equal(qh, ref_qh)


# -------------------------------------------------------------------------------------
# Symmetry-conserving Bravyi-Kitaev (SCBK)
# -------------------------------------------------------------------------------------


def _make_scbk_mapping(hamiltonian: Hamiltonian) -> MajoranaMapping:
    """Build a MajoranaMapping with name ``"symmetry-conserving-bravyi-kitaev"``.

    SCBK uses name-dispatch, not the Majorana table. Use a JW table as placeholder.
    """
    n_spin = _num_spin_orbitals(hamiltonian)
    jw = MajoranaMapping.jordan_wigner(num_modes=n_spin)
    return MajoranaMapping(table=list(jw.table), name="symmetry-conserving-bravyi-kitaev")


def test_openfermion_scbk_produces_qubit_hamiltonian():
    """SCBK encoding produces a valid QubitHamiltonian with reduced qubit count."""
    hamiltonian = create_nontrivial_test_hamiltonian()
    mapping = _make_scbk_mapping(hamiltonian)
    symmetries = Symmetries(n_alpha=1, n_beta=1)

    qh = create("qubit_mapper", "openfermion").run(hamiltonian, mapping, symmetries)

    assert qh is not None
    assert len(qh.pauli_strings) > 0
    # SCBK reduces qubit count by 2 relative to the number of spin-orbitals
    n_spin = _num_spin_orbitals(hamiltonian)
    assert qh.num_qubits == n_spin - 2


def test_openfermion_scbk_sets_interleaved_order():
    """SCBK encoding sets fermion_mode_order to INTERLEAVED."""
    hamiltonian = create_nontrivial_test_hamiltonian()
    mapping = _make_scbk_mapping(hamiltonian)
    symmetries = Symmetries(n_alpha=1, n_beta=1)

    qh = create("qubit_mapper", "openfermion").run(hamiltonian, mapping, symmetries)

    assert qh.fermion_mode_order == FermionModeOrder.INTERLEAVED


def test_openfermion_scbk_matches_direct_openfermion():
    """SCBK encoding matches direct use of openfermion.transforms.symmetry_conserving_bravyi_kitaev."""
    hamiltonian = create_nontrivial_test_hamiltonian()
    mapping = _make_scbk_mapping(hamiltonian)
    symmetries = Symmetries(n_alpha=1, n_beta=1)

    qh = create("qubit_mapper", "openfermion").run(hamiltonian, mapping, symmetries)

    # Build reference directly via OpenFermion
    fop = hamiltonian_to_fermion_operator(hamiltonian)
    h1_alpha, _ = hamiltonian.get_one_body_integrals()
    n_spinorbitals = 2 * h1_alpha.shape[0]
    n_active_electrons = symmetries.n_particles

    ref_qop = of.transforms.symmetry_conserving_bravyi_kitaev(fop, n_spinorbitals, n_active_electrons)
    ref_qop.compress()

    # Remove core energy (same as plugin does)
    core_energy = hamiltonian.get_core_energy()
    if abs(core_energy) > 1e-15:
        ref_qop -= core_energy * of.QubitOperator(())
        ref_qop.compress()

    ref_qh = qubit_operator_to_qubit_hamiltonian(
        ref_qop, encoding="symmetry-conserving-bravyi-kitaev", fermion_mode_order=FermionModeOrder.INTERLEAVED
    )

    _assert_pauli_ops_equal(qh, ref_qh)


def test_openfermion_scbk_requires_symmetries():
    """SCBK encoding raises ValueError when symmetries is not provided."""
    hamiltonian = create_nontrivial_test_hamiltonian()
    mapping = _make_scbk_mapping(hamiltonian)

    mapper = create("qubit_mapper", "openfermion")
    with pytest.raises(ValueError, match="symmetry-conserving Bravyi-Kitaev"):
        mapper.run(hamiltonian, mapping)


# -------------------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------------------


def test_openfermion_unsupported_mapping_raises():
    """A mapping with an unsupported name raises NotImplementedError."""
    hamiltonian = create_nontrivial_test_hamiltonian()
    n = _num_spin_orbitals(hamiltonian)
    mapping = MajoranaMapping(table=list(MajoranaMapping.jordan_wigner(n).table), name="invalid-encoding")

    mapper = create("qubit_mapper", "openfermion")
    with pytest.raises(NotImplementedError, match="invalid-encoding"):
        mapper.run(hamiltonian, mapping)


def test_symmetries_ignored_for_standard_encodings():
    """Symmetries parameter is accepted but ignored for standard encodings."""
    hamiltonian = create_nontrivial_test_hamiltonian()
    n = _num_spin_orbitals(hamiltonian)
    mapping = MajoranaMapping.jordan_wigner(n)
    symmetries = Symmetries(n_alpha=1, n_beta=1)

    # Should work without error - symmetries is simply ignored
    qh_with = create("qubit_mapper", "openfermion").run(hamiltonian, mapping, symmetries)
    qh_without = create("qubit_mapper", "openfermion").run(hamiltonian, mapping)

    _assert_pauli_ops_equal(qh_with, qh_without)


# -------------------------------------------------------------------------------------
# Conversion utilities: Hamiltonian → OpenFermion
# -------------------------------------------------------------------------------------


def test_hamiltonian_to_interaction_operator():
    """InteractionOperator has correct spin-orbital integrals and core energy."""
    hamiltonian = create_nontrivial_test_hamiltonian()
    h1_alpha, _ = hamiltonian.get_one_body_integrals()
    core_energy = hamiltonian.get_core_energy()

    iop = hamiltonian_to_interaction_operator(hamiltonian)

    assert iop.n_qubits == 2 * h1_alpha.shape[0]
    assert np.isclose(iop.constant, core_energy, atol=float_comparison_absolute_tolerance)

    # Spin-orbital one-body: alpha block on even indices must reproduce h1_alpha
    norb = h1_alpha.shape[0]
    h1_so_alpha = iop.one_body_tensor[::2, ::2]
    np.testing.assert_allclose(h1_so_alpha, h1_alpha, atol=float_comparison_absolute_tolerance)

    # Off-diagonal spin blocks must be zero (no spin-orbit coupling)
    h1_so_ab = iop.one_body_tensor[::2, 1::2]
    np.testing.assert_allclose(h1_so_ab, np.zeros((norb, norb)), atol=float_comparison_absolute_tolerance)


def test_hamiltonian_to_fermion_operator():
    """FermionOperator matrix matches InteractionOperator matrix."""
    hamiltonian = create_nontrivial_test_hamiltonian()

    iop = hamiltonian_to_interaction_operator(hamiltonian)
    fop = hamiltonian_to_fermion_operator(hamiltonian)

    iop_mat = of.linalg.get_sparse_operator(iop).toarray()
    fop_mat = of.linalg.get_sparse_operator(fop).toarray()

    np.testing.assert_allclose(fop_mat, iop_mat, atol=float_comparison_absolute_tolerance)


# -------------------------------------------------------------------------------------
# QubitOperator ↔ QubitHamiltonian round-trip
# -------------------------------------------------------------------------------------


def test_qubit_operator_round_trip():
    """QubitOperator → QubitHamiltonian → QubitOperator preserves terms."""
    original = of.QubitOperator("X0 Z1", 0.5) + of.QubitOperator("Y0 Y1", 0.3) + of.QubitOperator("", 1.0)

    qh = qubit_operator_to_qubit_hamiltonian(original, encoding="jordan-wigner")
    recovered = qubit_hamiltonian_to_qubit_operator(qh)

    n_qubits = 2
    np.testing.assert_allclose(
        of.linalg.get_sparse_operator(recovered, n_qubits=n_qubits).toarray(),
        of.linalg.get_sparse_operator(original, n_qubits=n_qubits).toarray(),
        atol=float_comparison_absolute_tolerance,
    )


def test_qubit_operator_to_qubit_hamiltonian_empty():
    """Converting an empty QubitOperator raises ValueError."""
    empty_op = of.QubitOperator()
    with pytest.raises(ValueError, match="empty"):
        qubit_operator_to_qubit_hamiltonian(empty_op)


def test_qubit_operator_to_qubit_hamiltonian_near_zero_raises():
    """A QubitOperator whose coefficients are all below machine epsilon raises ValueError."""
    eps = np.finfo(np.float64).eps
    near_zero_op = of.QubitOperator("X0 Z1", eps * 0.1) + of.QubitOperator("Y0", eps * 0.5)
    with pytest.raises(ValueError, match="empty"):
        qubit_operator_to_qubit_hamiltonian(near_zero_op)


def test_qubit_operator_to_qubit_hamiltonian_cancelled_raises():
    """A QubitOperator that cancels to zero via subtraction raises ValueError."""
    cancelled_op = of.QubitOperator("X0", 1.0) - of.QubitOperator("X0", 1.0)
    with pytest.raises(ValueError, match="empty"):
        qubit_operator_to_qubit_hamiltonian(cancelled_op)


def test_qubit_operator_to_qubit_hamiltonian_zero_identity_raises():
    """A QubitOperator with only a zero-coefficient identity term raises ValueError."""
    zero_identity = of.QubitOperator("", 0.0)
    with pytest.raises(ValueError, match="empty"):
        qubit_operator_to_qubit_hamiltonian(zero_identity)


def test_full_pipeline_round_trip():
    """Hamiltonian → FermionOp → QubitOp → QubitHamiltonian → QubitOp preserves terms."""
    hamiltonian = create_nontrivial_test_hamiltonian()

    fop = hamiltonian_to_fermion_operator(hamiltonian)
    qop = of.transforms.jordan_wigner(fop)
    qop.compress()
    qh = qubit_operator_to_qubit_hamiltonian(qop, encoding="jordan-wigner")
    qop_back = qubit_hamiltonian_to_qubit_operator(qh)

    np.testing.assert_allclose(
        of.linalg.get_sparse_operator(qop_back).toarray(),
        of.linalg.get_sparse_operator(qop).toarray(),
        atol=float_comparison_absolute_tolerance,
    )


# -------------------------------------------------------------------------------------
# Fermion mode order metadata
# -------------------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ["jordan-wigner", "bravyi-kitaev", "bravyi-kitaev-tree"])
def test_openfermion_standard_sets_blocked_order(encoding):
    """Standard OpenFermion encodings set fermion_mode_order to BLOCKED."""
    hamiltonian = create_nontrivial_test_hamiltonian()
    n = _num_spin_orbitals(hamiltonian)
    if encoding in _ENCODING_TO_MAPPING:
        mapping = _ENCODING_TO_MAPPING[encoding](n)
    else:
        mapping = MajoranaMapping(table=list(MajoranaMapping.jordan_wigner(n).table), name=encoding)
    qh = create("qubit_mapper", "openfermion").run(hamiltonian, mapping)
    assert qh.fermion_mode_order == FermionModeOrder.BLOCKED


@pytest.mark.skipif(not OPENFERMION_AVAILABLE, reason="OpenFermion not available")
def test_openfermion_unrestricted_jw_matches_qdk():
    """OpenFermion plugin produces same UHF JW result as QDK engine."""
    n = 2
    rng = np.random.default_rng(77)
    coeffs_a = np.eye(n)
    coeffs_b = np.eye(n) + rng.standard_normal((n, n)) * 0.1
    basis = create_test_basis_set(n, "uhf-of-test")
    orbitals = Orbitals(coeffs_a, coeffs_b, None, None, None, basis)

    raw_a = rng.standard_normal((n, n)) * 0.3
    h1_a = (raw_a + raw_a.T) / 2 + np.diag([1.0, -0.5])
    raw_b = rng.standard_normal((n, n)) * 0.3
    h1_b = (raw_b + raw_b.T) / 2 + np.diag([0.8, -0.3])

    def sym_eri(n, rng):
        h2 = np.zeros((n, n, n, n))
        seen = set()
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        perms = frozenset(
                            {
                                (p, q, r, s),
                                (q, p, r, s),
                                (p, q, s, r),
                                (q, p, s, r),
                                (r, s, p, q),
                                (s, r, p, q),
                                (r, s, q, p),
                                (s, r, q, p),
                            }
                        )
                        c = min(perms)
                        if c in seen:
                            continue
                        seen.add(c)
                        v = rng.standard_normal() * 0.2
                        for a, b, c2, d in perms:
                            h2[a, b, c2, d] = v
        return h2

    eri_aa = sym_eri(n, rng)
    eri_ab = sym_eri(n, rng)
    eri_bb = sym_eri(n, rng)

    hamiltonian = Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(
            h1_a,
            h1_b,
            eri_aa.ravel(),
            eri_ab.ravel(),
            eri_bb.ravel(),
            orbitals,
            0.0,
            np.eye(0),
            np.eye(0),
        )
    )
    assert not hamiltonian.get_orbitals().is_restricted()

    mapping = MajoranaMapping.jordan_wigner(num_modes=4)
    qh_qdk = create("qubit_mapper", "qdk").run(hamiltonian, mapping)
    qh_of = create("qubit_mapper", "openfermion").run(hamiltonian, mapping)

    # OpenFermion folds core_energy into identity; QDK doesn't.
    # Core energy is 0.0 here, so no adjustment needed.
    d_qdk = dict(zip(qh_qdk.pauli_strings, qh_qdk.coefficients, strict=False))
    d_of = dict(zip(qh_of.pauli_strings, qh_of.coefficients, strict=False))
    all_keys = set(d_qdk) | set(d_of)
    max_diff = max(abs(d_qdk.get(k, 0) - d_of.get(k, 0)) for k in all_keys)
    assert max_diff < 1e-10, f"UHF JW max coefficient diff: {max_diff}"
