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
from .test_helpers import create_nontrivial_test_hamiltonian

OPENFERMION_AVAILABLE = importlib.util.find_spec("openfermion") is not None

if OPENFERMION_AVAILABLE:
    import openfermion as of

    from qdk_chemistry.algorithms import QubitMapper, available, create
    from qdk_chemistry.plugins.openfermion.conversion import (
        hamiltonian_to_fermion_operator,
        hamiltonian_to_interaction_operator,
        qubit_hamiltonian_to_qubit_operator,
        qubit_operator_to_qubit_hamiltonian,
    )

if TYPE_CHECKING:
    from qdk_chemistry.data import QubitHamiltonian

pytestmark = pytest.mark.skipif(not OPENFERMION_AVAILABLE, reason="OpenFermion not available")


def _assert_pauli_ops_equal(actual: QubitHamiltonian, expected: QubitHamiltonian) -> None:
    """Assert two QubitHamiltonians have identical Pauli terms and coefficients."""
    assert actual.pauli_ops.equiv(expected.pauli_ops, atol=float_comparison_absolute_tolerance), (
        f"Pauli operators differ.\n  actual:   {actual.pauli_ops.sort()}\n  expected: {expected.pauli_ops.sort()}"
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
    assert mapper.settings().get("encoding") == "jordan-wigner"


# -------------------------------------------------------------------------------------
# Standard encodings: verify exact match against QDK native (JW, BK)
# -------------------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ["jordan-wigner", "bravyi-kitaev"])
def test_openfermion_matches_qdk_native(encoding):
    """OpenFermion and QDK native mappers produce identical Pauli terms."""
    hamiltonian = create_nontrivial_test_hamiltonian()

    of_qh = create("qubit_mapper", "openfermion", encoding=encoding).run(hamiltonian)
    qdk_qh = create("qubit_mapper", "qdk", encoding=encoding).run(hamiltonian)

    _assert_pauli_ops_equal(of_qh, qdk_qh)


def test_openfermion_bk_tree_encoding():
    """BK-tree encoding produces the correct QubitHamiltonian (no QDK native equivalent).

    Builds an independent reference by re-indexing the interleaved
    InteractionOperator to blocked ordering, applying the BK-tree transform,
    and removing core energy.
    """
    hamiltonian = create_nontrivial_test_hamiltonian()

    qh = create("qubit_mapper", "openfermion", encoding="bravyi-kitaev-tree").run(hamiltonian)

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
# Non-standard encodings (BKSF, SCBK) — OpenFermion-specific, no QDK native equivalent
# -------------------------------------------------------------------------------------


def test_openfermion_bksf_encoding():
    """BKSF encoding produces a QubitHamiltonian with correct Pauli terms."""
    hamiltonian = create_nontrivial_test_hamiltonian()

    mapper = create("qubit_mapper", "openfermion", encoding="bravyi-kitaev-fast")
    qh = mapper.run(hamiltonian)

    # Reference: apply BKSF directly to the InteractionOperator
    iop = hamiltonian_to_interaction_operator(hamiltonian)
    ref_qop = of.transforms.bravyi_kitaev_fast(iop)
    ref_qop.compress()

    # Remove core energy from reference (same convention as plugin)
    core_energy = hamiltonian.get_core_energy()
    if abs(core_energy) > 1e-15:
        ref_qop -= core_energy * of.QubitOperator(())
        ref_qop.compress()

    ref_qh = qubit_operator_to_qubit_hamiltonian(ref_qop, encoding="bravyi-kitaev-fast")

    _assert_pauli_ops_equal(qh, ref_qh)


def test_openfermion_scbk_encoding():
    """SCBK encoding produces a QubitHamiltonian with correct Pauli terms."""
    hamiltonian = create_nontrivial_test_hamiltonian()

    mapper = create(
        "qubit_mapper",
        "openfermion",
        encoding="symmetry-conserving-bravyi-kitaev",
        n_active_electrons=2,
    )
    qh = mapper.run(hamiltonian)

    # Reference: apply SCBK directly
    fop = hamiltonian_to_fermion_operator(hamiltonian)
    ref_qop = of.transforms.symmetry_conserving_bravyi_kitaev(fop, 4, 2)
    ref_qop.compress()

    # Remove core energy from reference (same convention as plugin)
    core_energy = hamiltonian.get_core_energy()
    if abs(core_energy) > 1e-15:
        ref_qop -= core_energy * of.QubitOperator(())
        ref_qop.compress()

    ref_qh = qubit_operator_to_qubit_hamiltonian(ref_qop, encoding="symmetry-conserving-bravyi-kitaev")

    _assert_pauli_ops_equal(qh, ref_qh)


def test_openfermion_invalid_encoding():
    """An invalid encoding raises ValueError."""
    mapper = create("qubit_mapper", "openfermion")
    with pytest.raises(ValueError, match="out of allowed options"):
        mapper.settings().set("encoding", "invalid-encoding")


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
