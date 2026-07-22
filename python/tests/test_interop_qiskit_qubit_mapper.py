"""Test Qiskit Qubit Mapper functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT_NATURE

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
)
from .test_helpers import create_test_basis_set, create_test_hamiltonian

if QDK_CHEMISTRY_HAS_QISKIT_NATURE:
    from qdk_chemistry.algorithms import QubitMapper, available, create
    from qdk_chemistry.data import (
        CanonicalFourCenterHamiltonianContainer,
        Hamiltonian,
        MajoranaMapping,
        Orbitals,
        QubitOperator,
    )

pytestmark = pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT_NATURE, reason="Qiskit Nature not available")


@pytest.mark.parametrize("encoding", ["jordan-wigner", "bravyi-kitaev", "parity"])
def test_qiskit_qubit_mappers(encoding) -> None:
    """Basic test for mapping a Hamiltonian to a qubit operator using Qiskit."""
    assert "qiskit" in available("qubit_mapper")
    qubit_mapper = create("qubit_mapper", "qiskit")
    assert isinstance(qubit_mapper, QubitMapper)

    hamiltonian = create_test_hamiltonian(2)
    assert isinstance(hamiltonian, Hamiltonian)
    n_modes = 2 * 2  # 2 spatial orbitals → 4 spin-orbitals
    factory = {
        "jordan-wigner": MajoranaMapping.jordan_wigner,
        "bravyi-kitaev": MajoranaMapping.bravyi_kitaev,
        "parity": MajoranaMapping.parity,
    }[encoding]
    mapping = factory(n_modes)
    qubit_hamiltonian = qubit_mapper.run(hamiltonian, mapping)
    assert isinstance(qubit_hamiltonian, QubitOperator)
    assert qubit_hamiltonian.num_qubits == 4
    assert isinstance(qubit_hamiltonian.pauli_strings, list)
    assert (
        qubit_hamiltonian.pauli_strings
        == {
            "jordan-wigner": ["IIII", "IIIZ", "IIZI", "IZII", "ZIII"],
            "bravyi-kitaev": ["IIII", "IIIZ", "IIZZ", "IZII", "ZZZI"],
            "parity": ["IIII", "IIIZ", "IIZZ", "IZZI", "ZZII"],
        }[encoding]
    )
    assert isinstance(qubit_hamiltonian.coefficients, np.ndarray)
    assert np.allclose(
        qubit_hamiltonian.coefficients,
        np.array([2.0, -0.5, -0.5, -0.5, -0.5]),
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )


def test_qiskit_unrestricted_jw_matches_qdk():
    """Qiskit plugin produces same UHF JW result as QDK engine for qubit operators."""
    n = 2
    rng = np.random.default_rng(77)
    coeffs_a = np.eye(n)
    coeffs_b = np.eye(n) + rng.standard_normal((n, n)) * 0.1
    basis = create_test_basis_set(n, "uhf-qiskit-test")
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
    qh_qis = create("qubit_mapper", "qiskit").run(hamiltonian, mapping)

    # Coefficient-exact comparison
    d_qdk = dict(zip(qh_qdk.pauli_strings, qh_qdk.coefficients, strict=False))
    d_qis = dict(zip(qh_qis.pauli_strings, qh_qis.coefficients, strict=False))
    all_keys = set(d_qdk) | set(d_qis)
    max_diff = max(abs(d_qdk.get(k, 0) - d_qis.get(k, 0)) for k in all_keys)
    assert max_diff < 1e-10, f"UHF JW max coefficient diff: {max_diff}"
