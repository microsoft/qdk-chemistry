"""Tests for the double factorization algorithm bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import DoubleFactorizer, create
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    FactorizedHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
)

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
)
from .test_helpers import create_nontrivial_test_hamiltonian, create_test_orbitals
from .test_qdk_qubit_mapper_factorized import _assert_term_by_term_equivalent


def create_positive_semidefinite_test_hamiltonian(num_orbitals=4, num_factors=3, seed=17):
    """Build a Hamiltonian whose two-electron supermatrix is positive semi-definite.

    ``g_pqrs = sum_k F_k[p,q] F_k[r,s]`` with symmetric ``F_k`` makes the
    supermatrix a Gram matrix, so a Cholesky decomposition exists and its rank
    is exactly ``num_factors``. :func:`create_nontrivial_test_hamiltonian`
    draws its unique elements independently instead, which is 8-fold symmetric
    but indefinite, so it cannot be double-factorized at all.
    """
    n = num_orbitals
    rng = np.random.default_rng(seed)

    two_body = np.zeros((n, n, n, n))
    for _ in range(num_factors):
        raw = rng.standard_normal((n, n)) * 0.3
        factor = (raw + raw.T) / 2
        two_body += np.einsum("pq,rs->pqrs", factor, factor)

    raw = rng.standard_normal((n, n)) * 0.3
    one_body = (raw + raw.T) / 2 + np.diag(np.linspace(1.0, -0.5, n))

    return Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(one_body, two_body.ravel(), create_test_orbitals(n), 0.5, np.eye(0))
    )


@pytest.fixture
def factorizer():
    return create("double_factorizer", "qdk")


class TestDoubleFactorizer:
    def test_metadata(self, factorizer):
        assert isinstance(factorizer, DoubleFactorizer)
        assert factorizer.type_name() == "double_factorizer"
        assert factorizer.name() == "qdk"
        assert factorizer.name() in factorizer.aliases()
        assert factorizer.settings().has("truncation_threshold")

    def test_run_returns_an_exact_factorized_hamiltonian(self, factorizer):
        hamiltonian = create_positive_semidefinite_test_hamiltonian()

        factorized = factorizer.run(hamiltonian)
        container = factorized.get_container()

        assert isinstance(factorized, Hamiltonian)
        assert isinstance(container, FactorizedHamiltonianContainer)
        np.testing.assert_allclose(
            factorized.get_two_body_integrals()[0],
            hamiltonian.get_two_body_integrals()[0],
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_rejects_an_indefinite_tensor(self, factorizer):
        # A Cholesky decomposition only exists for a positive semi-definite
        # supermatrix. Stopping at the breakdown would factorize a different
        # tensor and report a silently wrong lambda, so this is rejected.
        # create_nontrivial_test_hamiltonian draws its unique elements
        # independently, which is 8-fold symmetric but not positive
        # semi-definite, unlike genuine two-electron integrals.
        with pytest.raises(ValueError, match="not positive semi-definite"):
            factorizer.run(create_nontrivial_test_hamiltonian(4))

    @pytest.mark.parametrize("num_factors", [1, 2, 3, 4])
    def test_rank_equals_the_number_of_independent_factors(self, num_factors):
        hamiltonian = create_positive_semidefinite_test_hamiltonian(num_factors=num_factors)
        container = create("double_factorizer", "qdk").run(hamiltonian).get_container()
        assert container.get_num_ranks() == num_factors

    def test_truncation_threshold_is_applied(self):
        hamiltonian = create_positive_semidefinite_test_hamiltonian(num_factors=3)

        def num_ranks(threshold):
            truncated = create("double_factorizer", "qdk")
            truncated.settings().set("truncation_threshold", threshold)
            return truncated.run(hamiltonian).get_container().get_num_ranks()

        # The threshold is the pivoted-Cholesky stopping cutoff, so 0.0 means
        # "keep everything numerically resolvable" rather than "keep every
        # supermatrix eigenpair": the null space is never materialized.
        assert num_ranks(0.0) == 3

        # The three supermatrix eigenvalues are 2.28, 0.96 and 0.68, so a
        # cutoff of 0.5 has to leave exactly one fragment standing. Choosing a
        # threshold between two known eigenvalues is what makes this an
        # assertion about the cutoff rather than about a magnitude.
        assert num_ranks(0.5) == 1

        # Raising the threshold can only discard more fragments.
        assert num_ranks(0.5) <= num_ranks(1e-6) <= num_ranks(0.0)

    def test_rejects_a_threshold_that_discards_everything(self):
        hamiltonian = create_positive_semidefinite_test_hamiltonian(num_factors=3)
        truncating = create("double_factorizer", "qdk")
        truncating.settings().set("truncation_threshold", 1e6)
        with pytest.raises(ValueError, match="no two-body term"):
            truncating.run(hamiltonian)

    def test_factorizing_does_not_change_the_mapped_qubit_operator(self, factorizer):
        """End-to-end check through the downstream consumer of a factorization.

        Reconstructing the tensor is necessary but not sufficient: the qubit
        mapper reads one-body integrals, orbitals and core energy as well, so a
        factorization that silently dropped or rescaled any of them would still
        pass the reconstruction test. Mapping both Hamiltonians and comparing
        the operators term by term covers the whole payload.
        """
        norb = 4
        hamiltonian = create_positive_semidefinite_test_hamiltonian(norb)
        factorized = factorizer.run(hamiltonian)

        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * norb)
        mapper = create("qubit_mapper", "qdk")

        _assert_term_by_term_equivalent(mapper.run(hamiltonian, mapping), mapper.run(factorized, mapping))
