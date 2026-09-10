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
)

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
)
from .test_helpers import create_test_orbitals


def create_positive_semidefinite_test_hamiltonian(num_orbitals=4, num_factors=3, seed=17):
    """Build a Hamiltonian whose two-electron supermatrix is positive semi-definite.

    ``g_pqrs = sum_k F_k[p,q] F_k[r,s]`` with symmetric ``F_k`` makes the
    supermatrix a Gram matrix, so a Cholesky decomposition exists and its rank
    is exactly ``num_factors``.
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

    @pytest.mark.parametrize("num_factors", [1, 2, 3, 4])
    def test_rank_equals_the_number_of_independent_factors(self, num_factors):
        hamiltonian = create_positive_semidefinite_test_hamiltonian(num_factors=num_factors)
        container = create("double_factorizer", "qdk").run(hamiltonian).get_container()
        assert container.get_num_ranks() == num_factors
