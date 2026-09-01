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

from .test_helpers import create_nontrivial_test_hamiltonian, create_test_orbitals


@pytest.fixture
def factorizer():
    return create("double_factorizer", "eigen_decomposition")


class TestDoubleFactorizer:
    def test_metadata(self, factorizer):
        assert isinstance(factorizer, DoubleFactorizer)
        assert factorizer.type_name() == "double_factorizer"
        assert factorizer.name() == "eigen_decomposition"

    def test_run_returns_an_exact_factorized_hamiltonian(self, factorizer):
        hamiltonian = create_nontrivial_test_hamiltonian(4)

        factorized = factorizer.run(hamiltonian)
        container = factorized.get_container()

        assert isinstance(factorized, Hamiltonian)
        assert isinstance(container, FactorizedHamiltonianContainer)
        assert set(np.unique(container.get_signs())) <= {-1.0, 1.0}
        np.testing.assert_allclose(
            factorized.get_two_body_integrals()[0],
            hamiltonian.get_two_body_integrals()[0],
            atol=1e-10,
        )

    def test_truncation_threshold_is_applied(self, factorizer):
        hamiltonian = create_nontrivial_test_hamiltonian(4)
        num_ranks_exact = factorizer.run(hamiltonian).get_container().get_num_ranks()

        truncated_factorizer = create("double_factorizer", "eigen_decomposition")
        truncated_factorizer.settings().set("truncation_threshold", 1e-1)
        num_ranks_truncated = truncated_factorizer.run(hamiltonian).get_container().get_num_ranks()

        assert 0 < num_ranks_truncated < num_ranks_exact

    def test_asymmetric_two_body_integrals_are_rejected(self, factorizer):
        n = 4
        hamiltonian = create_nontrivial_test_hamiltonian(n)
        h2 = hamiltonian.get_two_body_integrals()[0].reshape(n, n, n, n).copy()

        # Break p<->q while leaving the (pq)<->(rs) supermatrix symmetric, so
        # only the second symmetry generator can catch this.
        h2[0, 1, 2, 2] += 1.0
        h2[2, 2, 0, 1] += 1.0
        asymmetric = Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
                hamiltonian.get_one_body_integrals()[0],
                h2.ravel(),
                create_test_orbitals(n),
                0.5,
                np.eye(0),
            )
        )

        with pytest.raises(ValueError, match="not symmetric"):
            factorizer.run(asymmetric)

        # Loosening the tolerance past the perturbation accepts it again, so
        # the rejection comes from the symmetry check and not another guard.
        tolerant = create("double_factorizer", "eigen_decomposition")
        tolerant.settings().set("symmetry_tolerance", 1e3)
        assert tolerant.run(asymmetric).get_container().get_num_ranks() > 0
