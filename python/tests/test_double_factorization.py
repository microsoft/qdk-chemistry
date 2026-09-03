"""Tests for the double factorization algorithm bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import DoubleFactorizer, create
from qdk_chemistry.data import (
    FactorizedHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
)

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
)
from .test_helpers import create_nontrivial_test_hamiltonian
from .test_qdk_qubit_mapper_factorized import _assert_term_by_term_equivalent


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
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_truncation_threshold_is_applied(self, factorizer):
        norb = 4
        hamiltonian = create_nontrivial_test_hamiltonian(norb)

        def num_ranks(threshold):
            truncated = create("double_factorizer", "eigen_decomposition")
            truncated.settings().set("truncation_threshold", threshold)
            return truncated.run(hamiltonian).get_container().get_num_ranks()

        # A threshold of 0.0 retains every supermatrix eigenpair, so the rank
        # equals the supermatrix dimension.
        assert num_ranks(0.0) == norb**2

        # The helper builds a tensor with full 8-fold symmetry, so the
        # supermatrix annihilates every antisymmetric pair vector. The default
        # threshold drops exactly those norb*(norb-1)/2 null fragments, leaving
        # the symmetric pair block. This holds whatever the integrals are, so
        # it does not depend on the helper's generated magnitudes.
        num_ranks_default = factorizer.run(hamiltonian).get_container().get_num_ranks()
        assert num_ranks_default == norb * (norb + 1) // 2

        # Raising the threshold can only discard more fragments.
        assert num_ranks(1e-1) <= num_ranks(1e-6) <= num_ranks_default

    def test_factorizing_does_not_change_the_mapped_qubit_operator(self, factorizer):
        """End-to-end check through the downstream consumer of a factorization.

        Reconstructing the tensor is necessary but not sufficient: the qubit
        mapper reads one-body integrals, orbitals and core energy as well, so a
        factorization that silently dropped or rescaled any of them would still
        pass the reconstruction test. Mapping both Hamiltonians and comparing
        the operators term by term covers the whole payload.
        """
        norb = 4
        hamiltonian = create_nontrivial_test_hamiltonian(norb)
        factorized = factorizer.run(hamiltonian)

        mapping = MajoranaMapping.jordan_wigner(num_modes=2 * norb)
        mapper = create("qubit_mapper", "qdk")

        _assert_term_by_term_equivalent(mapper.run(hamiltonian, mapping), mapper.run(factorized, mapping))
