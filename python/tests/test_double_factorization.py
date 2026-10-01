"""Tests for the double factorization algorithm bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import DoubleFactorization, HamiltonianFactorization, create
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    CholeskyHamiltonianContainer,
    DFTHCHamiltonianContainer,
    Hamiltonian,
)

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
)
from .test_helpers import create_test_orbitals


def create_cholesky_test_hamiltonian(num_orbitals: int = 4, num_factors: int = 3, seed: int = 17) -> Hamiltonian:
    """Build a Hamiltonian with supplied symmetric MO Cholesky factors."""
    n = num_orbitals
    rng = np.random.default_rng(seed)

    vectors = np.empty((n * n, num_factors))
    for k in range(num_factors):
        raw = rng.standard_normal((n, n)) * 0.3
        factor = (raw + raw.T) / 2
        vectors[:, k] = factor.ravel()

    raw = rng.standard_normal((n, n)) * 0.3
    one_body = (raw + raw.T) / 2 + np.diag(np.linspace(1.0, -0.5, n))

    return Hamiltonian(CholeskyHamiltonianContainer(one_body, vectors, create_test_orbitals(n), 0.5, np.eye(0)))


@pytest.fixture
def factorizer() -> DoubleFactorization:
    """Create the registered double factorizer."""
    return create("hamiltonian_factorization", "double_factorization")


class TestDoubleFactorization:
    """The factorizer consumes compatible Cholesky data without replacing it."""

    def test_metadata(self, factorizer: DoubleFactorization) -> None:
        """The registry exposes DF under its registered name."""
        assert isinstance(factorizer, DoubleFactorization)
        assert isinstance(factorizer, HamiltonianFactorization)
        assert factorizer.type_name() == "hamiltonian_factorization"
        assert factorizer.name() == "double_factorization"
        assert factorizer.name() in factorizer.aliases()

    def test_run_returns_an_exact_factorized_hamiltonian(self, factorizer: DoubleFactorization) -> None:
        """Each supplied column contributes its square in chemist pair order."""
        hamiltonian = create_cholesky_test_hamiltonian()
        vectors = hamiltonian.get_container().get_three_center_integrals()[0].copy()

        factorized = factorizer.run(hamiltonian)
        container = factorized.get_container()

        assert isinstance(factorized, Hamiltonian)
        assert isinstance(container, DFTHCHamiltonianContainer)
        np.testing.assert_allclose(
            factorized.get_two_body_integrals()[0],
            (vectors @ vectors.T).ravel(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    @pytest.mark.parametrize("num_factors", [1, 2, 3, 4])
    def test_rank_equals_the_number_of_supplied_factors(self, num_factors: int) -> None:
        """The output has one factor group per supplied auxiliary column."""
        hamiltonian = create_cholesky_test_hamiltonian(num_factors=num_factors)
        container = create("hamiltonian_factorization", "double_factorization").run(hamiltonian).get_container()
        assert container.get_num_ranks() == num_factors

    def test_accepts_four_center_input_after_cholesky_conversion(self, factorizer: DoubleFactorization) -> None:
        """A four-center Hamiltonian reaches DF once its two-electron tensor is converted to Cholesky factors."""
        source = create_cholesky_test_hamiltonian()
        canonical = Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
                source.get_one_body_integrals()[0],
                source.get_two_body_integrals()[0],
                source.get_orbitals(),
                source.get_core_energy(),
                np.eye(0),
            )
        )
        one_body = canonical.get_one_body_integrals()[0]
        two_body = canonical.get_two_body_integrals()[0]

        # The (pq|rs) supermatrix is positive semi-definite, so its eigenvectors with
        # non-zero eigenvalues give exact factors, each symmetric in its orbital pair.
        eigenvalues, eigenvectors = np.linalg.eigh(two_body.reshape(one_body.size, one_body.size))
        keep = eigenvalues > 1e-10 * eigenvalues.max()
        vectors = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
        cholesky = Hamiltonian(
            CholeskyHamiltonianContainer(
                one_body, vectors, canonical.get_orbitals(), canonical.get_core_energy(), np.eye(0)
            )
        )

        factorized = factorizer.run(cholesky)

        assert isinstance(factorized.get_container(), DFTHCHamiltonianContainer)
        assert factorized.get_container().get_num_ranks() == vectors.shape[1]
        np.testing.assert_allclose(
            factorized.get_two_body_integrals()[0],
            two_body,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
