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
    HamiltonianType,
    SettingNotFoundError,
)

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
)
from .test_helpers import create_h2_molecule, create_test_orbitals


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
        """The registry exposes DF without a private first-stage truncation setting."""
        assert isinstance(factorizer, DoubleFactorization)
        assert isinstance(factorizer, HamiltonianFactorization)
        assert factorizer.type_name() == "hamiltonian_factorization"
        assert factorizer.name() == "double_factorization"
        assert factorizer.name() in factorizer.aliases()
        assert not factorizer.settings().has("truncation_threshold")
        with pytest.raises(SettingNotFoundError, match="truncation_threshold"):
            factorizer.settings().set("truncation_threshold", 1e-8)
        with pytest.raises(SettingNotFoundError, match="truncation_threshold"):
            create("hamiltonian_factorization", "double_factorization", truncation_threshold=1e-8)

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

    def test_preserves_redundant_custom_factors_and_input_data(self, factorizer: DoubleFactorization) -> None:
        """Redundant custom input remains two groups with all supplied metadata intact."""
        vectors = np.array([[0.7, 1.4], [0.0, 0.0], [0.0, 0.0], [0.2, 0.4]])
        one_body = np.array([[-2.0, 0.3], [0.3, 1.5]])
        inactive_fock = np.array([[0.5, 0.1], [0.2, 0.6]])
        orbitals = create_test_orbitals(2)
        hamiltonian = Hamiltonian(
            CholeskyHamiltonianContainer(
                one_body, vectors, orbitals, 0.75, inactive_fock, type=HamiltonianType.NonHermitian
            )
        )
        input_hash = hamiltonian.content_hash()

        factorized = factorizer.run(hamiltonian)
        container = factorized.get_container()
        assert container.get_num_ranks() == 2
        assert container.get_num_bases() == 2
        assert container.get_num_copies() == 1
        rotations = container.get_u_matrices().reshape(2, 2, 2)
        weights = container.get_w_matrices().reshape(2, 2)
        reconstructed = np.einsum("rbp,rb,rbq->rpq", rotations, weights, rotations)
        np.testing.assert_allclose(reconstructed.reshape(2, 4).T, vectors, atol=float_comparison_absolute_tolerance)
        np.testing.assert_allclose(
            factorized.get_two_body_integrals()[0],
            (vectors @ vectors.T).ravel(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        for output in (hamiltonian, factorized):
            np.testing.assert_array_equal(output.get_one_body_integrals()[0], one_body)
            np.testing.assert_array_equal(output.get_inactive_fock_matrix()[0], inactive_fock)
            assert output.get_core_energy() == 0.75
            assert output.get_orbitals() is orbitals
            assert output.get_type() == HamiltonianType.NonHermitian
        np.testing.assert_array_equal(hamiltonian.get_container().get_three_center_integrals()[0], vectors)
        assert hamiltonian.content_hash() == input_hash

    def test_rejects_canonical_four_center_input(self, factorizer: DoubleFactorization) -> None:
        """DF must not privately refactorize a dense two-electron tensor."""
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
        assert canonical.is_restricted()
        with pytest.raises(ValueError, match="requires a CholeskyHamiltonianContainer"):
            factorizer.run(canonical)

    @pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_factors(self, factorizer: DoubleFactorization, non_finite: float) -> None:
        """Non-finite factors fail before reaching the eigensolver."""
        vectors = np.array([[non_finite], [0.0], [0.0], [0.2]])
        hamiltonian = Hamiltonian(
            CholeskyHamiltonianContainer(np.eye(2), vectors, create_test_orbitals(2), 0.0, np.eye(0))
        )
        with pytest.raises(ValueError, match="non-finite"):
            factorizer.run(hamiltonian)

    def test_rejects_asymmetric_factors(self, factorizer: DoubleFactorization) -> None:
        """An incompatible factor is rejected rather than silently symmetrized."""
        vectors = np.array([[0.7], [0.1], [0.0], [0.2]])
        hamiltonian = Hamiltonian(
            CholeskyHamiltonianContainer(np.eye(2), vectors, create_test_orbitals(2), 0.0, np.eye(0))
        )
        with pytest.raises(ValueError, match="not symmetric in its orbital pair"):
            factorizer.run(hamiltonian)

    def test_consumes_configured_cholesky_producer(self, factorizer: DoubleFactorization) -> None:
        """First-stage accuracy stays with the configured producer, not DF."""
        _, wavefunction = create("scf_solver", "qdk", method="hf").run(create_h2_molecule(), 0, 1, "sto-3g")
        producer = create(
            "hamiltonian_constructor",
            "qdk_cholesky",
            cholesky_tolerance=1e-5,
            eri_threshold=1e-11,
            store_ao_cholesky_vectors=True,
            cholesky_gemm_batch_cols=2,
        )
        assert producer.settings().get("cholesky_tolerance") == 1e-5
        assert producer.settings().get("eri_threshold") == 1e-11
        hamiltonian = producer.run(wavefunction.get_orbitals())
        vectors = hamiltonian.get_container().get_three_center_integrals()[0].copy()
        input_hash = hamiltonian.content_hash()

        factorized = factorizer.run(hamiltonian)
        assert factorized.get_container().get_num_ranks() == vectors.shape[1]
        np.testing.assert_allclose(
            factorized.get_two_body_integrals()[0],
            (vectors @ vectors.T).ravel(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        np.testing.assert_array_equal(hamiltonian.get_container().get_three_center_integrals()[0], vectors)
        np.testing.assert_array_equal(factorized.get_one_body_integrals(), hamiltonian.get_one_body_integrals())
        assert factorized.has_inactive_fock_matrix() == hamiltonian.has_inactive_fock_matrix()
        assert factorized.get_orbitals() is hamiltonian.get_orbitals()
        assert factorized.get_core_energy() == hamiltonian.get_core_energy()
        assert factorized.get_type() == hamiltonian.get_type()
        assert hamiltonian.content_hash() == input_hash
