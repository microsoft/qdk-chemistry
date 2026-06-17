"""Tests for the Zassenhaus time-evolution builder."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from fractions import Fraction
from functools import cache

import numpy as np
import pytest
from scipy.linalg import expm

import qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus as zassenhaus_module
from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus import Zassenhaus
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    FlatPartition,
    MajoranaMapping,
    QubitHamiltonian,
    Structure,
    UnitaryRepresentation,
)
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
from qdk_chemistry.utils.phase import qpe_evolution_time_from_hamiltonian, resolve_energy_aliases


def _pauli_label(num_qubits: int, ops: dict[int, str]) -> str:
    chars = ["I"] * num_qubits
    for qubit, op in ops.items():
        chars[num_qubits - 1 - qubit] = op
    return "".join(chars)


def _heisenberg_chain(num_sites: int = 4) -> QubitHamiltonian:
    labels: list[str] = []
    coefficients: list[float] = []
    for site in range(num_sites - 1):
        for axis in ("X", "Y", "Z"):
            labels.append(_pauli_label(num_sites, {site: axis, site + 1: axis}))
            coefficients.append(1.0)
    return QubitHamiltonian(labels, np.asarray(coefficients))


@cache
def _h2_sto3g_hamiltonians():
    structure = Structure(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]), symbols=["H", "H"])
    _, wavefunction = registry.create("scf_solver").run(
        structure,
        charge=0,
        spin_multiplicity=1,
        basis_or_guess="sto-3g",
    )
    hamiltonian = registry.create("hamiltonian_constructor").run(wavefunction.get_orbitals())
    n_spin_orbitals = 2 * hamiltonian.get_orbitals().get_num_molecular_orbitals()
    qubit_hamiltonian = registry.create("qubit_mapper").run(
        hamiltonian,
        MajoranaMapping.jordan_wigner(n_spin_orbitals),
    )
    return hamiltonian, qubit_hamiltonian


@cache
def _h2_sto3g_qubit_hamiltonian() -> QubitHamiltonian:
    _, qubit_hamiltonian = _h2_sto3g_hamiltonians()
    return qubit_hamiltonian


@cache
def _h2_sto3g_phase_estimation_problem() -> tuple[QubitHamiltonian, Circuit, float]:
    hamiltonian, qubit_hamiltonian = _h2_sto3g_hamiltonians()
    _, wavefunction = registry.create("multi_configuration_calculator").run(hamiltonian, 1, 1)
    state_preparation = registry.create("state_prep", "sparse_isometry_gf2x").run(wavefunction)
    reference_energy = float(np.linalg.eigvalsh(qubit_hamiltonian.to_matrix()).min())
    return qubit_hamiltonian, state_preparation, reference_energy


def _container_to_unitary(container: PauliProductFormulaContainer) -> np.ndarray:
    num_qubits = container.num_qubits
    unitary = np.eye(2**num_qubits, dtype=complex)
    for term in container.step_terms:
        label = ["I"] * num_qubits
        for qubit, op in term.pauli_term.items():
            label[num_qubits - 1 - qubit] = op
        pauli = pauli_to_dense_matrix(["".join(label)], np.array([1.0]))
        unitary = expm(-1j * term.angle * pauli) @ unitary
    return np.linalg.matrix_power(unitary, container.step_reps)


def _error_slope(hamiltonian: QubitHamiltonian, order: int) -> float:
    times = np.logspace(-3, -1, 7)
    fit_tail_points = 5
    exact_hamiltonian = hamiltonian.to_matrix()
    errors: list[float] = []
    for time in times:
        builder = Zassenhaus(order=order, time=float(time), num_divisions=1, weight_threshold=1e-14)
        approximate = _container_to_unitary(builder.run(hamiltonian).get_container())
        exact = expm(-1j * exact_hamiltonian * time)
        errors.append(float(np.linalg.norm(exact - approximate, ord=2)))

    errors_array = np.asarray(errors)
    if not np.all(np.isfinite(errors_array)) or not np.all(errors_array > 0.0):
        raise AssertionError(
            "All sampled Zassenhaus errors must be positive and finite for the log-log fit: "
            f"{np.array2string(errors_array, precision=3)}"
        )

    # The smallest H2/STO-3G order-4 samples have true fifth-order errors below
    # double-precision resolution. Fit the resolved asymptotic tail instead of
    # letting the numerical floor dominate the log-log regression.
    return float(np.polyfit(np.log(times[-fit_tail_points:]), np.log(errors_array[-fit_tail_points:]), 1)[0])


class TestZassenhaus:
    """Tests for the Zassenhaus builder."""

    def test_registered(self):
        """Test that the Zassenhaus builder is available through the registry."""
        assert "zassenhaus" in registry.available("hamiltonian_unitary_builder")

        builder = registry.create("hamiltonian_unitary_builder", "zassenhaus", order=2, time=0.1)
        assert builder.name() == "zassenhaus"

    def test_returns_pauli_product_formula_container(self):
        """Test the output shape required by downstream phase-estimation consumers."""
        hamiltonian = QubitHamiltonian(["X", "Z"], np.array([1.0, 0.5]))
        builder = Zassenhaus(order=2, time=0.2, num_divisions=3, power=2)

        unitary = builder.run(hamiltonian)
        container = unitary.get_container()

        assert isinstance(unitary, UnitaryRepresentation)
        assert isinstance(container, PauliProductFormulaContainer)
        assert container.num_qubits == 1
        assert container.step_reps == 6
        assert len(container.step_terms) > 0

    def test_order_one_delegates_to_trotter(self):
        """Order-one Zassenhaus is exactly the first-order Trotter product."""
        hamiltonian = QubitHamiltonian(["XI", "ZZ"], np.array([2.0, 1.0]))

        zassenhaus_container = Zassenhaus(order=1, time=0.2, num_divisions=4).run(hamiltonian).get_container()
        trotter_container = Trotter(order=1, time=0.2, num_divisions=4).run(hamiltonian).get_container()

        assert zassenhaus_container.step_reps == trotter_container.step_reps
        assert zassenhaus_container.step_terms == trotter_container.step_terms

    def test_target_accuracy_uses_commutator_trotter_bound(self):
        """Zassenhaus automatic divisions reuse Trotter's commutator bound."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])

        zassenhaus_container = Zassenhaus(order=2, target_accuracy=0.01, time=1.0).run(hamiltonian).get_container()
        trotter_container = Trotter(order=2, target_accuracy=0.01, time=1.0).run(hamiltonian).get_container()

        assert zassenhaus_container.step_reps == trotter_container.step_reps

    def test_target_accuracy_uses_naive_trotter_bound(self):
        """Zassenhaus automatic divisions reuse Trotter's naive bound."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])

        zassenhaus_container = (
            Zassenhaus(order=2, target_accuracy=0.01, error_bound="naive", time=1.0).run(hamiltonian).get_container()
        )
        trotter_container = (
            Trotter(order=2, target_accuracy=0.01, error_bound="naive", time=1.0).run(hamiltonian).get_container()
        )

        assert zassenhaus_container.step_reps == trotter_container.step_reps

    def test_target_accuracy_odd_order_uses_supported_trotter_error_order(self):
        """Odd Zassenhaus orders use the next lower supported Trotter error order."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])

        zassenhaus_container = Zassenhaus(order=3, target_accuracy=0.01, time=1.0).run(hamiltonian).get_container()
        trotter_container = Trotter(order=2, target_accuracy=0.01, time=1.0).run(hamiltonian).get_container()

        assert zassenhaus_container.step_reps == trotter_container.step_reps

    def test_target_accuracy_keeps_manual_num_divisions_as_lower_bound(self):
        """Manual num_divisions remains a lower bound when target_accuracy is set."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])

        container = Zassenhaus(order=2, num_divisions=10, target_accuracy=0.01, time=1.0).run(
            hamiltonian
        ).get_container()

        assert container.step_reps == 10

    def test_cached_word_factors_are_read_only(self):
        """Cached formal word factors should not be mutable through callers."""
        factors = zassenhaus_module._zassenhaus_word_factors(2, 3)
        original = tuple(dict(factor) for factor in factors)

        with pytest.raises(TypeError):
            factors[0][(0, 0)] = Fraction(999)

        assert tuple(dict(factor) for factor in zassenhaus_module._zassenhaus_word_factors(2, 3)) == original

    def test_vanishing_corrections_short_circuit_before_pauli_multiplication(self, monkeypatch):
        """Commuting correction blocks should be skipped before sparse Pauli multiplication."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "IX"],
            coefficients=np.array([1.0, 1.0]),
            term_partition=FlatPartition(strategy="manual", groups=((0,), (1,))),
        )

        class FailingAccumulator:
            @staticmethod
            def multiply_uncached(*_args, **_kwargs):
                raise AssertionError("vanishing commutators should be skipped before Pauli multiplication")

        monkeypatch.setattr(zassenhaus_module, "PauliTermAccumulator", FailingAccumulator)

        container = Zassenhaus(order=2, time=0.2).run(hamiltonian).get_container()

        assert len(container.step_terms) == 2

    def test_invalid_error_bound_raises(self):
        """Invalid Zassenhaus error_bound values are rejected by settings."""
        with pytest.raises(ValueError, match="allowed options"):
            Zassenhaus(error_bound="invalid")

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_x_z_error_scales_as_order_plus_one(self, order: int):
        """Test Zassenhaus order scaling on a noncommuting one-qubit Hamiltonian."""
        hamiltonian = QubitHamiltonian(["X", "Z"], np.array([1.0, 1.0]))
        assert _error_slope(hamiltonian, order) == pytest.approx(order + 1, abs=0.1)

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_heisenberg_chain_error_scales_as_order_plus_one(self, order: int):
        """Test order scaling on the 4-site open Heisenberg chain with J=1."""
        assert _error_slope(_heisenberg_chain(), order) == pytest.approx(order + 1, abs=0.1)

    @pytest.mark.slow
    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_h2_sto3g_error_scales_as_order_plus_one(self, order: int):
        """Test order scaling on H2/STO-3G mapped with Jordan-Wigner."""
        assert _error_slope(_h2_sto3g_qubit_hamiltonian(), order) == pytest.approx(order + 1, abs=0.1)

    @pytest.mark.slow
    def test_iterative_phase_estimation_h2_sto3g_chemical_accuracy(self):
        """Test Zassenhaus-backed PhaseEstimation on H2/STO-3G within chemical accuracy."""
        hamiltonian, state_preparation, reference_energy = _h2_sto3g_phase_estimation_problem()
        num_bits = 11
        evolution_time = qpe_evolution_time_from_hamiltonian(hamiltonian)
        zassenhaus_order = 3
        zassenhaus_num_divisions = 4

        first_iteration_power = 2 ** (num_bits - 1)
        first_iteration_container = Zassenhaus(
            order=zassenhaus_order,
            time=evolution_time,
            num_divisions=zassenhaus_num_divisions,
            power=first_iteration_power,
        ).run(hamiltonian).get_container()
        assert len(first_iteration_container.step_terms) * first_iteration_container.step_reps <= 200_000

        phase_estimation = registry.create("phase_estimation", "qdk_iterative", shots_per_bit=3)
        phase_estimation.settings().set(
            "qpe_circuit_builder",
            AlgorithmRef(
                "qpe_circuit_builder",
                "qdk_iterative",
                num_bits=num_bits,
                controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
                unitary_builder=AlgorithmRef(
                    "hamiltonian_unitary_builder",
                    "zassenhaus",
                    order=zassenhaus_order,
                    time=evolution_time,
                    num_divisions=zassenhaus_num_divisions,
                ),
            ),
        )
        phase_estimation.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42),
        )

        result = phase_estimation.run(
            qubit_hamiltonian=hamiltonian,
            state_preparation=state_preparation,
        )

        resolved_energy = resolve_energy_aliases(
            result.raw_energy,
            evolution_time=evolution_time,
            reference_energy=reference_energy,
            shift_range=range(-4, 5),
        )
        assert resolved_energy == pytest.approx(reference_energy, abs=1.6e-3)

    def test_rejects_non_hermitian_hamiltonian(self):
        """Test that non-Hermitian Hamiltonians are rejected."""
        hamiltonian = QubitHamiltonian(["X"], np.array([1.0 + 1.0j]))
        with pytest.raises(ValueError, match="Non-Hermitian"):
            Zassenhaus(order=2, time=0.1).run(hamiltonian)

    def test_rejects_non_positive_num_divisions(self):
        """Test validation of manual division count."""
        hamiltonian = QubitHamiltonian(["X"], np.array([1.0]))
        with pytest.raises(ValueError, match="num_divisions"):
            Zassenhaus(order=2, time=0.1, num_divisions=0).run(hamiltonian)
