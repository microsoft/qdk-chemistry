"""Tests for the Zassenhaus product-formula builder in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest
import scipy

from qdk_chemistry.algorithms import registry
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution import zassenhaus as zassenhaus_module
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus import Zassenhaus
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    LatticeGraph,
    QubitHamiltonian,
    UnitaryRepresentation,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.utils.model_hamiltonians import create_heisenberg_hamiltonian
from qdk_chemistry.utils.phase import energy_from_phase
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

# H2 / STO-3G qubit Hamiltonian (4-qubit Jordan-Wigner, equilibrium geometry).
# Coefficients follow the standard ordering used throughout the literature
# (e.g. Seeley, Richard, Love 2012; O'Malley et al. 2016).
_H2_LABELS = [
    "IIII",
    "IIIZ",
    "IIZI",
    "IZII",
    "ZIII",
    "IIZZ",
    "IZIZ",
    "ZIIZ",
    "IZZI",
    "ZIZI",
    "ZZII",
    "XXYY",
    "YYXX",
    "XYYX",
    "YXXY",
]
_H2_COEFFS = np.array(
    [
        -0.81261,
        0.171201,
        0.171201,
        -0.2227965,
        -0.2227965,
        0.16862325,
        0.12054625,
        0.165868,
        0.165868,
        0.12054625,
        0.17434925,
        -0.04532175,
        -0.04532175,
        0.04532175,
        0.04532175,
    ]
)


def _h2_hamiltonian() -> QubitHamiltonian:
    """Return the 4-qubit H2/STO-3G Jordan-Wigner qubit Hamiltonian."""
    return QubitHamiltonian(pauli_strings=list(_H2_LABELS), coefficients=_H2_COEFFS.copy())


def _heisenberg_hamiltonian() -> QubitHamiltonian:
    """Return the 4-site open Heisenberg chain Hamiltonian with J = 1."""
    chain = LatticeGraph.chain(4, periodic=False)
    return create_heisenberg_hamiltonian(chain, jx=1.0, jy=1.0, jz=1.0)


def _builder_unitary(container: PauliProductFormulaContainer) -> np.ndarray:
    """Reconstruct the dense unitary realised by a Pauli-product-formula container.

    The Pauli sequence mapper applies ``step_terms[0]`` first (rightmost in the
    operator product) and repeats the whole step ``step_reps`` times. Identity
    terms (empty Pauli map) contribute a global phase.
    """
    num_qubits = container.num_qubits
    dim = 2**num_qubits
    step = np.eye(dim, dtype=complex)
    for term in container.step_terms:
        chars = ["I"] * num_qubits
        for qubit, axis in term.pauli_term.items():
            chars[num_qubits - 1 - qubit] = axis  # little-endian: qubit 0 -> rightmost
        pauli = QubitHamiltonian([("".join(chars))], np.array([1.0])).to_matrix()
        step = scipy.linalg.expm(-1j * term.angle * pauli) @ step
    return np.linalg.matrix_power(step, container.step_reps)


def _zassenhaus_unitary(hamiltonian: QubitHamiltonian, time: float, order: int) -> np.ndarray:
    """Build the Zassenhaus approximation to ``exp(-i H t)`` as a dense matrix."""
    builder = Zassenhaus(order=order, time=time, num_divisions=1)
    return _builder_unitary(builder.run(hamiltonian).get_container())


def _exact_unitary(hamiltonian: QubitHamiltonian, time: float) -> np.ndarray:
    """Return the exact time-evolution operator ``exp(-i H t)``."""
    return scipy.linalg.expm(-1j * time * hamiltonian.to_matrix())


def _fitted_error_slope(hamiltonian: QubitHamiltonian, order: int) -> float:
    """Fit the slope of ``log(error)`` vs ``log(t)`` for the operator-norm error.

    The acceptance criterion specifies ``t in [1e-3, 1e-1]``. The fit is taken
    over the upper decade ``[1e-2, 1e-1]``: there every order is comfortably
    above the ~1e-13 double-precision floor, whereas an order-4 (O(t^5)) error
    near ``t = 1e-3`` is ~1e-13 and would be pure round-off noise.
    """
    times = np.geomspace(1e-2, 1e-1, 7)
    errors = np.array(
        [np.linalg.norm(_exact_unitary(hamiltonian, t) - _zassenhaus_unitary(hamiltonian, t, order), 2) for t in times]
    )
    return float(np.polyfit(np.log(times), np.log(errors), 1)[0])


class TestZassenhausBuilder:
    """Structural and registry tests for the Zassenhaus builder."""

    def test_name(self):
        """The builder reports its registry name and algorithm type."""
        builder = Zassenhaus()
        assert builder.name() == "zassenhaus"
        assert builder.type_name() == "hamiltonian_unitary_builder"

    def test_discoverable_via_registry(self):
        """The builder is discoverable under the hamiltonian_unitary_builder type."""
        builder = registry.create("hamiltonian_unitary_builder", "zassenhaus", time=0.1, order=2)
        assert isinstance(builder, Zassenhaus)
        assert "zassenhaus" in registry.available("hamiltonian_unitary_builder")

    def test_returns_pauli_product_formula_container(self):
        """The output has the same shape as the trotter builder's output."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZI", "IZ"], coefficients=[0.7, 0.3, 0.5])
        unitary = Zassenhaus(order=2, time=0.1, num_divisions=1).run(hamiltonian)

        assert isinstance(unitary, UnitaryRepresentation)
        container = unitary.get_container()
        assert isinstance(container, PauliProductFormulaContainer)
        assert container.num_qubits == 2
        assert container.step_reps == 1
        assert len(container.step_terms) > 0

    def test_num_divisions_sets_step_reps(self):
        """num_divisions controls the repetition count of the product formula."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZI", "IZ"], coefficients=[0.7, 0.3, 0.5])
        container = Zassenhaus(order=2, time=0.2, num_divisions=4).run(hamiltonian).get_container()
        assert container.step_reps == 4

    def test_large_generator_warns(self, monkeypatch):
        """The symbolic generator warns (not silently hangs) when ~K^order is huge."""
        messages: list[str] = []

        class _RecordingLogger:
            @staticmethod
            def warn(message: str) -> None:
                messages.append(message)

        monkeypatch.setattr(zassenhaus_module, "Logger", _RecordingLogger)
        zassenhaus_module._warn_if_series_large(20, 7)  # ~1.3e9 words -> over threshold
        zassenhaus_module._warn_if_series_large(3, 7)  # 2187 words (e.g. Heisenberg) -> under threshold
        assert len(messages) == 1
        assert "may be slow" in messages[0]

    def test_num_divisions_below_one_raises(self):
        """num_divisions < 1 is rejected at construction, not silently clamped."""
        with pytest.raises(ValueError, match="num_divisions"):
            Zassenhaus(order=2, time=0.1, num_divisions=0)

    @pytest.mark.parametrize("order", [2, 3])
    def test_target_accuracy_increases_divisions(self, order):
        """target_accuracy raises N until the error meets the target (even and odd orders).

        The shared ``commutator_bound_higher_order`` estimate is applied directly, so
        it also covers odd orders that the ``trotter_steps_*`` wrappers reject.
        """
        hamiltonian = _heisenberg_hamiltonian()
        time = 1.0
        target = 1e-2
        container = Zassenhaus(order=order, time=time, target_accuracy=target).run(hamiltonian).get_container()
        assert container.step_reps > 1  # auto-estimation engaged
        error = np.linalg.norm(_exact_unitary(hamiltonian, time) - _builder_unitary(container), 2)
        assert error < target

    def test_accepts_partitioned_hamiltonian(self):
        """A Hamiltonian carrying a commuting term_partition is accepted (as trotter accepts)."""
        hamiltonian = _heisenberg_hamiltonian()
        partitioned = registry.create("term_grouper", "commuting").run(hamiltonian)
        unitary = Zassenhaus(order=2, time=0.05, num_divisions=1).run(partitioned)
        assert isinstance(unitary.get_container(), PauliProductFormulaContainer)

    def test_invalid_order_raises(self):
        """Orders below 1 are rejected (1 = Trotter fallback, >= 2 = Zassenhaus)."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZI"], coefficients=[0.7, 0.5])
        with pytest.raises(ValueError, match="order"):
            Zassenhaus(order=0, time=0.1).run(hamiltonian)

    def test_order_one_falls_back_to_trotter(self):
        """Order 1 (no commutator corrections) reproduces the first-order Trotter product."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZI", "IZ"], coefficients=[0.7, 0.3, 0.5])
        zassenhaus = Zassenhaus(order=1, time=0.2, num_divisions=2).run(hamiltonian).get_container()
        trotter = (
            registry.create("hamiltonian_unitary_builder", "trotter", order=1, time=0.2, num_divisions=2)
            .run(hamiltonian)
            .get_container()
        )

        assert zassenhaus.step_reps == trotter.step_reps
        assert [(t.pauli_term, t.angle) for t in zassenhaus.step_terms] == [
            (t.pauli_term, t.angle) for t in trotter.step_terms
        ]


class TestZassenhausErrorScaling:
    """Verify the operator-norm error scales as O(t^{p+1}) for orders p in {2, 3, 4}."""

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_heisenberg_scaling(self, order):
        """Heisenberg 4-site open chain (K = 3 commuting groups) scales as O(t^{p+1})."""
        slope = _fitted_error_slope(_heisenberg_hamiltonian(), order)
        assert abs(slope - (order + 1)) < 0.1

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_h2_scaling(self, order):
        """H2/STO-3G (K = 2 commuting groups, with an identity term) scales as O(t^{p+1})."""
        slope = _fitted_error_slope(_h2_hamiltonian(), order)
        assert abs(slope - (order + 1)) < 0.1

    def test_more_accurate_than_first_order_trotter(self):
        """At small t the order-2 Zassenhaus beats first-order Trotter on the same Hamiltonian."""
        hamiltonian = _heisenberg_hamiltonian()
        time = 0.05
        trotter = registry.create("hamiltonian_unitary_builder", "trotter", time=time, num_divisions=1)
        exact = _exact_unitary(hamiltonian, time)
        trotter_error = np.linalg.norm(exact - _builder_unitary(trotter.run(hamiltonian).get_container()), 2)
        zassenhaus_error = np.linalg.norm(exact - _zassenhaus_unitary(hamiltonian, time, order=2), 2)
        assert zassenhaus_error < trotter_error


class TestZassenhausPhaseEstimation:
    """End-to-end phase-estimation tests using the Zassenhaus builder as the unitary source."""

    @staticmethod
    def _run_iqpe(
        hamiltonian: QubitHamiltonian,
        state_vector: np.ndarray,
        *,
        time: float,
        num_bits: int,
        order: int,
        reference_energy: float,
        num_divisions: int = 1,
    ) -> float:
        """Run iterative phase estimation with the Zassenhaus builder; return the estimated energy.

        ``reference_energy`` selects which periodic phase-fraction branch (and
        eigenstate) to resolve against.
        """
        num_qubits = int(np.log2(len(state_vector)))
        params = {
            "rowMap": list(range(num_qubits - 1, -1, -1)),
            "stateVector": list(state_vector),
            "expansionOps": [],
            "numQubits": num_qubits,
        }
        qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(params)
        factories = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=params
        )
        circuit_builder = AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_iterative",
            num_bits=num_bits,
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
            unitary_builder=AlgorithmRef(
                "hamiltonian_unitary_builder", "zassenhaus", time=time, order=order, num_divisions=num_divisions
            ),
        )
        iqpe = IterativePhaseEstimation(shots_per_bit=1)
        iqpe.settings().set("qpe_circuit_builder", circuit_builder)
        iqpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42))
        result = iqpe.run(
            qubit_hamiltonian=hamiltonian,
            state_preparation=Circuit(qsharp_factory=factories, qsharp_op=qsharp_op),
        )
        # Resolve the periodic phase-fraction branch closest to the reference energy.
        candidates = [result.phase_fraction % 1.0, (1.0 - result.phase_fraction) % 1.0]
        energies = [energy_from_phase(candidate, evolution_time=time) for candidate in candidates]
        return energies[int(np.argmin([abs(energy - reference_energy) for energy in energies]))]

    def test_consumed_by_phase_estimation_exact_for_commuting(self):
        """A commuting Hamiltonian is evolved exactly, so QPE recovers the energy with no algorithmic error.

        ``H = 0.25 XX + 0.5 ZZ`` has the Bell state ``(|00> + |11>)/sqrt(2)`` as an
        exact eigenstate with eigenvalue ``0.25 + 0.5 = 0.75`` (XX and ZZ both act
        as +1 on it). Using a genuine eigenstate makes the result deterministic --
        a non-eigenstate would collapse onto an eigenvalue only probabilistically.
        """
        hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
        bell = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
        energy = self._run_iqpe(
            hamiltonian,
            bell,
            time=float(np.pi / 2),
            num_bits=4,
            order=2,
            reference_energy=0.75,
        )
        assert np.isclose(energy, 0.75, atol=1e-9)

    @pytest.mark.slow
    def test_h2_ground_state_to_chemical_accuracy(self):
        """Iterative QPE with the Zassenhaus builder recovers the H2 ground-state energy to 1.6e-3 Ha."""
        hamiltonian = _h2_hamiltonian()
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.to_matrix())
        ground_energy = float(eigenvalues[0])
        ground_state = eigenvectors[:, 0].real

        energy = self._run_iqpe(
            hamiltonian,
            ground_state,
            time=0.5,
            num_bits=12,
            order=4,
            num_divisions=2,
            reference_energy=ground_energy,
        )
        assert abs(energy - ground_energy) < 1.6e-3
