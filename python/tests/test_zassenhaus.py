"""Test suite for the Zassenhaus evolution builder."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
from collections.abc import Hashable
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg

from qdk_chemistry.algorithms import create, registry
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus import Zassenhaus
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus_error import (
    zassenhaus_steps_commutator,
    zassenhaus_steps_naive,
)
from qdk_chemistry.data import (
    AlgorithmRef,
    FlatPartition,
    MajoranaMapping,
    PauliProductFormulaContainer,
    QubitHamiltonian,
    Structure,
    UnitaryRepresentation,
)
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import ExponentiatedPauliTerm
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
from qdk_chemistry.utils.zassenhaus_generation import (
    CommutatorPlan,
    PlanExpr,
    PlanTerm,
    zassenhaus_commutator_plan,
)

from .reference_tolerances import (
    float_comparison_relative_tolerance,
)

_RUN_SLOW_TESTS = os.getenv("QDK_CHEMISTRY_RUN_SLOW_TESTS", "").lower() in {"1", "true", "yes"}


class TestZassenhausGeneration:
    """Tests for symbolic Zassenhaus generation utilities."""

    @staticmethod
    def _expand_plan_term(ref: PlanTerm, plan: CommutatorPlan) -> Hashable | tuple:
        """Recursively expand a plan term into its nested commutator tuple representation."""
        if ref not in plan:
            return ref
        left, right = plan[ref]
        return (
            TestZassenhausGeneration._expand_plan_term(left, plan),
            TestZassenhausGeneration._expand_plan_term(right, plan),
        )

    @staticmethod
    def _expand_plan_expr(expr: PlanExpr, plan: CommutatorPlan) -> dict:
        """Expand all terms in a plan expression to their nested commutator tuple representations."""
        return {TestZassenhausGeneration._expand_plan_term(ref, plan): coeff for ref, coeff in expr.items()}

    @staticmethod
    def _assert_plan_is_dependency_ordered(plan: CommutatorPlan, leaves: tuple[Hashable, ...]) -> None:
        """Assert that every node in the plan only depends on previously computed nodes."""
        available: set[PlanTerm] = set(leaves)
        for node, (left, right) in plan.items():
            assert left in available
            assert right in available
            available.add(node)

    @staticmethod
    def _commutator(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return the matrix commutator [left, right] = left @ right - right @ left."""
        return left @ right - right @ left

    @staticmethod
    def _evaluate_matrix_plan_term(ref: PlanTerm, plan: CommutatorPlan, leaves: tuple[np.ndarray, ...]) -> np.ndarray:
        """Recursively evaluate a plan term as nested matrix commutators over the given leaf matrices."""
        if ref not in plan:
            return leaves[ref]
        left, right = plan[ref]
        return TestZassenhausGeneration._commutator(
            TestZassenhausGeneration._evaluate_matrix_plan_term(left, plan, leaves),
            TestZassenhausGeneration._evaluate_matrix_plan_term(right, plan, leaves),
        )

    @staticmethod
    def _zassenhaus_matrix_product(leaves: tuple[np.ndarray, ...], *, order: int, time: float) -> np.ndarray:
        """Compute the Zassenhaus product formula matrix for the given leaf generators up to the specified order."""
        planned_exponents, plan = zassenhaus_commutator_plan(tuple(range(len(leaves))), max_order=order)
        product = np.eye(leaves[0].shape[0], dtype=complex)
        for leaf in leaves:
            product = product @ scipy.linalg.expm(time * leaf)
        for degree in range(2, order + 1):
            correction = sum(
                float(coeff) * TestZassenhausGeneration._evaluate_matrix_plan_term(ref, plan, leaves)
                for ref, coeff in planned_exponents[degree].items()
            )
            product = product @ scipy.linalg.expm((time**degree) * correction)
        return product

    def test_zassenhaus_symbolic_generation(self):
        """Verify plan ordering and correct expansion coefficients for 2 and 3 operator formulas."""
        # 2-operator case
        planned_exponents_2, plan_2 = zassenhaus_commutator_plan(("A", "B"), max_order=5)
        self._assert_plan_is_dependency_ordered(plan_2, ("A", "B"))

        assert self._expand_plan_expr(planned_exponents_2[2], plan_2) == {("A", "B"): Fraction(-1, 2)}
        assert self._expand_plan_expr(planned_exponents_2[3], plan_2) == {
            ("A", ("A", "B")): Fraction(1, 6),
            ("B", ("A", "B")): Fraction(1, 3),
        }
        assert self._expand_plan_expr(planned_exponents_2[4], plan_2) == {
            ("A", ("A", ("A", "B"))): Fraction(-1, 24),
            ("B", ("A", ("A", "B"))): Fraction(-1, 8),
            ("B", ("B", ("A", "B"))): Fraction(-1, 8),
        }

        # 3-operator case
        planned_exponents_3, plan_3 = zassenhaus_commutator_plan(("A", "B", "C"), max_order=2)
        self._assert_plan_is_dependency_ordered(plan_3, ("A", "B", "C"))
        assert self._expand_plan_expr(planned_exponents_3[2], plan_3) == {
            ("A", "B"): Fraction(-1, 2),
            ("A", "C"): Fraction(-1, 2),
            ("B", "C"): Fraction(-1, 2),
        }

    def test_zassenhaus_matrix_product_scaling(self):
        """Check generated plans yield expected asymptotic order for noncommuting random matrices."""
        rng = np.random.default_rng(123)
        leaves = tuple((rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) / 8.0 for _ in range(4))
        exact_generator = sum(leaves)
        times = np.logspace(-2, -1, 5)

        for order in (2, 3, 4):
            errors = np.array(
                [
                    np.linalg.norm(
                        self._zassenhaus_matrix_product(leaves, order=order, time=time)
                        - scipy.linalg.expm(time * exact_generator),
                        ord=2,
                    )
                    for time in times
                ]
            )
            slope = np.polyfit(np.log(times), np.log(errors), deg=1)[0]
            assert np.isclose(slope, order + 1, atol=0.1)


class TestZassenhausStepEstimation:
    """Consolidated tests for Zassenhaus steps estimation under naive and commutator bounds."""

    @staticmethod
    def _first_order_product_formula_error(hamiltonian: QubitHamiltonian, *, time: float, steps: int) -> float:
        """Return the operator-norm error of the first-order Lie-Trotter product formula."""
        step_unitary = np.eye(2**hamiltonian.num_qubits, dtype=complex)
        for label, coeff in zip(hamiltonian.pauli_strings, hamiltonian.coefficients, strict=True):
            p_matrix = pauli_to_dense_matrix([label], np.asarray([complex(coeff).real], dtype=complex))
            step_unitary = scipy.linalg.expm(-1j * time / steps * p_matrix) @ step_unitary
        approximate = np.linalg.matrix_power(step_unitary, steps)
        exact = scipy.linalg.expm(-1j * time * hamiltonian.to_matrix())
        return float(np.linalg.norm(approximate - exact, ord=2))

    def test_zassenhaus_steps_bounds(self):
        """Verify naive and commutator-aware bounds at different orders and inputs."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        assert zassenhaus_steps_naive(h, 1.0, 0.1, order=1) == 40
        assert zassenhaus_steps_naive(h, 1.0, 0.1, order=2) == 13

        custom_exponents = {2: {"custom": Fraction(3, 2)}}
        assert zassenhaus_steps_naive(h, 1.0, 0.1, order=1, commutator_exponents=custom_exponents) == 120

        h_single = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        assert zassenhaus_steps_naive(h_single, 1.0, 0.1, order=1) == 1

        # Invalid input checking
        with pytest.raises(ValueError, match="C_3"):
            zassenhaus_steps_naive(h, 1.0, 0.1, order=2, commutator_exponents={2: {}})

        # Commutator-aware bounds
        h_anticommuting = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        assert zassenhaus_steps_commutator(h_anticommuting, 1.0, 0.1, order=1) == 10

        # Commuting terms are exact
        h_commuting = QubitHamiltonian(pauli_strings=["ZI", "IZ", "ZZ"], coefficients=[0.5, -0.25, 0.125])
        assert zassenhaus_steps_commutator(h_commuting, 4.0, 1e-6, order=1) == 1
        assert zassenhaus_steps_commutator(h_commuting, 4.0, 1e-6, order=2) == 1

        # Commutator-aware is tighter than naive
        h_dense = QubitHamiltonian(
            pauli_strings=["XI", "ZI", "IX", "IZ", "ZZ"], coefficients=[0.7, -0.2, 0.5, 0.3, -0.4]
        )
        assert zassenhaus_steps_commutator(h_dense, 1.0, 0.01, order=2) < zassenhaus_steps_naive(
            h_dense, 1.0, 0.01, order=2
        )

        # Verify computed steps bounds the actual operator-norm error
        steps = zassenhaus_steps_commutator(h_anticommuting, 0.7, 0.01, order=1)
        assert self._first_order_product_formula_error(h_anticommuting, time=0.7, steps=steps) <= 0.01


class TestZassenhausTimeEvolution:
    """Tests for the Zassenhaus time evolution builder."""

    @staticmethod
    def _pauli_label_from_map(pauli_term: dict[int, str], num_qubits: int) -> str:
        """Convert a qubit-index-to-Pauli dict to a full Pauli string in big-endian order."""
        return "".join(pauli_term.get(i, "I") for i in reversed(range(num_qubits)))

    @staticmethod
    def _pauli_label_from_qubit_ops(num_qubits: int, qubit_ops: dict[int, str]) -> str:
        """Build a Pauli string of length num_qubits from a mapping of qubit index to Pauli operator."""
        chars = ["I"] * num_qubits
        for qubit, pauli in qubit_ops.items():
            chars[num_qubits - qubit - 1] = pauli
        return "".join(chars)

    @staticmethod
    def _zassenhaus_unitary_matrix(hamiltonian: QubitHamiltonian, *, order: int, time: float) -> np.ndarray:
        """Build the Zassenhaus approximate unitary matrix for the given Hamiltonian, order, and time."""
        builder = Zassenhaus(num_divisions=1, order=order, time=time)
        container = builder.run(hamiltonian).get_container()

        step_unitary = np.eye(2**hamiltonian.num_qubits, dtype=complex)
        for term in container.step_terms:
            pauli_label = TestZassenhausTimeEvolution._pauli_label_from_map(
                term.pauli_term, num_qubits=hamiltonian.num_qubits
            )
            pauli_matrix = pauli_to_dense_matrix([pauli_label], np.array([1.0]))
            cos_val = np.cos(term.angle)
            sin_val = np.sin(term.angle)
            rotation_matrix = cos_val * np.eye(pauli_matrix.shape[0], dtype=complex) - 1j * sin_val * pauli_matrix
            step_unitary = rotation_matrix @ step_unitary

        return np.linalg.matrix_power(step_unitary, container.step_reps)

    @staticmethod
    def _fit_zassenhaus_error_slope(hamiltonian: QubitHamiltonian, *, order: int) -> float:
        """Fit the log-log slope of the Zassenhaus operator-norm error as a function of time."""
        times = np.logspace(-3, -1, 5)
        ham_mat = hamiltonian.to_matrix()
        errors = np.array(
            [
                np.linalg.norm(
                    TestZassenhausTimeEvolution._zassenhaus_unitary_matrix(hamiltonian, order=order, time=t)
                    - scipy.linalg.expm(-1j * t * ham_mat),
                    ord=2,
                )
                for t in times
            ]
        )
        resolved = errors > 1e-13
        assert np.count_nonzero(resolved) >= 3
        return float(np.polyfit(np.log(times[resolved]), np.log(errors[resolved]), deg=1)[0])

    @staticmethod
    def _open_heisenberg_chain_4_site() -> QubitHamiltonian:
        """Return the open-boundary Heisenberg XXX chain Hamiltonian on 4 sites."""
        labels = [
            TestZassenhausTimeEvolution._pauli_label_from_qubit_ops(4, {site: pauli, site + 1: pauli})
            for site in range(3)
            for pauli in ("X", "Y", "Z")
        ]
        h = QubitHamiltonian(pauli_strings=labels, coefficients=[1.0] * len(labels))
        grouper = registry.create("term_grouper", "commuting")
        return grouper.run(h)

    @staticmethod
    def _h2_sto3g_jordan_wigner_hamiltonian() -> QubitHamiltonian:
        """Return the H2/STO-3G active-space Hamiltonian mapped via Jordan-Wigner."""
        structure = Structure(
            np.array([[0.0, 0.0, -0.72], [0.0, 0.0, 0.72]], dtype=float),
            ["H", "H"],
        )
        scf_solver = create("scf_solver")
        _, scf_wavefunction = scf_solver.run(
            structure,
            charge=0,
            spin_multiplicity=1,
            basis_or_guess="sto-3g",
        )
        selector = create(
            "active_space_selector",
            "qdk_valence",
            num_active_electrons=2,
            num_active_orbitals=2,
        )
        active_orbitals = selector.run(scf_wavefunction).get_orbitals()
        constructor = create("hamiltonian_constructor")
        active_hamiltonian = constructor.run(active_orbitals)
        n_spin_orbitals = 2 * active_hamiltonian.get_orbitals().get_num_molecular_orbitals()
        h = create("qubit_mapper", "qdk").run(
            active_hamiltonian,
            MajoranaMapping.jordan_wigner(n_spin_orbitals),
        )
        grouper = registry.create("term_grouper", "commuting")
        return grouper.run(h)

    def test_zassenhaus_builder_and_decomposition(self):
        """Test metadata, registry creation, settings, decomposition, filtering, and evolution examples."""
        builder = Zassenhaus()
        assert builder.name() == "zassenhaus"
        assert builder.type_name() == "hamiltonian_unitary_builder"

        builder_registry = create(
            "hamiltonian_unitary_builder", "zassenhaus", order=4, num_divisions=3, time=0.2, weight_threshold=1e-10
        )
        assert isinstance(builder_registry, Zassenhaus)
        assert builder_registry.settings().get("order") == 4
        assert builder_registry.settings().get("num_divisions") == 3
        assert builder_registry.settings().get("time") == 0.2
        assert builder_registry.settings().get("weight_threshold") == 1e-10

        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "ZZ"], coefficients=[2.0, 1.0])
        rep = create("hamiltonian_unitary_builder", "zassenhaus", order=2, num_divisions=4, time=0.2).run(hamiltonian)
        assert isinstance(rep, UnitaryRepresentation)
        container = rep.get_container()
        assert isinstance(container, PauliProductFormulaContainer)
        assert container.num_qubits == hamiltonian.num_qubits
        assert container.step_reps == 4
        assert len(container.step_terms) > 0

        # Basic decomposition at order=1 (no corrections)
        ham_single = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        terms = builder._decompose_zassenhaus_step(ham_single, time=2.0, order=1)
        assert len(terms) == 2
        assert terms[0].pauli_term == {0: "X"}
        assert np.isclose(terms[0].angle, 2.0)
        assert terms[1].pauli_term == {0: "Z"}
        assert np.isclose(terms[1].angle, 1.0)

        # Filtering small coefficients
        ham_filter = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1e-15, 1.0])
        terms_filtered = builder._decompose_zassenhaus_step(ham_filter, time=1.0, atol=1e-12)
        assert len(terms_filtered) == 1
        assert terms_filtered[0].pauli_term == {0: "Z"}

        # Non-Hermitian rejection
        ham_non_hermitian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0 + 1.0j])
        with pytest.raises(ValueError, match="Non-Hermitian"):
            builder._decompose_zassenhaus_step(ham_non_hermitian, time=1.0)

        # Multi-step angles and reps
        ham_two = QubitHamiltonian(pauli_strings=["XI", "ZZ"], coefficients=[2.0, 1.0])
        builder_steps = Zassenhaus(num_divisions=4, order=2, time=0.2)
        container_steps = builder_steps.run(ham_two).get_container()
        assert container_steps.step_reps == 4
        assert len(container_steps.step_terms) == 3
        sorted_terms = sorted(container_steps.step_terms, key=lambda t: abs(t.angle))

        # Commutator correction term (C2 angle = coeff * dt^2 = 2 * 0.05^2 = 0.005)
        assert sorted_terms[0].pauli_term == {0: "Z", 1: "Y"}
        assert np.isclose(abs(sorted_terms[0].angle), 0.005, atol=1e-12)
        # ZZ term (angle = 1.0 * dt = 0.05)
        assert sorted_terms[1].pauli_term == {0: "Z", 1: "Z"}
        assert np.isclose(abs(sorted_terms[1].angle), 0.05, atol=1e-12)
        # XI term (angle = 2.0 * dt = 0.1)
        assert sorted_terms[2].pauli_term == {1: "X"}
        assert np.isclose(abs(sorted_terms[2].angle), 0.1, atol=1e-12)

        # Order 1 & 2 X-Z simulation comparison against scipy expm
        ham_sim = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.5, 0.5])
        t_sim = 0.1

        # Order 1 comparison
        u_z1 = np.eye(2, dtype=complex)
        container1 = Zassenhaus(num_divisions=1, order=1, time=t_sim).run(ham_sim).get_container()
        for term in container1.step_terms:
            pauli_char = next(iter(term.pauli_term.values()))
            pauli_mat = pauli_to_dense_matrix([pauli_char], np.array([1.0]))
            u_z1 = scipy.linalg.expm(-1j * term.angle * pauli_mat) @ u_z1
        u_exact = scipy.linalg.expm(-1j * t_sim * ham_sim.to_matrix())
        assert np.linalg.norm(u_z1 - u_exact, ord=2) < 0.05

        # Order 2 comparison
        u_z2 = np.eye(2, dtype=complex)
        container2 = Zassenhaus(num_divisions=1, order=2, time=t_sim).run(ham_sim).get_container()
        for term in container2.step_terms:
            pauli_char = next(iter(term.pauli_term.values()))
            pauli_mat = pauli_to_dense_matrix([pauli_char], np.array([1.0]))
            u_z2 = scipy.linalg.expm(-1j * term.angle * pauli_mat) @ u_z2
        assert np.linalg.norm(u_z2 - u_exact, ord=2) < 0.01

        # Auto order selection is removed; order <= 0 now raises NotImplementedError
        with pytest.raises(NotImplementedError):
            Zassenhaus(order=0).run(ham_two)

        with pytest.raises(NotImplementedError):
            Zassenhaus(order=-1).run(ham_two)

        # Commuting groups partitioning
        ham_partitioned = QubitHamiltonian(
            pauli_strings=["XI", "IX", "ZZ", "YY"],
            coefficients=[0.7, -0.2, 0.3, 0.11],
            term_partition=FlatPartition(strategy="commuting", groups=((0, 1), (2, 3))),
        )
        container_part = Zassenhaus(num_divisions=1, order=4, time=t_sim).run(ham_partitioned).get_container()
        assert len(container_part.step_terms) == 12

    def test_zassenhaus_custom_term_grouper(self):
        """Verify that a custom term grouper is correctly applied to correction terms."""
        h = QubitHamiltonian(pauli_strings=["XI", "ZZ"], coefficients=[2.0, 1.0])
        # Default term grouper should work
        builder_default = Zassenhaus(order=2, time=0.2)
        container_default = builder_default.run(h).get_container()

        # Let's configure it with a custom term grouper (e.g. qubit_wise_commuting)
        custom_grouper = AlgorithmRef("term_grouper", "qubit_wise_commuting")
        builder_custom = Zassenhaus(order=2, time=0.2, term_grouper=custom_grouper)
        container_custom = builder_custom.run(h).get_container()

        assert len(container_default.step_terms) > 0
        assert len(container_custom.step_terms) > 0

    def test_zassenhaus_bubble_and_merge_optimization(self):
        """Verify the Bubble & Merge optimization reduces term counts and preserves accuracy."""
        builder = Zassenhaus()
        terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5),
            ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=0.2),
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.3),
        ]
        optimized = builder._optimize_pauli_sequence(terms)
        assert len(optimized) == 2  # Should merge the two X0 terms
        assert optimized[0].pauli_term == {0: "X"}
        assert np.isclose(optimized[0].angle, 0.8)
        assert optimized[1].pauli_term == {1: "Z"}
        assert np.isclose(optimized[1].angle, 0.2)

        non_commuting_terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5),
            ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.2),
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.3),
        ]
        optimized_nc = builder._optimize_pauli_sequence(non_commuting_terms)
        assert len(optimized_nc) == 3  # X0 and Z0 do not commute

        canceling_terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5),
            ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=0.2),
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=-0.5),
        ]
        optimized_cancel = builder._optimize_pauli_sequence(canceling_terms)
        assert len(optimized_cancel) == 1
        assert optimized_cancel[0].pauli_term == {1: "Z"}

        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "IX", "ZZ", "YY"],
            coefficients=[0.7, -0.2, 0.3, 0.11],
            term_partition=FlatPartition(strategy="commuting", groups=((0, 1), (2, 3))),
        )
        container_opt = Zassenhaus(num_divisions=1, order=4, time=0.1).run(hamiltonian).get_container()
        raw_builder = Zassenhaus(num_divisions=1, order=4, time=0.1)
        raw_terms = raw_builder._decompose_zassenhaus_step(hamiltonian, time=0.1, order=4)
        assert len(container_opt.step_terms) < len(raw_terms)

        def terms_to_unitary(step_terms):
            """Convert a sequence of ExponentiatedPauliTerms to a unitary matrix."""
            u = np.eye(2**hamiltonian.num_qubits, dtype=complex)
            for t in step_terms:
                pauli_label = TestZassenhausTimeEvolution._pauli_label_from_map(
                    t.pauli_term, num_qubits=hamiltonian.num_qubits
                )
                pauli_matrix = pauli_to_dense_matrix([pauli_label], np.array([1.0]))
                u = scipy.linalg.expm(-1j * t.angle * pauli_matrix) @ u
            return u

        u_opt = terms_to_unitary(container_opt.step_terms)
        u_raw = terms_to_unitary(raw_terms)

        assert np.allclose(u_opt, u_raw, atol=1e-12)

    def test_zassenhaus_operator_norm_error_scaling(self):
        """Check empirical operator-norm error scaling slopes match order + 1 to within 0.1."""
        cases = [
            (self._open_heisenberg_chain_4_site, 2),
            (self._open_heisenberg_chain_4_site, 3),
            (self._open_heisenberg_chain_4_site, 4),
            (self._h2_sto3g_jordan_wigner_hamiltonian, 2),
            (self._h2_sto3g_jordan_wigner_hamiltonian, 3),
            (self._h2_sto3g_jordan_wigner_hamiltonian, 4),
        ]
        for factory, order in cases:
            slope = self._fit_zassenhaus_error_slope(factory(), order=order)
            assert np.isclose(slope, order + 1, atol=0.1)


class TestZassenhausPhaseEstimation:
    """End-to-end H2/STO-3G IQPE tests using Zassenhaus evolution."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not _RUN_SLOW_TESTS,
        reason="Skipping slow test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
    )
    def test_zassenhaus_h2_ground_state(self) -> None:
        """Estimate the H2/STO-3G ground-state energy with QDK IQPE and Zassenhaus evolution."""
        Logger.set_global_level("error")

        structure = Structure(
            np.array([[0.0, 0.0, -0.72], [0.0, 0.0, 0.72]], dtype=float),
            ["H", "H"],
        )

        scf_solver = create("scf_solver")
        scf_total_energy, scf_wavefunction = scf_solver.run(
            structure,
            charge=0,
            spin_multiplicity=1,
            basis_or_guess="sto-3g",
        )

        selector = create(
            "active_space_selector",
            "qdk_valence",
            num_active_electrons=2,
            num_active_orbitals=2,
        )
        active_orbitals = selector.run(scf_wavefunction).get_orbitals()

        constructor = create("hamiltonian_constructor")
        active_hamiltonian = constructor.run(active_orbitals)

        mc_calculator = create("multi_configuration_calculator")
        casci_total_energy, casci_wavefunction = mc_calculator.run(active_hamiltonian, 1, 1)

        n_spin_orbitals = 2 * active_hamiltonian.get_orbitals().get_num_molecular_orbitals()
        qubit_hamiltonian = create("qubit_mapper", "qdk").run(
            active_hamiltonian,
            MajoranaMapping.jordan_wigner(n_spin_orbitals),
        )
        state_preparation = create("state_prep", "sparse_isometry_gf2x").run(casci_wavefunction)

        # Configure the Zassenhaus evolution builder reference
        num_bits = 12
        evolution_time = 1.175
        zassenhaus_order = 2
        zassenhaus_num_divisions = 10
        unitary_builder_ref = AlgorithmRef(
            "hamiltonian_unitary_builder",
            "zassenhaus",
            time=evolution_time,
            order=zassenhaus_order,
            num_divisions=zassenhaus_num_divisions,
        )

        # Configure the iterative phase estimation (IQPE) circuit builder with Zassenhaus evolution
        qpe_builder_ref = AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_iterative",
            num_bits=num_bits,
            unitary_builder=unitary_builder_ref,
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )
        iqpe = create("phase_estimation", "qdk_iterative", shots_per_bit=15)
        iqpe.settings().set("qpe_circuit_builder", qpe_builder_ref)
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42),
        )

        result = iqpe.run(
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
        )
        estimated_total_energy = result.raw_energy + active_hamiltonian.get_core_energy()

        # Assert that total energy is within chemical accuracy of CASCI energy
        assert np.isclose(
            estimated_total_energy,
            casci_total_energy,
            rtol=float_comparison_relative_tolerance,
            atol=1.6e-3,
        )
