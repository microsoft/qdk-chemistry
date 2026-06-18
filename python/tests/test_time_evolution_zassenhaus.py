"""Tests for the Zassenhaus-expansion Hamiltonian unitary builder in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import functools

import numpy as np
import pytest
import scipy.linalg

from qdk_chemistry.algorithms import create, registry
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus import Zassenhaus
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus_expansion import (
    zassenhaus_factors,
)
from qdk_chemistry.data import MajoranaMapping, QubitHamiltonian, Structure, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils.model_hamiltonians import create_heisenberg_hamiltonian

try:
    from qdk_chemistry.data import LatticeGraph

    _HAS_LATTICE = True
except ImportError:  # pragma: no cover - LatticeGraph should be available
    _HAS_LATTICE = False


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_matrix(label: str) -> np.ndarray:
    """Dense matrix of a little-endian Pauli label (leftmost char = highest qubit)."""
    matrix = np.array([[1]], dtype=complex)
    for char in label:
        matrix = np.kron(matrix, _PAULI[char])
    return matrix


def _container_unitary(container: PauliProductFormulaContainer) -> np.ndarray:
    r"""Reconstruct the dense unitary of a :class:`PauliProductFormulaContainer`.

    Follows the same convention as downstream consumers (``PauliSequenceMapper``)
    and the Trotter tests: ``step_terms[0]`` is applied *first* (rightmost in the
    matrix product), and the single-step product is repeated ``step_reps`` times.
    """
    dim = 2**container.num_qubits
    step = np.eye(dim, dtype=complex)
    for term in container.step_terms:
        label = ["I"] * container.num_qubits
        for qubit, pauli in term.pauli_term.items():
            label[container.num_qubits - 1 - qubit] = pauli
        step = scipy.linalg.expm(-1j * term.angle * _pauli_matrix("".join(label))) @ step
    return np.linalg.matrix_power(step, container.step_reps)


def _terms_to_label(terms: dict[int, str], num_qubits: int) -> str:
    """Convert a ``{qubit: axis}`` mapping to a little-endian Pauli label."""
    chars = ["I"] * num_qubits
    for qubit, axis in terms.items():
        chars[num_qubits - 1 - qubit] = axis
    return "".join(chars)


def _heisenberg_chain_4() -> QubitHamiltonian:
    """4-site open Heisenberg chain with J=1 (XX + YY + ZZ on each bond)."""
    if _HAS_LATTICE:
        graph = LatticeGraph.chain(4, periodic=False)
        return create_heisenberg_hamiltonian(graph, jx=1.0, jy=1.0, jz=1.0, include_term_groups=False)
    # Fallback: build the chain directly.
    labels: list[str] = []
    for bond in range(3):
        for axis in "XYZ":
            mapping = {bond: axis, bond + 1: axis}
            labels.append(_terms_to_label(mapping, 4))
    return QubitHamiltonian(pauli_strings=labels, coefficients=np.ones(len(labels)))


@functools.lru_cache(maxsize=1)
def _h2_sto3g_jw() -> QubitHamiltonian:
    """H2/STO-3G qubit Hamiltonian (Jordan-Wigner, 4 qubits) built via the QDK pipeline.

    Produced the same way the rest of the library does (SCF -> Hamiltonian constructor
    -> Jordan-Wigner qubit mapping), rather than hardcoding coefficients, at the H2
    equilibrium bond length (1.4 Bohr ~ 0.74 A; ``Structure`` coordinates are in Bohr).
    Cached so the SCF runs only once across the test module.
    """
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    structure = Structure(coords, symbols=["H", "H"])

    scf_solver = create("scf_solver")
    _, wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g")

    hamiltonian = create("hamiltonian_constructor").run(wavefunction.get_orbitals())
    n_spin_orbitals = 2 * hamiltonian.get_orbitals().get_num_molecular_orbitals()
    return create("qubit_mapper").run(hamiltonian, MajoranaMapping.jordan_wigner(n_spin_orbitals))


# ----------------------------------------------------------------------------------
# Basic builder behaviour
# ----------------------------------------------------------------------------------


class TestZassenhausBuilder:
    """Smoke tests for the Zassenhaus builder."""

    def test_name_and_type(self):
        """The builder advertises the expected algorithm name and type."""
        builder = Zassenhaus()
        assert builder.name() == "zassenhaus"
        assert builder.type_name() == "hamiltonian_unitary_builder"

    def test_registered_in_factory(self):
        """The builder is discoverable through the standard registry mechanism."""
        assert "zassenhaus" in registry.available("hamiltonian_unitary_builder")
        builder = registry.create("hamiltonian_unitary_builder", "zassenhaus", order=3, time=0.5)
        assert isinstance(builder, Zassenhaus)
        assert builder.settings().get("order") == 3

    def test_returns_pauli_product_formula_container(self):
        """Running the builder yields a PauliProductFormulaContainer of the right shape."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "ZZ"], coefficients=np.array([1.0, 0.5]))
        builder = Zassenhaus(order=2, time=0.3)
        unitary = builder.run(hamiltonian)

        assert isinstance(unitary, UnitaryRepresentation)
        container = unitary.get_container()
        assert isinstance(container, PauliProductFormulaContainer)
        assert container.num_qubits == 2
        assert container.step_reps == 1
        assert len(container.step_terms) >= 2  # first-order terms plus corrections

    def test_num_divisions_sets_step_reps(self):
        """``num_divisions`` controls the repetition count, like the Trotter builder."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "ZZ"], coefficients=np.array([1.0, 0.5]))
        builder = Zassenhaus(order=2, time=0.4, num_divisions=5)
        container = builder.run(hamiltonian).get_container()
        assert container.step_reps == 5

    def test_non_hermitian_raises(self):
        """A non-Hermitian Hamiltonian is rejected."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XI"], coefficients=np.array([1.0 + 1.0j]))
        builder = Zassenhaus(order=2, time=0.1)
        with pytest.raises(ValueError, match="Non-Hermitian"):
            builder.run(hamiltonian)

    def test_invalid_order_raises(self):
        """Expansion orders below 1 are rejected."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XI"], coefficients=np.array([1.0]))
        builder = Zassenhaus(order=0, time=0.1)
        with pytest.raises(ValueError, match="order must be >= 1"):
            builder.run(hamiltonian)

    def test_first_order_is_bare_trotter_product(self):
        """At order 1 the Zassenhaus product is the bare first-order Trotter product.

        The product *ordering* differs (Zassenhaus emits its factors reversed), but
        for a first-order product only the multiset of single-term exponentials
        matters; it must coincide with the Trotter builder's.
        """
        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "ZZ", "IZ"], coefficients=np.array([1.0, 0.5, 0.25]))
        time = 0.2

        zassenhaus = Zassenhaus(order=1, time=time).run(hamiltonian).get_container()
        trotter = registry.create("hamiltonian_unitary_builder", "trotter", order=1, time=time, num_divisions=1)
        trotter_container = trotter.run(hamiltonian).get_container()

        def _multiset(container: PauliProductFormulaContainer) -> set:
            return {
                (frozenset(term.pauli_term.items()), round(term.angle, 12)) for term in container.step_terms
            }

        assert len(zassenhaus.step_terms) == len(hamiltonian.pauli_strings)
        assert _multiset(zassenhaus) == _multiset(trotter_container)


# ----------------------------------------------------------------------------------
# Error-scaling acceptance criterion
# ----------------------------------------------------------------------------------


def _fit_error_slope(hamiltonian: QubitHamiltonian, order: int) -> float:
    r"""Fit the slope of ``log || U_exact - U_builder ||`` vs ``log t``.

    Evaluated at a single time division over ``t in [1e-3, 1e-1]``; for a correct
    order-``p`` expansion the slope equals ``p + 1``.

    For high orders on small-coefficient Hamiltonians the error at the smallest
    times reaches the floating-point floor (``~1e-15``), where the log-log slope is
    meaningless.  Points within ~2 orders of that floor (``< 1e-13``) are dropped
    before the fit; every remaining point still follows the power law and lies within
    the specified ``[1e-3, 1e-1]`` window.
    """
    hamiltonian_matrix = hamiltonian.to_matrix()
    times = np.logspace(-3, -1, 12)
    errors = []
    for time in times:
        builder = Zassenhaus(order=order, time=float(time), num_divisions=1)
        container = builder.run(hamiltonian).get_container()
        u_builder = _container_unitary(container)
        u_exact = scipy.linalg.expm(-1j * hamiltonian_matrix * time)
        errors.append(np.linalg.norm(u_exact - u_builder, 2))
    errors = np.asarray(errors)

    mask = errors > 1e-13
    if int(mask.sum()) < 4:
        # Too few points above the floor: keep the six largest-t points.
        mask = np.zeros_like(errors, dtype=bool)
        mask[-6:] = True

    slope, _ = np.polyfit(np.log(times[mask]), np.log(errors[mask]), 1)
    return float(slope)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_error_scaling_heisenberg_chain(order: int):
    """4-site Heisenberg chain: operator-norm error scales as O(t^(p+1))."""
    slope = _fit_error_slope(_heisenberg_chain_4(), order)
    assert abs(slope - (order + 1)) < 0.1


@pytest.mark.parametrize("order", [2, 3, 4])
def test_error_scaling_h2_sto3g(order: int):
    """H2/STO-3G (Jordan-Wigner): operator-norm error scales as O(t^(p+1))."""
    slope = _fit_error_slope(_h2_sto3g_jw(), order)
    assert abs(slope - (order + 1)) < 0.1


def test_factor_count_grows_with_order():
    """Higher expansion orders append additional correction factors."""
    terms = [("XI", 1.0), ("ZZ", 0.5), ("IY", 0.25)]
    counts = [len(zassenhaus_factors(terms, order)) for order in (1, 2, 3, 4)]
    assert counts == sorted(counts)
    assert counts[0] == len(terms)  # order 1 == bare Trotter product
    assert counts[-1] > counts[0]


def test_target_accuracy_increases_divisions():
    """A tighter ``target_accuracy`` raises the automatic time-division count."""
    hamiltonian = QubitHamiltonian(pauli_strings=["XI", "ZZ", "IY"], coefficients=np.array([1.0, 0.6, 0.4]))

    baseline = Zassenhaus(order=3, time=1.0).run(hamiltonian).get_container().step_reps
    loose = Zassenhaus(order=3, time=1.0, target_accuracy=1e-2).run(hamiltonian).get_container().step_reps
    tight = Zassenhaus(order=3, time=1.0, target_accuracy=1e-4).run(hamiltonian).get_container().step_reps

    assert baseline == 1  # target_accuracy disabled by default
    assert tight > loose > baseline


def test_term_partition_is_consumed():
    """A populated ``term_partition`` orders the seed terms by group; result stays O(t^(p+1))."""
    hamiltonian = create_heisenberg_hamiltonian(
        LatticeGraph.chain(4, periodic=False), jx=1.0, jy=1.0, jz=1.0, include_term_groups=True
    )
    assert hamiltonian.term_partition is not None

    # _ordered_terms must reproduce the partition's grouped index order.
    ordered = Zassenhaus._ordered_terms(hamiltonian, weight_threshold=1e-12)
    expected_indices: list[int] = []
    for group in hamiltonian.term_partition.groups:
        for layer in group:  # LayeredPartition: group -> layers -> indices
            expected_indices.extend(layer)
    expected_labels = [hamiltonian.pauli_strings[i] for i in expected_indices]
    assert [label for label, _ in ordered] == expected_labels

    # Consuming the partition must not break the error scaling.
    slope = _fit_error_slope(hamiltonian, order=3)
    assert abs(slope - 4) < 0.1


# ----------------------------------------------------------------------------------
# Phase-estimation consumability
# ----------------------------------------------------------------------------------

# The phase-estimation stack relies on the Q#/QDK simulator utilities, mirrored
# from ``test_phase_estimation_iterative.py``.
from qdk_chemistry.data import AlgorithmRef, Circuit  # noqa: E402
from qdk_chemistry.data.circuit import QsharpFactoryData  # noqa: E402
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import (  # noqa: E402
    IterativePhaseEstimation,
)
from qdk_chemistry.utils.phase import energy_from_phase  # noqa: E402
from qdk_chemistry.utils.qsharp import QSHARP_UTILS  # noqa: E402

_SEED = 42


def _resolve_energy(phase_fraction: float, evolution_time: float, expected_energy: float) -> float:
    """Resolve QPE phase periodicity by selecting the energy closest to ``expected_energy``."""
    candidates = [phase_fraction % 1.0, (1.0 - phase_fraction) % 1.0]
    energies = [energy_from_phase(c, evolution_time=evolution_time) for c in candidates]
    index = int(np.argmin([abs(e - expected_energy) for e in energies]))
    return energies[index]


def _run_iterative_zassenhaus(
    hamiltonian: QubitHamiltonian,
    state_vector: np.ndarray,
    *,
    evolution_time: float,
    num_bits: int,
    order: int,
    num_divisions: int,
    shots_per_bit: int = 1,
) -> float:
    """Run iterative QPE using the Zassenhaus builder and return the raw phase fraction."""
    num_qubits = int(np.log2(len(state_vector)))
    state_prep_params = {
        "rowMap": list(range(num_qubits - 1, -1, -1)),
        "stateVector": np.asarray(state_vector, dtype=float).tolist(),
        "expansionOps": [],
        "numQubits": num_qubits,
    }
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
    qsharp_factories = QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=state_prep_params
    )

    iqpe = IterativePhaseEstimation(num_bits=num_bits, shots_per_bit=shots_per_bit)
    iqpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=_SEED))
    iqpe.settings().set("circuit_mapper", AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"))
    iqpe.settings().set(
        "unitary_builder",
        AlgorithmRef(
            "hamiltonian_unitary_builder",
            "zassenhaus",
            time=evolution_time,
            order=order,
            num_divisions=num_divisions,
        ),
    )

    result = iqpe.run(
        qubit_hamiltonian=hamiltonian,
        state_preparation=Circuit(qsharp_factory=qsharp_factories, qsharp_op=qsharp_op),
    )
    return result.phase_fraction


def test_phase_estimation_consumable():
    """The builder is consumable by IterativePhaseEstimation end-to-end (toy 2-qubit case).

    Uses the exact top eigenstate ``(|00> + |11>)/sqrt(2)`` of ``H = 0.25 XX + 0.5 ZZ``
    (eigenvalue 0.75) so the recovered energy is deterministic.  Several divisions keep
    the Zassenhaus unitary accurate at ``t = pi/2``.
    """
    hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=np.array([0.25, 0.5]))
    state_vector = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    evolution_time = float(np.pi / 2.0)

    phase_fraction = _run_iterative_zassenhaus(
        hamiltonian,
        state_vector,
        evolution_time=evolution_time,
        num_bits=4,
        order=3,
        num_divisions=20,
        shots_per_bit=1,
    )
    resolved = _resolve_energy(phase_fraction, evolution_time, expected_energy=0.75)
    assert abs(resolved - 0.75) < 1e-2


@pytest.mark.slow
def test_h2_ground_state_chemical_accuracy():
    """Estimate the H2/STO-3G ground-state energy to chemical accuracy via QPE + Zassenhaus.

    The exact ground eigenvector is used as the input state, so the result is
    deterministic.  ``num_bits = 12`` gives an energy resolution of
    ``2*pi / 2**12 ~ 1.5e-3`` (max rounding error ~7.7e-4 at ``t = 1``), and the
    builder's product-formula error at the most-significant bit (power ``2**11``) is
    ``~2**11 * t**(p+1) / N**p``; the high expansion ``order = 6`` makes this converge
    quickly, so ``num_divisions = 16`` keeps it well below chemical accuracy while
    keeping the circuit depth tractable.  This test is still compute-heavy (hence
    ``slow``); the bit/division/order counts may need adjustment on different simulators.
    """
    hamiltonian = _h2_sto3g_jw()
    matrix = hamiltonian.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    ground_energy = float(eigenvalues[0])
    ground_state = np.real(eigenvectors[:, 0])

    evolution_time = 1.0
    phase_fraction = _run_iterative_zassenhaus(
        hamiltonian,
        ground_state,
        evolution_time=evolution_time,
        num_bits=12,
        order=6,
        num_divisions=16,
        shots_per_bit=1,
    )
    resolved = _resolve_energy(phase_fraction, evolution_time, expected_energy=ground_energy)
    error = abs(resolved - ground_energy)
    print(
        f"\n[H2/STO-3G QPE] estimated = {resolved:.6f} Ha | exact (FCI) = {ground_energy:.6f} Ha "
        f"| |error| = {error:.2e} Ha (chemical accuracy = 1.6e-3 Ha)"
    )
    assert error < 1.6e-3
