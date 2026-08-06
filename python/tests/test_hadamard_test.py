"""Tests for Hadamard test generator algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest
from qdk import TargetProfile

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hadamard_test.hadamard_test import HadamardTestBasis
from qdk_chemistry.data import AlgorithmRef, Circuit, MajoranaMapping, Structure, UnitaryRepresentation
from qdk_chemistry.utils.qsharp import create_qsharp_context, use_qsharp_context

_HAS_QSHARP = importlib.util.find_spec("qdk.qsharp") is not None


@pytest.fixture(scope="module", autouse=True)
def _base_profile_context():
    """Compile this module's Q# under the Base profile.

    These tests pin exact shot counts, so they depend on the simulator's RNG stream. Base keeps
    the measurement-based uncompute in the AND ladders lowered to its unitary decomposition,
    while the adaptive default emits real mid-circuit measurements that draw extra samples and
    shift every count.

    The scope has to be ``module`` rather than the usual per-test
    ``usefixtures("use_base_qdk_ctx")`` because ``water_hadamard_benchmark`` is module-scoped
    and builds Q# callables. Pytest sets module-scoped fixtures up before function-scoped ones,
    so a function-scoped context would leave those callables bound to the adaptive context and
    absent from the Base-lowered package when QIR is generated.
    """
    with use_qsharp_context(create_qsharp_context(TargetProfile.Base)) as context:
        yield context


_SHOTS = 100
_EVOLUTION_TIME = float(np.pi / 48.0)
_OBSERVABLE_POWER = 10


def _make_hadamard_test(test_basis: HadamardTestBasis = HadamardTestBasis.X):
    """Create a Hadamard test configured with the given measurement basis."""
    return create(
        "hadamard_test",
        test_basis=test_basis.value,
        circuit_executor=AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=42),
    )


@dataclass(frozen=True)
class HadamardWaterBenchmark:
    """Container for the water benchmark used by Hadamard generator tests."""

    state_preparation: Circuit
    unitary: UnitaryRepresentation


@pytest.fixture(scope="module")
def water_hadamard_benchmark() -> HadamardWaterBenchmark:
    """Construct water-system state prep and controlled evolution circuits once per module."""
    num_active_electrons = 2
    num_active_orbitals = 3

    water_coords = np.array(
        [
            [0.000000, 0.000000, 0.000000],
            [0.758602, 0.000000, 0.504284],
            [-0.758602, 0.000000, 0.504284],
        ],
        dtype=float,
    )
    structure = Structure(water_coords, ["O", "H", "H"])

    scf_solver = create("scf_solver")
    _, scf_wfn = scf_solver.run(
        structure,
        charge=0,
        spin_multiplicity=1,
        basis_or_guess="cc-pvdz",
    )

    active_space_selector = create(
        "active_space_selector",
        algorithm_name="qdk_valence",
        num_active_electrons=num_active_electrons,
        num_active_orbitals=num_active_orbitals,
    )
    active_wfn = active_space_selector.run(scf_wfn)

    hamiltonian_constructor = create("hamiltonian_constructor")
    active_hamiltonian = hamiltonian_constructor.run(active_wfn.get_orbitals())

    qubit_mapper = create("qubit_mapper", algorithm_name="qdk")
    mapping = MajoranaMapping.jordan_wigner(2 * num_active_orbitals)
    qubit_hamiltonian = qubit_mapper.run(active_hamiltonian, mapping)

    state_prep_builder = create("state_prep", algorithm_name="sparse_isometry_gf2x")
    state_preparation = state_prep_builder.run(active_wfn)

    evolution_builder = create("hamiltonian_unitary_builder", "trotter", time=_EVOLUTION_TIME, power=_OBSERVABLE_POWER)
    unitary = evolution_builder.run(qubit_hamiltonian)

    return HadamardWaterBenchmark(
        state_preparation=state_preparation,
        unitary=unitary,
    )


@pytest.mark.skipif(not _HAS_QSHARP, reason="Q# not available")
def test_qdk_hadamard_test_measures_water_observable(
    water_hadamard_benchmark: HadamardWaterBenchmark,
) -> None:
    """Q# Hadamard generator reproduces the reference observable for water."""
    result = _make_hadamard_test().run(
        water_hadamard_benchmark.state_preparation,
        water_hadamard_benchmark.unitary,
        shots=_SHOTS,
    )
    counts = result.bitstring_counts
    observable_value = (counts.get("0", 0) - counts.get("1", 0)) / sum(counts.values())

    assert np.isclose(observable_value, 0.34, atol=1e-12), (
        f"X-basis observable = {observable_value!r} over {sum(counts.values())} shots"
    )


@pytest.mark.skipif(not _HAS_QSHARP, reason="Q# not available")
def test_qdk_hadamard_test_measures_water_observable_in_y_basis(
    water_hadamard_benchmark: HadamardWaterBenchmark,
) -> None:
    """Q# Hadamard generator reproduces the Y-basis reference observable for water."""
    hadamard_test = _make_hadamard_test(HadamardTestBasis.Y)
    result = hadamard_test.run(
        water_hadamard_benchmark.state_preparation,
        water_hadamard_benchmark.unitary,
        shots=_SHOTS,
    )
    counts = result.bitstring_counts
    observable_value = (counts.get("0", 0) - counts.get("1", 0)) / sum(counts.values())

    assert np.isclose(observable_value, 0.98, atol=1e-12), (
        f"Y-basis observable = {observable_value!r} over {sum(counts.values())} shots"
    )


def test_hadamard_test_rejects_invalid_test_basis() -> None:
    """Setting an unsupported measurement basis is rejected."""
    hadamard_test = create("hadamard_test")
    with pytest.raises(ValueError, match="allowed options"):
        hadamard_test.settings().set("test_basis", "InvalidBasis")
    with pytest.raises(ValueError, match="allowed options"):
        hadamard_test.settings().set("test_basis", "Z")


@pytest.mark.skipif(not _HAS_QSHARP, reason="Q# not available")
def test_qdk_hadamard_test_rejects_incompatible_input_circuits(
    water_hadamard_benchmark: HadamardWaterBenchmark,
) -> None:
    """Q# generator raises errors when input circuits do not expose Q# operations."""
    generator = _make_hadamard_test()
    bad_state_preparation_circuit = Circuit(qasm="bad_state_preparation")

    with pytest.raises(ValueError, match="state_preparation"):
        generator.run(
            bad_state_preparation_circuit,
            water_hadamard_benchmark.unitary,
            shots=_SHOTS,
        )

    with pytest.raises(TypeError, match="UnitaryRepresentation"):
        generator.run(
            water_hadamard_benchmark.state_preparation,
            object(),
            shots=_SHOTS,
        )
