"""Structured-lowering regressions from PR707 and PR728 sparse/QPE integration checks."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import h5py
import numpy as np
import pytest
import scipy

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.circuit_mapper.pauli_sequence_mapper import PauliSequenceMapper
from qdk_chemistry.algorithms.controlled_circuit_mapper.controlled_pauli_sequence_mapper import (
    ControlledPauliSequenceMapper,
)
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.plaquette_trotter import PlaquetteTrotter
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator, UnitaryRepresentation
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    BatchedExponentiatedPauliTerm,
    ConjugatedExponentiatedPauliTerm,
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.data.unitary_representation.containers.sparse_pauli_product_formula import SparsePauliTerms
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, create_qsharp_context, get_qsharp_context

from .test_helpers import dense_matrix
from .test_plaquette_trotter import _hubbard_operator, _qsharp_groups

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit.quantum_info import Operator


class TestStructuredExponentiatedPauliTerms:
    """Tests for structural batching and conjugation metadata."""

    def test_batch_owns_one_angle_and_disjoint_pauli_strings(self):
        """A batch represents equal-angle factors without neighbour-dependent IDs."""
        batch = BatchedExponentiatedPauliTerm(pauli_terms=[{0: "Z"}, {2: "X", 3: "Y"}], angle=0.25)
        assert batch.pauli_terms == [{0: "Z"}, {2: "X", 3: "Y"}]
        assert batch.angle == 0.25

    @pytest.mark.parametrize(
        ("pauli_terms", "match"),
        [
            ([{0: "Z"}], "at least two"),
            ([{}, {1: "Z"}], "identity"),
            ([{0: "Z"}, {0: "X"}], "disjoint support"),
        ],
    )
    def test_batch_rejects_invalid_hamming_weight_groups(self, pauli_terms, match):
        """Singleton, identity, and overlapping groups cannot use one weight register."""
        with pytest.raises(ValueError, match=match):
            BatchedExponentiatedPauliTerm(pauli_terms=pauli_terms, angle=0.25)

    def test_conjugation_stores_within_and_apply_blocks(self):
        """The representation directly records V D V-dagger."""
        within = [ExponentiatedPauliTerm({0: "X"}, 0.2)]
        apply = [ExponentiatedPauliTerm({0: "Z"}, 0.3)]
        conjugated = ConjugatedExponentiatedPauliTerm(within_terms=within, apply_terms=apply)
        assert conjugated.within_terms == within
        assert conjugated.apply_terms == apply

    def test_outer_conjugation_is_distinct_from_the_repeated_step(self):
        """Moving a factor into the outer conjugation changes the represented circuit."""
        term = ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.25)
        conjugated = PauliProductFormulaContainer(step_terms=[], step_reps=3, num_qubits=1, conjugating_terms=[term])
        repeated = PauliProductFormulaContainer(step_terms=[term], step_reps=3, num_qubits=1)
        assert conjugated.conjugating_terms == [term]
        assert conjugated.content_hash() != repeated.content_hash()

    @pytest.mark.parametrize("file_format", ["json", "hdf5"])
    def test_structured_formula_persistence_fails_explicitly(self, file_format, tmp_path):
        """Persistence stays limited to the established flat wire format."""
        batch = BatchedExponentiatedPauliTerm([{0: "Z"}, {1: "Z"}], 0.4)
        container = PauliProductFormulaContainer(step_terms=[batch], step_reps=5, num_qubits=2)
        if file_format == "json":
            with pytest.raises(ValueError, match="Structured Pauli product formulas cannot be serialized"):
                container.to_json()
        else:
            with (
                h5py.File(tmp_path / "structured.h5", "w") as handle,
                pytest.raises(ValueError, match="Structured Pauli product formulas cannot be serialized"),
            ):
                container.to_hdf5(handle)

    @pytest.mark.parametrize("structured", ["batch", "conjugated", "outer"])
    def test_structured_composition_and_incompatible_mappers_reject(self, structured):
        """Do not silently drop boundaries or batching in flat-only consumers."""
        term = ExponentiatedPauliTerm({0: "Z"}, 0.2)
        batch = BatchedExponentiatedPauliTerm([{0: "Z"}, {1: "Z"}], 0.4)
        if structured == "outer":
            container = PauliProductFormulaContainer([term], 2, 2, conjugating_terms=[term])
        else:
            block = batch if structured == "batch" else ConjugatedExponentiatedPauliTerm([term], [batch])
            container = PauliProductFormulaContainer([block], 2, 2)
        flat = PauliProductFormulaContainer([term], 1, 2)
        for left, right in ((flat, container), (container, flat)):
            with pytest.raises(ValueError, match="batched or conjugated"):
                left.combine(right)
        mapper = create("controlled_circuit_mapper", "cswap_pauli_sequence", control_indices=[2])
        with pytest.raises(ValueError, match="pauli_sequence"):
            mapper.run(UnitaryRepresentation(container))
        reordered = container.reorder_terms([0])
        assert reordered.content_hash() == container.content_hash()
        assert reordered.conjugating_terms == container.conjugating_terms


class TestStructuredLowering:
    """PR707 ordinary and controlled matrix and resource regressions."""

    def test_outer_conjugation_surrounds_the_repeated_step(self):
        """A noncommuting conjugation executes once around the repeated body."""
        conjugating = ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.2)
        repeated = ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.3)
        container = PauliProductFormulaContainer(
            step_terms=[repeated], step_reps=3, num_qubits=1, conjugating_terms=[conjugating]
        )
        actual = dense_matrix(PauliSequenceMapper().run(UnitaryRepresentation(container))._qsharp_op, 1)
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        expected = (
            scipy.linalg.expm(1j * conjugating.angle * x)
            @ np.linalg.matrix_power(scipy.linalg.expm(-1j * repeated.angle * z), container.step_reps)
            @ scipy.linalg.expm(-1j * conjugating.angle * x)
        )
        assert np.max(np.abs(actual - expected)) < 1e-5

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_outer_conjugator_is_bare_but_the_repeated_body_is_controlled(self):
        """Q# within/apply controls only the body of the outer conjugation."""
        container = PauliProductFormulaContainer(
            step_terms=[ExponentiatedPauliTerm({0: "Z"}, 0.3)],
            step_reps=3,
            num_qubits=1,
            conjugating_terms=[ExponentiatedPauliTerm({0: "X"}, 0.2)],
        )
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [1])
        actual = Operator(mapper.run(UnitaryRepresentation(container)).get_qiskit_circuit()).data
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        conjugated = (
            scipy.linalg.expm(1j * 0.2 * x) @ scipy.linalg.expm(-1j * 0.9 * z) @ scipy.linalg.expm(-1j * 0.2 * x)
        )
        expected = np.kron(np.array([[1, 0], [0, 0]]), np.eye(2)) + np.kron(np.array([[0, 0], [0, 1]]), conjugated)
        assert np.max(np.abs(actual - expected)) < 1e-5

    def test_batched_outer_conjugator_is_not_controlled(self):
        """HWP remains useful in Campbell's one-time outer within block."""
        container = PauliProductFormulaContainer(
            step_terms=[],
            step_reps=3,
            num_qubits=8,
            conjugating_terms=[BatchedExponentiatedPauliTerm([{index: "Z"} for index in range(8)], 0.2)],
        )
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [8])
        counts = mapper.run(UnitaryRepresentation(container)).estimate()["logicalCounts"]
        assert counts["rotationCount"] == 8

    def test_controlled_hwp_preserves_relative_phase(self):
        """Control-off identity and control-on phases agree, including HWP scalar compensation."""
        context = create_qsharp_context()
        group = BatchedExponentiatedPauliTerm([{index: "Z"} for index in range(8)], 0.31)
        context.eval("use qs = Qubit[9]; for q in qs { H(q); }")
        context.eval(
            "Controlled QDKChemistry.Utils.PauliExp.SparsePauliExpGroups"
            f"([qs[0]], ({_qsharp_groups([group])}, qs[1...]));"
        )
        actual = np.asarray(context.dump_machine().as_dense_state())
        expected = np.ones(512, dtype=complex) / np.sqrt(512)
        expected[256:] *= np.exp(-1j * 0.31 * np.array([8 - 2 * bits.bit_count() for bits in range(256)]))
        assert np.allclose(actual, expected, atol=1e-9)
        context.eval("ResetAll(qs);")


@pytest.mark.parametrize("side", [2, 4])
@pytest.mark.parametrize("automatic_divisions", [False, True])
def test_sparse_repeat_powers_and_standard_qpe10(
    side: int, automatic_divisions: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the requested timing recipe without label expansion or power rescaling."""
    dense_operator = _hubbard_operator(side, side, interaction=8.0)
    operator = QubitOperator.from_sparse_terms(
        dense_operator.num_qubits,
        (word for word, _ in dense_operator.iter_sparse_terms()),
        dense_operator.coefficients,
        encoding=dense_operator.encoding,
        fermion_mode_order=dense_operator.fermion_mode_order,
        term_partition=dense_operator.term_partition,
        tapering=dense_operator.tapering,
    )
    assert isinstance(operator.pauli_strings, SparsePauliTerms)

    def no_labels(_self, _index):
        pytest.fail("The plaquette builder must not densify sparse Pauli labels.")

    monkeypatch.setattr(SparsePauliTerms, "__getitem__", no_labels)
    total_time = 1.0 / (0.0051 * side**2)
    base = total_time / 512
    settings = {
        "order": 2,
        "time": base,
        "target_accuracy": 0.0051 * side**2 if automatic_divisions else 0.0,
        "num_divisions": 3,
        "lattice_width": side,
        "lattice_height": side,
        "power_strategy": "repeat",
    }
    first = PlaquetteTrotter(**settings).run(operator).get_container()
    if not automatic_divisions:
        assert first.step_reps == settings["num_divisions"]
    for power in (2, 512):
        powered = PlaquetteTrotter(**settings, power=power).run(operator).get_container()
        assert powered.step_reps == power * first.step_reps
        assert powered.step_terms == first.step_terms
        assert powered.conjugating_terms == first.conjugating_terms
        assert powered.scale == first.scale == base
    state_prep_params = {"bitStrings": [0] * operator.num_qubits, "numQubits": operator.num_qubits}
    state_prep = Circuit(
        qsharp_factory=QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeSingleReferenceStateCircuit,
            parameter=state_prep_params,
        ),
        qsharp_op=QSHARP_UTILS.StatePreparation.MakePrepareSingleReferenceStateOp(state_prep_params),
        num_qubits=operator.num_qubits,
    )
    builder = create(
        "qpe_circuit_builder",
        "qdk_standard",
        num_bits=10,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "plaquette", **settings),
        controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
    )
    circuits = builder.run(state_preparation=state_prep, qubit_hamiltonian=operator)
    assert len(circuits) == 1
    factory = circuits[0]._qsharp_factory
    counts = get_qsharp_context().logical_counts(factory.program, *factory.parameter.values())
    assert counts["numQubits"] >= operator.num_qubits + 10
    assert counts["rotationCount"] > 0
