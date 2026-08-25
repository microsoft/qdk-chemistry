"""Tests for alias sampling and QROM state preparation as PREPARE oracles inside PSP."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.lcu import LCUBuilder
from qdk_chemistry.algorithms.state_preparation.alias_sampling import AliasSamplingStatePreparation
from qdk_chemistry.algorithms.state_preparation.dense_pure_state import DensePureStatePreparation
from qdk_chemistry.algorithms.state_preparation.qrom_state_prep import QROMStatePreparation
from qdk_chemistry.algorithms.state_preparation.state_preparation import PrepareLayout
from qdk_chemistry.data import (
    AlgorithmRef,
    Configuration,
    ModelOrbitals,
    QubitOperator,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.utils.qsharp import create_qsharp_context

# H = 0.25 X + 0.75 Z on one qubit. lambda = 1, so the LCU probabilities are 0.25 and 0.75,
# both exactly representable in mu = 2 keep bits, which makes the alias table lossless and
# lets the block-encoding assertion below use a tight tolerance.
EXACT_COEFFS = [0.25, 0.75]

# QROM discretizes its rotation angles, so its Hamiltonian is chosen for a wide index
# register and distinct coefficients rather than for exactness.
QROM_COEFFS = [0.5, 0.3, 0.2]

PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]])
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]])


def _make_wavefunction(amplitudes: list[float]) -> Wavefunction:
    """Create a Wavefunction whose determinant ``idx`` is position ``idx``."""
    num_qubits = math.ceil(math.log2(len(amplitudes))) if len(amplitudes) > 1 else 1
    dets = [Configuration.from_bitstring(format(idx, f"0{num_qubits}b")) for idx in range(len(amplitudes))]
    container = StateVectorContainer(np.array([float(a) for a in amplitudes]), dets, ModelOrbitals(num_qubits))
    return Wavefunction(container)


def _psp_mapper(prepare_name: str):
    """Build a PSP mapper whose PREPARE oracle is *prepare_name*."""
    return create("circuit_mapper", "prepare_select_prepare", prepare=AlgorithmRef("state_prep", prepare_name))


def _lcu(pauli_strings, coefficients, *, power=1):
    """Build an LCU unitary representation from Pauli strings and coefficients."""
    hamiltonian = QubitOperator(pauli_strings=pauli_strings, coefficients=np.array(coefficients))
    return LCUBuilder(power=power).run(hamiltonian)


def _block_of(context, num_system: int, num_ancilla: int) -> np.ndarray:
    """Return the ancilla-|0> block of the current simulator state."""
    dense = np.array(context.dump_machine().as_dense_state())
    return dense[:: 1 << num_ancilla][: 1 << num_system]


def _up_to_global_phase(vector: np.ndarray) -> np.ndarray:
    """Rotate *vector* so its largest entry is real and positive."""
    return vector * np.exp(-1j * np.angle(vector[np.argmax(np.abs(vector))]))


def _expected_block(paulis: list[np.ndarray], coefficients: list[float], theta: float) -> np.ndarray:
    """Return ``H|psi>/lambda`` for ``H = sum(coefficients * paulis)`` and ``|psi> = Ry(theta)|0>``."""
    hamiltonian = sum(coefficient * pauli for coefficient, pauli in zip(coefficients, paulis, strict=True))
    return hamiltonian @ np.array([np.cos(theta / 2), np.sin(theta / 2)]) / sum(coefficients)


class TestPrepareLayout:
    """Each oracle has to describe the register it needs before it can be embedded."""

    def test_default_layout_matches_the_lcu_ancilla_count(self):
        """A pure state preparation indexes exactly the register the LCU sized for it.

        This is the invariant that keeps every pre-existing block encoding byte-identical:
        the default layout has to reproduce ``LCUContainer.num_prepare_ancillas`` exactly.
        """
        lcu, _ = _psp_mapper("dense_pure_state").resolve_lcu(_lcu(["XX", "ZZ", "XZ"], [0.25, 0.5, 0.1]).get_container())
        layout = DensePureStatePreparation().prepare_layout(lcu.prepare)

        assert layout == PrepareLayout(num_select_qubits=2, num_block_ancillas=2, num_shared_ancillas=0)
        assert layout.num_select_qubits == lcu.num_prepare_ancillas

    def test_alias_layout_widens_the_block_ancilla_register(self):
        """Alias sampling indexes n qubits but leaves 2n + 2mu + 1 of them entangled.

        SELECT must still see only the n index qubits; the rest is garbage that
        ``PREPARE``:sup:`dagger` uncomputes.
        """
        layout = AliasSamplingStatePreparation(bits_precision=4).prepare_layout(
            _make_wavefunction([0.5, 0.3, 0.7, 0.1])
        )

        assert layout == PrepareLayout(num_select_qubits=2, num_block_ancillas=2 * 2 + 2 * 4 + 1)

    def test_qrom_layout_requests_a_shared_gradient(self):
        """QROM's phase gradient is shared ancilla, not block ancilla.

        It is left in |phi> between uses rather than |0>, so a qubitization walk must not
        reflect about it.
        """
        layout = QROMStatePreparation(rotation_bit_precision=6).prepare_layout(_make_wavefunction([0.5, 0.3, 0.7, 0.1]))

        assert layout == PrepareLayout(num_select_qubits=2, num_block_ancillas=2, num_shared_ancillas=6)


class TestAliasSamplingConvention:
    """Alias sampling samples its input, so the input decides which state comes out."""

    def test_coefficients_are_squared_before_reaching_qsharp(self):
        """The class prepares amplitudes c/||c||, which means feeding Q# the squares.

        Q# samples p proportional to whatever it is handed, so handing it |c| would prepare
        sqrt(|c|/sum|c|) and silently disagree with every other state preparation in the
        package, including the sqrt(alpha/lambda) vector the LCU builder produces.
        """
        coefficients = [0.5, 0.3, 0.7, 0.1]
        circuit = AliasSamplingStatePreparation(bits_precision=4).run(_make_wavefunction(coefficients))

        passed = circuit._qsharp_factory.parameter["coefficients"]
        np.testing.assert_allclose(passed, np.square(coefficients))


class TestPSPMapperWithPrepareOracles:
    """The mapper has to size the register from the layout, not from the index width."""

    def test_alias_sampling_widens_the_block_encoding_register(self):
        """The mapper has to size the block ancilla from the layout, not from the index width.

        Sizing it from the index width is not a crash, it is an out-of-range read inside the
        Q# oracle, so the assertion is on the width the mapper actually requests.
        """
        circuit = _psp_mapper("alias_sampling").run(_lcu(["XX", "ZZ", "XZ"], [0.25, 0.5, 0.1]))

        num_index, mu = 2, 10
        parameters = circuit._qsharp_factory.parameter
        assert parameters["numSelectQubits"] == num_index
        assert parameters["numBlockAncillaQubits"] == 2 * num_index + 2 * mu + 1
        assert parameters["numSharedQubits"] == 0

    def test_qrom_adds_a_shared_gradient_register(self):
        """The gradient is allocated alongside the block ancilla, not inside it.

        A qubitization walk reflects about the block ancilla, and the gradient is left in
        |phi> rather than |0>, so putting it there would corrupt the reflection.
        """
        circuit = _psp_mapper("qrom").run(_lcu(["XX", "ZZ", "XZ"], [0.25, 0.5, 0.1]))

        num_index, b_rot = 2, 10
        parameters = circuit._qsharp_factory.parameter
        assert parameters["numSelectQubits"] == num_index
        assert parameters["numBlockAncillaQubits"] == num_index
        assert parameters["numSharedQubits"] == b_rot

    def test_gradient_cost_does_not_scale_with_power(self):
        """Hoisting the gradient out of the power loop is the whole point of sharing it.

        Every arbitrary-angle rotation QROM state preparation reports comes from preparing
        the phase gradient; the state-prep body itself contributes none. Preparing it once
        per block encoding therefore multiplies the rotation count by the power, which is
        exactly what a phase estimation schedule does not want.
        """
        rotations = {}
        for power in (1, 4):
            circuit = _psp_mapper("qrom").run(_lcu(["XX", "ZZ", "XZ"], [0.25, 0.5, 0.1], power=power))
            rotations[power] = circuit.estimate()["logicalCounts"]["rotationCount"]

        assert rotations[4] == rotations[1], f"gradient was re-prepared per power: {rotations}"

    def test_index_width_mismatch_is_rejected(self, monkeypatch):
        """A PREPARE oracle that indexes a different register than SELECT is a hard error.

        The two disagree when a state preparation indexes coefficients by list position
        while the decomposition indexes them by determinant bit pattern. Nothing downstream
        would raise: SELECT would just pair coefficients with the wrong terms.
        """
        unitary = _lcu(["XX", "ZZ", "XZ"], [0.25, 0.5, 0.1])
        monkeypatch.setattr(
            DensePureStatePreparation,
            "prepare_layout",
            lambda *_: PrepareLayout(num_select_qubits=5, num_block_ancillas=5),
        )

        with pytest.raises(ValueError, match="indexes 5 qubits"):
            _psp_mapper("dense_pure_state").run(unitary)


class TestBlockEncodingIdentity:
    """Both oracles have to reproduce H/lambda once embedded, not just fit."""

    def test_alias_sampling_block_encodes_the_hamiltonian(self):
        """Alias sampling's garbage stays out of SELECT and is uncomputed by PREPARE-dagger.

        The block carries a global phase set by the measurement outcomes of the uncompute,
        so the comparison is up to global phase.
        """
        mu, theta = 2, 0.7
        num_ancilla = 2 * 1 + 2 * mu + 1
        probabilities = [c / sum(EXACT_COEFFS) for c in EXACT_COEFFS]
        prepare = (
            "QDKChemistry.Utils.AliasSampling.MakeAliasSamplingOp("
            "new QDKChemistry.Utils.AliasSampling.AliasSamplingParams {"
            f"coefficients = {probabilities}, bitsPrecision = {mu}, "
            f"numIndexQubits = 1, numQubits = {num_ancilla} }})"
        )
        context = create_qsharp_context()
        context.eval(f"use register = Qubit[{1 + num_ancilla}]; Ry({theta}, register[0]);")
        context.eval(
            "QDKChemistry.Utils.PrepSelPrep.PrepSelPrep("
            f"{prepare}, "
            "QDKChemistry.Utils.Select.MakeSelectOp("
            "new QDKChemistry.Utils.Select.PauliSelectParams {"
            "pauliTerms = [[PauliX], [PauliZ]], signs = [1, 1], controlStates = [0, 1] }), "
            "register[0..0], register[1...], 1);"
        )

        block = _block_of(context, num_system=1, num_ancilla=num_ancilla)
        expected = _expected_block([PAULI_X, PAULI_Z], EXACT_COEFFS, theta)

        np.testing.assert_allclose(_up_to_global_phase(block), _up_to_global_phase(expected), atol=1e-9)

    def test_qrom_block_encodes_the_hamiltonian_with_a_shared_gradient(self):
        """QROM's index has to be reversed to match the bit order SELECT controls on.

        ``ApplyControlledOnInt`` reads its control register least significant qubit first,
        while QROM writes ``amplitudes[l]`` most significant qubit first. Without the
        reversal this still runs and still produces a unitary, it just attaches the
        coefficients to the wrong Pauli terms, which is why the three coefficients here are
        distinct and the term count is above two.
        """
        b_rot, theta = 12, 0.7
        amplitudes = [c**0.5 for c in QROM_COEFFS] + [0.0]
        gradient = f"register[3..{2 + b_rot}]"
        context = create_qsharp_context()
        context.eval(f"use register = Qubit[{3 + b_rot}]; Ry({theta}, register[0]);")
        context.eval(f"QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState({gradient});")
        context.eval(
            "QDKChemistry.Utils.PrepSelPrep.PrepSelPrep("
            "QDKChemistry.Utils.QROMStatePrep.MakeQROMStatePrepOpShared("
            "new QDKChemistry.Utils.QROMStatePrep.QROMStatePrepParams {"
            f"amplitudes = {amplitudes}, rotationBitPrecision = {b_rot}, numStateQubits = 2 }}), "
            "QDKChemistry.Utils.Select.MakeSelectOp("
            "new QDKChemistry.Utils.Select.PauliSelectParams {"
            "pauliTerms = [[PauliX], [PauliZ], [PauliY]], signs = [1, 1, 1], "
            "controlStates = [0, 1, 2] }), "
            "register[0..0], register[1...], 2);"
        )
        context.eval(f"Adjoint QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState({gradient});")

        block = _block_of(context, num_system=1, num_ancilla=2 + b_rot)
        expected = _expected_block([PAULI_X, PAULI_Z, PAULI_Y], QROM_COEFFS, theta)

        # Measured 5.4e-4 at bRot = 12 and 3.1e-3 at bRot = 10, so this accepts the
        # requested precision and rejects one step below it.
        np.testing.assert_allclose(_up_to_global_phase(block), _up_to_global_phase(expected), atol=2e-3)
