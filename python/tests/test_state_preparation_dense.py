"""Tests for the DensePureStatePreparation algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import pytest
from qdk import qsharp

from qdk_chemistry.algorithms import create, registry
from qdk_chemistry.algorithms.state_preparation.dense_pure_state import DensePureStatePreparation
from qdk_chemistry.data import Circuit, Configuration, StateVectorContainer, Wavefunction
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT, QDK_CHEMISTRY_HAS_QISKIT_AER
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .reference_tolerances import estimator_energy_tolerance, float_comparison_absolute_tolerance
from .test_helpers import create_test_orbitals

try:
    from qdk._native import Circuit as QdkCircuitType
except ImportError:
    from qsharp._native import Circuit as QdkCircuitType


def _run_state_prep_and_dump(circuit: Circuit) -> np.ndarray:
    """Run a state preparation circuit via Q# eval and return the statevector.

    Reinitializes the Q# interpreter to ensure a clean qubit state,
    reloads the StatePreparation Q# sources, allocates qubits, applies
    the state preparation, and captures the statevector via
    ``qsharp.dump_machine()``.

    Returns:
        The dense statevector as a complex numpy array (Q# big-endian ordering).

    """
    # Re-initialize to clear any stale qubits from prior calls
    qsharp.init()
    # Trigger lazy reload of Q# sources after interpreter reset
    _ = QSHARP_UTILS.StatePreparation

    params = circuit._qsharp_factory.parameter
    row_map_str = str(params["rowMap"])
    sv_str = "[" + ", ".join(f"{v:.16f}" for v in params["stateVector"]) + "]"
    n_qubits = params["numQubits"]
    exp_ops_str = str(params["expansionOps"])

    params_expr = (
        f"new QDKChemistry.Utils.StatePreparation.StatePreparationParams {{"
        f" rowMap = {row_map_str},"
        f" stateVector = {sv_str},"
        f" expansionOps = {exp_ops_str},"
        f" numQubits = {n_qubits} }}"
    )

    qsharp.eval(f"use qs = Qubit[{n_qubits}];")
    qsharp.eval(f"QDKChemistry.Utils.StatePreparation.StatePreparation({params_expr}, qs);")

    state = qsharp.dump_machine()
    return np.array(state.as_dense_state())


def _build_expected_statevector(
    coeffs: np.ndarray,
    dets: list[Configuration],
    num_orbitals: int,
) -> np.ndarray:
    """Build a normalized expected statevector matching Q# dump_machine output.

    Q# dump_machine uses little-endian (qubit k = bit k of index).
    With reversed rowMap, to_bits()[k] maps to Q# qubit (n-1-k), so
    dump_machine index is the little-endian interpretation of to_bits().
    """
    n_qubits = 2 * num_orbitals
    expected = np.zeros(2**n_qubits, dtype=complex)
    for coeff, det in zip(coeffs, dets, strict=True):
        bits = det.to_bits()
        # bits is [alpha_0,...,alpha_{N-1}, beta_0,...,beta_{N-1}]
        # dump_machine index = LE interpretation: qubit k = bit k
        index = 0
        for i, b in enumerate(bits):
            index |= b << i
        expected[index] = coeff
    norm = np.linalg.norm(expected)
    if norm > 0:
        expected /= norm
    return expected


class TestDensePureStatePreparation:
    """Tests for the DensePureStatePreparation algorithm."""

    def test_name_and_type(self):
        """Test that name and type_name return correct values."""
        prep = DensePureStatePreparation()
        assert prep.name() == "dense_pure_state"
        assert prep.type_name() == "state_prep"

    def test_registered_in_registry(self):
        """Verify dense pure state preparation is accessible via the registry."""
        prep = registry.create("state_prep", "dense_pure_state")
        assert isinstance(prep, DensePureStatePreparation)

    def test_basic_circuit_creation(self, wavefunction_4e4o):
        """Test that run produces a valid Circuit with a Q# op and factory."""
        prep = DensePureStatePreparation()
        circuit = prep.run(wavefunction_4e4o)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_circuit_has_correct_qubit_count(self, wavefunction_4e4o):
        """Test that the circuit uses 2*num_orbitals qubits (alpha + beta)."""
        prep = DensePureStatePreparation()
        circuit = prep.run(wavefunction_4e4o)

        qsc = circuit.get_qsharp_circuit()
        assert isinstance(qsc, QdkCircuitType)
        qsc_json = json.loads(qsc.json())
        num_qubits = len(qsc_json["qubits"])
        # 4 orbitals -> 8 qubits (4 alpha + 4 beta)
        assert num_qubits == 8

    def test_statevector_matches_wavefunction_4e4o(self, wavefunction_4e4o):
        """Verify the prepared state matches the expected statevector for the 4e4o problem.

        The wavefunction has two determinants:
          - "2200" (coeffs[0] = -0.9838) -> alpha "1100", beta "1100"
          - "2020" (coeffs[1] =  0.1793) -> alpha "1010", beta "1010"
        """
        prep = DensePureStatePreparation()
        circuit = prep.run(wavefunction_4e4o)
        actual_sv = _run_state_prep_and_dump(circuit)

        coeffs = np.array([-0.9837947571031265, 0.17929828748875612])
        dets = [Configuration.from_spin_half_string("2200"), Configuration.from_spin_half_string("2020")]
        expected = _build_expected_statevector(coeffs, dets, num_orbitals=4)

        # Compare up to global phase: |<actual|expected>| should be 1
        fidelity = abs(np.dot(np.conj(actual_sv), expected))
        assert np.isclose(fidelity, 1.0, atol=1e-6)

    def test_statevector_matches_wavefunction_10e6o(self, wavefunction_10e6o):
        """Verify the prepared state matches the expected statevector for the 10e6o F2 problem.

        The wavefunction has three determinants:
          - "222220" (coeffs[0] = -0.9731) -> alpha "111110", beta "111110"
          - "220222" (coeffs[1] =  0.2261) -> alpha "110111", beta "110111"
          - "222202" (coeffs[2] =  0.0438) -> alpha "111101", beta "111101"
        """
        prep = DensePureStatePreparation()
        circuit = prep.run(wavefunction_10e6o)
        actual_sv = _run_state_prep_and_dump(circuit)

        coeffs = np.array([-0.9731147049456421, 0.22612369393111892, 0.04377037881377919])
        dets = [
            Configuration.from_spin_half_string("222220"),
            Configuration.from_spin_half_string("220222"),
            Configuration.from_spin_half_string("222202"),
        ]
        expected = _build_expected_statevector(coeffs, dets, num_orbitals=6)

        fidelity = abs(np.dot(np.conj(actual_sv), expected))
        assert np.isclose(fidelity, 1.0, atol=1e-6)

    def test_config_from_bitstring(self):
        """Verify state preparation for configurations created from bitstrings."""
        test_orbitals = create_test_orbitals(2)
        det = Configuration.from_bitstring("11")
        dets = [det]
        coeffs = [1.0]

        container = StateVectorContainer(coeffs, dets, test_orbitals)
        wavefunction = Wavefunction(container)

        prep = DensePureStatePreparation()
        circuit = prep.run(wavefunction)
        actual_sv = _run_state_prep_and_dump(circuit)

        # Build expected statevector for 1-bit-per-mode config using to_bits().
        bits = det.to_bits()
        n_qubits = len(bits)
        index = 0
        for b in bits:
            index = (index << 1) | b
        expected = np.zeros(2**n_qubits, dtype=complex)
        expected[index] = 1.0

        fidelity = abs(np.dot(np.conj(actual_sv), expected))
        assert np.isclose(fidelity, 1.0, atol=1e-6)

    def test_single_determinant(self):
        """Verify state preparation for a single-determinant wavefunction."""
        test_orbitals = create_test_orbitals(2)
        det = Configuration.from_spin_half_string("du")
        dets = [det]
        coeffs = [1.0]

        container = StateVectorContainer(coeffs, dets, test_orbitals)
        wavefunction = Wavefunction(container)

        prep = DensePureStatePreparation()
        circuit = prep.run(wavefunction)
        actual_sv = _run_state_prep_and_dump(circuit)
        expected = _build_expected_statevector(np.array([1.0]), [det], num_orbitals=2)

        fidelity = abs(np.dot(np.conj(actual_sv), expected))
        assert np.isclose(fidelity, 1.0, atol=1e-6)

    @pytest.mark.skipif(
        not QDK_CHEMISTRY_HAS_QISKIT_AER or not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit Aer not available."
    )
    def test_energy_matches_sparse_isometry(self, wavefunction_4e4o, hamiltonian_4e4o, ref_energy_4e4o):
        """Verify that dense preparation yields the same energy as sparse isometry."""
        from qiskit.quantum_info import SparsePauliOp  # noqa: PLC0415
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator  # noqa: PLC0415

        # Re-init with Base profile required for QIR compilation (get_qiskit_circuit).
        # Load only StatePreparation.qs (the full Q# project includes files
        # with dynamic bools that are incompatible with Base profile).
        qsharp.init(target_profile=qsharp.TargetProfile.Base)
        _qs_src = Path(__file__).resolve().parent.parent / "src" / "qdk_chemistry" / "utils" / "qsharp" / "src"
        qsharp.eval((_qs_src / "StatePreparation.qs").read_text())

        dense_prep = create("state_prep", "dense_pure_state")
        sparse_prep = create("state_prep", "sparse_isometry_gf2x")

        # Dense prep with reversed rowMap produces a Qiskit circuit with reversed
        # qubit ordering; reverse_bits() aligns it with the Hamiltonian convention.
        dense_circuit = dense_prep.run(wavefunction_4e4o).get_qiskit_circuit().reverse_bits()
        sparse_circuit = sparse_prep.run(wavefunction_4e4o).get_qiskit_circuit()

        hamiltonian_op = SparsePauliOp(hamiltonian_4e4o.pauli_strings, hamiltonian_4e4o.coefficients)

        estimator = AerEstimator()
        dense_energy = estimator.run([(dense_circuit, hamiltonian_op)]).result()[0].data.evs
        sparse_energy = estimator.run([(sparse_circuit, hamiltonian_op)]).result()[0].data.evs

        assert np.isclose(dense_energy, ref_energy_4e4o, atol=estimator_energy_tolerance)
        assert np.isclose(dense_energy, sparse_energy, atol=float_comparison_absolute_tolerance)

        # Reset to default profile so subsequent tests are not affected
        qsharp.init()
