"""QDK/Chemistry energy estimator abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3
from qiskit.quantum_info import Pauli, PauliList

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, EnergyExpectationResult, QubitHamiltonian
from qdk_chemistry.utils import Logger

__all__: list[str] = []


def _parity(integer: int) -> int:
    """Return the parity of an integer."""
    return integer.bit_count() % 2


def _paulis_to_indices(paulis: PauliList) -> list[int]:
    """Converts a list of Pauli operators into a list of integer indices.

    Example:
        PauliList(["IZ", "ZX", "YZ", "ZY"]) -> [1, 3, 3, 3]

    Args:
        paulis: ``PauliList`` representing Pauli operators.

    Returns:
        A list of integer indices corresponding to the non-identity components of the Pauli operators.

    """
    nonid = paulis.z | paulis.x
    packed_vals = np.packbits(nonid, axis=1, bitorder="little").astype(object)
    power_uint8 = 1 << (8 * np.arange(packed_vals.shape[1], dtype=object))
    inds = packed_vals @ power_uint8
    return inds.tolist()


def _compute_expval_and_variance_from_bitstrings(
    bitstring_counts: dict[str, int], paulis: PauliList
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the expectation values and variances for a given set of Pauli operators.

    Args:
        bitstring_counts: A dictionary of measurement outcomes.
        paulis: ``PauliList`` representing Pauli operators for computing expectation values.

    Returns:
        A tuple containing expectation values and variances.

    """
    Logger.trace_entering()
    # Determine measurement basis and Pauli contains only measured terms (drop I terms)
    basis = _determine_measurement_basis(paulis)
    measured_indices = np.where(basis.z | basis.x)[0]
    measured_paulis = PauliList.from_symplectic(paulis.z[:, measured_indices], paulis.x[:, measured_indices])

    diag_inds = _paulis_to_indices(measured_paulis)
    expvals = np.zeros(len(measured_paulis), dtype=float)
    nshots = sum(bitstring_counts.values())
    if nshots == 0:
        raise ValueError("Bitstring counts are empty.")

    for bitstr, freq in bitstring_counts.items():
        try:
            outcome = int(bitstr, 16) if bitstr.startswith("0x") else int(bitstr, 2)
        except ValueError as err:
            raise ValueError(f"Unsupported bitstring format: {bitstr}") from err
        for i, mask in enumerate(diag_inds):
            expvals[i] += freq * (-1) ** _parity(mask & outcome)

    expvals /= nshots
    variances = (1 - expvals**2) / nshots
    return expvals, variances


def _determine_measurement_basis(paulis: PauliList) -> Pauli:
    """Determine the measurement basis for a group of Pauli operators.

    Example: PauliList(["IZ", "YZ"]) -> Pauli("YZ")

    Args:
        paulis: A list of ``PauliList`` representing Pauli operators.

    Returns:
        A ``Pauli`` representing the measurement basis.

    """
    qubit_wise_grouped_paulis = paulis.group_commuting(qubit_wise=True)
    if len(qubit_wise_grouped_paulis) != 1:
        raise ValueError(
            "Paulis are not qubit-wise commuting. Please group them first to generate a valid measurement basis."
        )
    paulis = qubit_wise_grouped_paulis[0]

    return Pauli((np.logical_or.reduce(paulis.z), np.logical_or.reduce(paulis.x)))


def _build_measurement_circuit(basis: Pauli) -> QuantumCircuit:
    """Generate a circuit with measurement operations for a given Pauli operator.

    Args:
        basis: ``Pauli`` operator defining the measurement basis.

    Returns:
        A ``QuantumCircuit`` representing the measurement circuit for the given Pauli operator.

    """
    # Locate active qubits by binary symplectic representation of the Pauli basis
    # I: x=0, z=0, X: x=1, z=0, Z: x=0, z=1, Y: x=1, z=1
    active = np.arange(basis.num_qubits)[basis.z | basis.x]
    qreg = QuantumRegister(basis.num_qubits, "q")
    creg = ClassicalRegister(len(active), "c")
    qc = QuantumCircuit(qreg, creg)

    for cidx, qidx in enumerate(active):
        if basis.x[qidx]:
            if basis.z[qidx]:
                qc.sdg(qreg[qidx])  # If x=1 and z=1, Y basis
            qc.h(qreg[qidx])  # If x=1 and z=0, X basis
        qc.measure(qreg[qidx], creg[cidx])
    return qc


class EnergyEstimator(Algorithm):
    """Abstract base class for energy estimator algorithms."""

    def __init__(self):
        """Initialize the EnergyEstimator."""
        super().__init__()

    def type_name(self) -> str:
        """Return ``energy_estimator`` as the algorithm type name."""
        return "energy_estimator"

    def run(
        self,
        circuit: Circuit,
        qubit_hamiltonians: list[QubitHamiltonian],
        total_shots: int,
        noise_model: Any | None = None,
        classical_coeffs: list | None = None,
    ) -> EnergyExpectationResult:
        """Estimate the expectation value and variance of Hamiltonians.

        Args:
            circuit: Circuit that provides an OpenQASM3 string of the quantum circuit to be evaluated.
            qubit_hamiltonians: List of ``QubitHamiltonian`` to estimate.
            total_shots: Total number of shots to allocate across the observable terms.
            noise_model: Optional noise model to simulate noise in the quantum circuit.
            classical_coeffs: Optional list of coefficients for classical Pauli terms to calculate energy offset.

        Returns:
            ``EnergyExpectationResult`` containing the energy expectation value and variance.

        Note:
            * Measurement circuits are generated for each QubitHamiltonian term.
            * Parameterized circuits are not supported.
            * Only one circuit is supported per run.

        """
        # This function definition is not required it is present to add type hints and docstrings
        #  for the derived classes specialized run() method.
        return super().run(circuit, qubit_hamiltonians, total_shots, noise_model, classical_coeffs)

    @staticmethod
    def _create_measurement_circuits(circuit: Circuit, grouped_hamiltonians: list[QubitHamiltonian]) -> list[Circuit]:
        """Create measurement circuits for each QubitHamiltonian.

        Args:
            circuit: Circuit that provides an OpenQASM3 string of the base circuit.
            grouped_hamiltonians: List of ``QubitHamiltonian`` grouped in qubit-wise commuting sets.

        Returns:
            List of Circuits that provide the measurement circuits in OpenQASM3 format.

        """
        Logger.trace_entering()
        meas_circuits = []
        base_circuit = qasm3.loads(circuit.get_qasm())

        if base_circuit.num_qubits != grouped_hamiltonians[0].num_qubits:
            raise ValueError(
                f"Number of qubits in the base circuit ({base_circuit.num_qubits}) does not match "
                f"the number of qubits in the Hamiltonian ({grouped_hamiltonians[0].num_qubits})."
            )

        for hamiltonian in grouped_hamiltonians:
            basis = _determine_measurement_basis(hamiltonian.pauli_ops.paulis)
            meas_ops = _build_measurement_circuit(basis)
            full_circ = base_circuit.compose(meas_ops, inplace=False)
            meas_circuits.append(Circuit(qasm=qasm3.dumps(full_circ)))

        return meas_circuits

    @staticmethod
    def _compute_energy_expectation_from_bitstrings(
        hamiltonians: list[QubitHamiltonian],
        bitstring_counts_list: list[dict[str, int] | None],
        energy_offset: float = 0.0,
    ) -> EnergyExpectationResult:
        """Compute total energy expectation value and variance for a QubitHamiltonian.

        Args:
            hamiltonians: List of ``QubitHamiltonian`` defining Pauli terms and coefficients.
            bitstring_counts_list: List of bitstring count dictionaries corresponding to each QubitHamiltonian.
            energy_offset: Optional energy shift to include.

        Returns:
            ``EnergyExpectationResult`` containing the energy expectation value and variance.

        """
        Logger.trace_entering()
        if len(bitstring_counts_list) != len(hamiltonians):
            raise ValueError(f"Expected {len(hamiltonians)} bitstring result sets, got {len(bitstring_counts_list)}.")

        total_expval = 0.0
        total_var = 0.0
        expvals_list, vars_list = [], []

        for counts, group in zip(bitstring_counts_list, hamiltonians, strict=True):
            if counts is None:
                continue
            paulis = group.pauli_ops.paulis
            coeffs = group.pauli_ops.coeffs

            expvals, variances = _compute_expval_and_variance_from_bitstrings(counts, paulis)
            expvals_list.append(expvals)
            vars_list.append(variances)

            total_expval += np.dot(expvals, coeffs)
            total_var += np.dot(variances, np.abs(coeffs) ** 2)

        return EnergyExpectationResult(
            energy_expectation_value=float(np.real_if_close(total_expval + energy_offset)),
            energy_variance=float(np.real_if_close(total_var)),
            expvals_each_term=expvals_list,
            variances_each_term=vars_list,
        )


class EnergyEstimatorFactory(AlgorithmFactory):
    """Factory class for creating EnergyEstimator instances."""

    def algorithm_type_name(self) -> str:
        """Return ``energy_estimator`` as the algorithm type name."""
        return "energy_estimator"

    def default_algorithm_name(self) -> str:
        """Return ``qdk_base_simulator`` as the default algorithm name."""
        return "qdk_base_simulator"
