"""QDK/Chemistry energy estimator abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import qsharp

from qdk_chemistry._core.utils import pauli_string_to_masks
from qdk_chemistry.algorithms import CircuitExecutor
from qdk_chemistry.data import (
    Circuit,
    EnergyExpectationResult,
    MeasurementData,
    QuantumErrorProfile,
    QubitHamiltonian,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .energy_estimator import EnergyEstimator

__all__: list[str] = ["QdkEnergyEstimator"]


def _parity(integer: int) -> int:
    """Return the parity of an integer."""
    return integer.bit_count() % 2


def _paulis_to_nonid_masks(pauli_strings: list[str]) -> list[int]:
    """Converts a list of Pauli operators into a list of non-identity bitmasks.

    Example:
        ["IZ", "ZX", "YZ", "ZY"] -> [1, 3, 3, 3]

    Args:
        pauli_strings: List of Pauli label strings.

    Returns:
        A list of integer bitmasks (``x_mask | z_mask``) for each operator.

    """
    return [x | z for x, z, _ in (pauli_string_to_masks(ps) for ps in pauli_strings)]


def _compute_expval_and_variance_from_bitstrings(
    bitstring_counts: dict[str, int], pauli_strings: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the expectation values and variances for a given set of Pauli operators.

    Args:
        bitstring_counts: A dictionary of measurement outcomes.
        pauli_strings: List of Pauli label strings for computing expectation values.

    Returns:
        A tuple containing expectation values and variances.

    """
    Logger.trace_entering()
    # Determine measurement basis and restrict Paulis to measured qubits (drop I terms)
    basis = _determine_measurement_basis(pauli_strings)
    n_qubits = len(basis)

    # For each Pauli string, extract only the measured-qubit bits and compress
    # them into a contiguous index via bit extraction.
    measured_indices = sorted(n_qubits - 1 - j for j, ch in enumerate(basis) if ch != "I")
    nonid_masks = _paulis_to_nonid_masks(pauli_strings)
    diag_inds = []
    for nonid in nonid_masks:
        val = 0
        for bit_pos, q in enumerate(measured_indices):
            if nonid & (1 << q):
                val |= 1 << bit_pos
        diag_inds.append(val)

    expvals = np.zeros(len(pauli_strings), dtype=float)
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


def _determine_measurement_basis(pauli_strings: list[str]) -> str:
    """Determine the measurement basis for a group of qubit-wise commuting Pauli operators.

    Example: ["IZ", "YZ"] -> "YZ"

    Args:
        pauli_strings: List of Pauli label strings that must be qubit-wise commuting.

    Returns:
        A Pauli label string representing the combined measurement basis.

    """
    n_qubits = len(pauli_strings[0])
    basis = ["I"] * n_qubits
    for ps in pauli_strings:
        for j, ch in enumerate(ps):
            if ch != "I":
                if basis[j] == "I":
                    basis[j] = ch
                elif basis[j] != ch:
                    raise ValueError(
                        "Paulis are not qubit-wise commuting. "
                        "Please group them first to generate a valid measurement basis."
                    )
    return "".join(basis)


def _append_measurement_to_circuit(base_circuit: Circuit, m_basis: str) -> Circuit:
    """Append measurement operations to a base circuit according to the specified measurement basis.

    Args:
        base_circuit: The original quantum circuit to which measurement operations will be appended.
        m_basis: Pauli label string (e.g. ``"YZI"``).

    Returns:
        Circuit: The modified circuit with measurement operations appended.

    """
    if base_circuit._qsharp_op:  # noqa: SLF001
        pauli_base = []
        for pauli in reversed(m_basis):
            pauli_base.append(getattr(qsharp.Pauli, pauli))
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.MeasurementBasis.MakeMeasurementCircuit,
            parameter={"baseCircuit": base_circuit._qsharp_op, "bases": pauli_base, "numQubits": len(m_basis)},  # noqa: SLF001
        )
        return Circuit(qsharp_factory=qsharp_factory)

    try:
        from qiskit import (  # noqa: PLC0415
            ClassicalRegister,
            QuantumCircuit,
            QuantumRegister,
            qasm3,
        )
        from qiskit.quantum_info import Pauli  # noqa: PLC0415
    except ImportError as err:
        raise ImportError("Qiskit is required to use Qiskit circuits with EnergyEstimator.") from err
    base_circuit = base_circuit.get_qiskit_circuit()
    basis = Pauli(m_basis)
    active = np.arange(basis.num_qubits)[basis.z | basis.x]
    qreg = QuantumRegister(basis.num_qubits, "q")
    creg = ClassicalRegister(len(active), "c")
    qc = QuantumCircuit(qreg, creg)
    qc.compose(base_circuit, inplace=True)
    for cidx, qidx in enumerate(active):
        if basis.x[qidx]:
            if basis.z[qidx]:
                qc.sdg(qreg[qidx])  # If x=1 and z=1, Y basis
            qc.h(qreg[qidx])  # If x=1 and z=0, X basis
        qc.measure(qreg[qidx], creg[cidx])
    return Circuit(qasm=qasm3.dumps(qc))


class QdkEnergyEstimator(EnergyEstimator):
    """QDK implementation of the EnergyEstimator."""

    def __init__(self):
        """Initialize the QdkEnergyEstimator."""
        super().__init__()

    def type_name(self) -> str:
        """Return ``energy_estimator`` as the algorithm type name."""
        return "energy_estimator"

    def _run_impl(
        self,
        circuit: Circuit,
        qubit_hamiltonians: list[QubitHamiltonian],
        circuit_executor: CircuitExecutor,
        total_shots: int,
        noise_model: QuantumErrorProfile | None = None,
        classical_coeffs: list | None = None,
    ) -> tuple[EnergyExpectationResult, MeasurementData]:
        """Estimate the expectation value and variance of Hamiltonians.

        Args:
            circuit: Circuit.
            qubit_hamiltonians: List of ``QubitHamiltonian`` to estimate.
            circuit_executor: An instance of ``CircuitExecutor`` to run quantum circuits.
            total_shots: Total number of shots to allocate across the observable terms.
            noise_model: Optional noise model to simulate noise in the quantum circuit.
            classical_coeffs: Optional list of coefficients for classical Pauli terms to calculate energy offset.

        Returns:
            tuple[EnergyExpectationResult, MeasurementData]: Tuple containing:

                * ``energy_result``: Energy expectation value and variance for the provided Hamiltonians.
                * ``measurement_data``: Raw measurement counts and metadata used to compute the expectation value.

        Note:
            * Measurement circuits are generated for each QubitHamiltonian term.
            * Parameterized circuits are not supported.
            * Only one circuit is supported per run.

        """
        # This function definition is not required it is present to add type hints and docstrings
        #  for the derived classes specialized run() method.
        Logger.trace_entering()
        num_observables = len(qubit_hamiltonians)
        if total_shots < num_observables:
            raise ValueError(
                f"Total shots {total_shots} is less than the number of observables {num_observables}. "
                "Please increase total shots to ensure each observable is measured."
            )

        # Evenly distribute shots across all observables
        shots_list = [total_shots // num_observables] * num_observables
        Logger.debug(f"Shots allocated: {shots_list}")

        energy_offset = sum(classical_coeffs) if classical_coeffs else 0.0

        # Create measurement circuits
        measurement_circuits = self._create_measurement_circuits(
            circuit=circuit,
            grouped_hamiltonians=qubit_hamiltonians,
        )

        measurement_data = self._get_measurement_data(
            measurement_circuits=measurement_circuits,
            qubit_hamiltonians=qubit_hamiltonians,
            circuit_executor=circuit_executor,
            shots_list=shots_list,
            noise_model=noise_model,
        )

        return self._compute_energy_expectation_from_bitstrings(
            qubit_hamiltonians, measurement_data.bitstring_counts, energy_offset
        ), measurement_data

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

        for hamiltonian in grouped_hamiltonians:
            basis = _determine_measurement_basis(hamiltonian.pauli_strings)
            full_circuit = _append_measurement_to_circuit(circuit, basis)
            meas_circuits.append(full_circuit)

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
            paulis = group.pauli_strings
            coeffs = group.coefficients

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

    def _run_measurement_circuits_and_get_bitstring_counts(
        self,
        measurement_circuits: list[Circuit],
        circuit_executor: CircuitExecutor,
        shots_list: list[int],
        noise_model: QuantumErrorProfile | None = None,
    ) -> list[dict[str, int]]:
        """Run the measurement circuits and return the bitstring counts.

        Args:
            measurement_circuits: A list of Circuits that provide measurement circuits in OpenQASM3 format to run.
            circuit_executor: An instance of CircuitExecutor to run the circuits.
            shots_list: A list of shots allocated for each measurement circuit.
            noise_model: Optional noise model to simulate noise in the quantum circuit.

        Returns:
            A list of dictionaries containing the bitstring counts for each measurement circuit.

        """
        all_bitstring_counts: list[dict[str, int]] = []
        for circuit, shots in zip(measurement_circuits, shots_list, strict=True):
            result = circuit_executor.run(
                circuit,
                shots=shots,
                noise=noise_model,
            )
            all_bitstring_counts.append(result.bitstring_counts if result and result.bitstring_counts else {})
        return all_bitstring_counts

    def _get_measurement_data(
        self,
        measurement_circuits: list[Circuit],
        qubit_hamiltonians: list[QubitHamiltonian],
        circuit_executor: CircuitExecutor,
        shots_list: list[int],
        noise_model: QuantumErrorProfile | None = None,
    ) -> MeasurementData:
        """Get ``MeasurementData`` from running measurement circuits.

        Args:
            measurement_circuits: A list of measurement circuits to run.
            qubit_hamiltonians: A list of ``QubitHamiltonian`` to be evaluated.
            circuit_executor: An instance of ``CircuitExecutor`` to run the circuits.
            shots_list: A list of shots allocated for each measurement circuit.
            noise_model: Optional noise model to simulate noise in the quantum circuit.

        Returns:
            MeasurementData: Measurement counts paired with their corresponding ``QubitHamiltonian`` objects.

        """
        counts = self._run_measurement_circuits_and_get_bitstring_counts(
            measurement_circuits, circuit_executor, shots_list, noise_model
        )
        return MeasurementData(
            bitstring_counts=counts,
            hamiltonians=qubit_hamiltonians,
            shots_list=shots_list,
        )

    def name(self) -> str:
        """Get the name of the estimator for registry purposes."""
        return "qdk"
