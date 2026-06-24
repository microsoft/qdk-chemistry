r"""QDK/Chemistry implementation of the LCU (Linear Combination of Unitaries).

References:
    Childs, A. M. and Wiebe, N. "Hamiltonian simulation using linear
    combinations of unitary operations." *Quantum Information &
    Computation* 12.11-12 (2012): 901-924.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import (
    HamiltonianUnitaryBuilder,
    HamiltonianUnitaryBuilderSettings,
)
from qdk_chemistry.data import (
    Configuration,
    ModelOrbitals,
    QubitHamiltonian,
    StateVectorContainer,
    UnitaryRepresentation,
    Wavefunction,
)
from qdk_chemistry.data.unitary_representation.containers.block_encoding import (
    ControlledOperation,
    LCUContainer,
    Select,
)
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer

__all__: list[str] = ["LCUBuilder", "LCUSettings"]


class LCUSettings(HamiltonianUnitaryBuilderSettings):
    """Settings for the LCU block encoding builder."""

    def __init__(self):
        """Initialize LCUSettings with default values.

        Attributes:
            power: The power to which the Hamiltonian is raised.
            quantum_walk: If True, wrap block encoding with quantum walk operator (use with QPE).
                If False, use plain block encoding (use with Hadamard test).
            tolerance: Minimum L1 norm below which the LCU decomposition is numerically ill-defined.

        """
        super().__init__()
        self._set_default("power", "int", 1, "The power to which the Hamiltonian is raised.")
        self._set_default(
            "quantum_walk",
            "bool",
            False,
            "If True, wrap block encoding with quantum walk operator (use with QPE). "
            "If False, use plain block encoding (use with Hadamard test).",
        )
        self._set_default(
            "tolerance",
            "float",
            1e-12,
            "Minimum L1 norm below which the LCU decomposition is numerically ill-defined.",
        )


class LCUBuilder(HamiltonianUnitaryBuilder):
    r"""LCU (Linear Combination of Unitaries) block encoding builder."""

    def __init__(
        self,
        power: int = 1,
        quantum_walk: bool = False,
    ):
        r"""Initialize the LCU builder.

        Given a qubit Hamiltonian :math:`H = \sum_{j=1}^{L} \alpha_j P_j` expressed as a
        linear combination of Pauli strings :math:`P_j` with scalar coefficients
        :math:`\alpha_j`, this builder constructs an LCU representation that block-encodes
        :math:`H / \lambda`, where :math:`\lambda` is the L1 norm of the
        Hamiltonian. The LCU representation follows the PREPARE-SELECT-PREPARE† pattern.

        The PREPARE oracle encodes amplitudes :math:`\sqrt{|\alpha_j| / \lambda}` into
        an ancilla register of :math:`\lceil \log_2 L \rceil` qubits. The SELECT oracle applies
        the corresponding Pauli string :math:`P_j` (with sign correction) controlled on the
        ancilla index. The :math:`\alpha_j` is positive by absorbing the sign into phases of
        the SELECT operation.

        Args:
            power: The power to raise the unitary to. Defaults to 1.
            quantum_walk: If True, the circuit mapper wraps the block encoding with
                a quantum walk operator (use with QPE). If False, use the plain
                block encoding (use with Hadamard test). Defaults to False.

        """
        super().__init__()
        self._settings = LCUSettings()
        self._settings.set("power", power)
        self._settings.set("quantum_walk", quantum_walk)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Construct the unitary representation using LCU block encoding.

        Computes normalized amplitudes, signs, and controlled operations from the
        qubit Hamiltonian, then packages them into generalized Prepare/Select
        dataclasses stored in an LCUContainer.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.

        Returns:
            UnitaryRepresentation: The unitary representation wrapping the built LCUContainer.

        """
        power: int = self._settings.get("power")
        quantum_walk: bool = self._settings.get("quantum_walk")

        if not qubit_hamiltonian.is_hermitian():
            raise ValueError("LCU block encoding requires a Hermitian Hamiltonian.")

        coefficients = qubit_hamiltonian.coefficients
        num_terms = len(coefficients)
        num_prepare_ancillas = int(np.ceil(np.log2(num_terms)))

        prepare_wfn = self._build_prepare(qubit_hamiltonian, num_prepare_ancillas, self._settings.get("tolerance"))
        select = self._build_select(qubit_hamiltonian, num_prepare_ancillas)

        lcu_container = LCUContainer(
            power=power,
            prepare=prepare_wfn,
            select=select,
        )

        container = (
            LCUWalkContainer(block_encoding=lcu_container, power=power, scale=qubit_hamiltonian.schatten_norm)
            if quantum_walk
            else lcu_container
        )

        return UnitaryRepresentation(container=container)

    @staticmethod
    def _build_prepare(
        qubit_hamiltonian: QubitHamiltonian, num_prepare_ancillas: int, tolerance: float
    ) -> "Wavefunction":
        """Compute the prepare wavefunction from Hamiltonian coefficients.

        Normalizes the absolute Hamiltonian coefficients by the L1 norm and
        takes the element-wise square root to produce the state-preparation amplitudes.
        Returns a :class:`Wavefunction` with 1-bit-per-mode configurations.
        For single-term Hamiltonians (``num_prepare_ancillas == 0``), returns a
        trivial 0-mode wavefunction.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian whose coefficients define the amplitudes.
            num_prepare_ancillas: Number of qubits in the prepare ancillary register.
            tolerance: Minimum allowable L1 norm; raises if the norm is below
                this threshold.

        Returns:
            Wavefunction: A wavefunction over the ancilla register whose amplitudes
            encode the normalized coefficients (0-mode for single-term case).

        """
        l1_norm = qubit_hamiltonian.schatten_norm
        if l1_norm < tolerance:
            raise ValueError("L1 norm is too small, cannot build LCU block encoding.")

        if num_prepare_ancillas == 0:
            orbitals = ModelOrbitals(0)
            container = StateVectorContainer([1.0], [Configuration.from_bitstring("")], orbitals)
            return Wavefunction(container)

        coefficients = np.array([c for _, c in qubit_hamiltonian.get_real_coefficients()])
        abs_coeffs = np.abs(coefficients)
        amplitudes = np.sqrt(abs_coeffs / l1_norm)

        # Build 1-bit-per-mode Configuration determinants for each nonzero amplitude.
        coeffs_list: list[float] = []
        dets: list[Configuration] = []
        for idx, amp in enumerate(amplitudes):
            if amp != 0.0:
                bitstring = format(idx, f"0{num_prepare_ancillas}b")[::-1]
                dets.append(Configuration.from_bitstring(bitstring))
                coeffs_list.append(float(amp))

        coeffs_array = np.array(coeffs_list)
        orbitals = ModelOrbitals(num_prepare_ancillas)
        container = StateVectorContainer(coeffs_array, dets, orbitals)
        return Wavefunction(container)

    @staticmethod
    def _build_select(qubit_hamiltonian: QubitHamiltonian, num_prepare_ancillas: int) -> Select:
        """Compute SELECT controlled operations and phases from Hamiltonian terms.

        Builds a list of controlled Pauli-string operations (one per Hamiltonian term)
        and an array of sign phases extracted from the real coefficients.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian whose Pauli strings and coefficients
                define the controlled operations.
            num_prepare_ancillas: Number of qubits in the prepare ancillary register.

        Returns:
            Select: The SELECT oracle dataclass containing controlled operations, phases,
                and qubit layout.

        """
        real_coeffs = qubit_hamiltonian.get_real_coefficients()
        pauli_strings = [label for label, _ in real_coeffs]
        coefficients = np.array([coeff for _, coeff in real_coeffs])
        num_terms = len(coefficients)
        num_system_qubits = qubit_hamiltonian.num_qubits
        phases = np.where(coefficients >= 0, 1, -1)

        controlled_ops = [
            ControlledOperation(
                ctrl_state=i,
                operation=pauli_strings[i],
            )
            for i in range(num_terms)
        ]

        return Select(
            controlled_operations=controlled_ops,
            phases=phases,
            num_prepare_ancillas=num_prepare_ancillas,
            num_target_qubits=num_system_qubits,
            prepare_qubits=list(range(num_prepare_ancillas)),
            target_qubits=list(range(num_prepare_ancillas, num_prepare_ancillas + num_system_qubits)),
        )

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"lcu"``.

        """
        return "lcu"

    def type_name(self) -> str:
        """Return the algorithm type name.

        Returns:
            str: The type name ``"hamiltonian_unitary_builder"``.

        """
        return "hamiltonian_unitary_builder"
