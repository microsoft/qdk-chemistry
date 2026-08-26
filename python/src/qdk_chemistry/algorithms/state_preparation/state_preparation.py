"""QDK/Chemistry state preparation abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings

import numpy as np

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, Settings, Wavefunction

__all__: list[str] = []

MAX_DENSE_QUBITS = 32


def dense_coefficients(wavefunction: Wavefunction, label: str) -> tuple[np.ndarray, int]:
    """Scatter a wavefunction's coefficients onto their determinant-derived indices.

    A ``Wavefunction`` stores only occupied determinants, so the coefficient list is not
    positionally aligned with the basis index. Each coefficient is placed at the index its
    determinant's bits encode (little-endian, matching ``dense_pure_state``), and the register
    width comes from the configuration set rather than the coefficient count.

    Args:
        wavefunction: The target wavefunction.
        label: Algorithm name, used to prefix error messages.

    Returns:
        The dense real coefficient vector and the width of the state register in qubits.

    Raises:
        ValueError: If the wavefunction has no coefficients, has a non-zero imaginary
            part, or is too wide to densify.

    """
    coefficients = np.asarray(wavefunction.get_coefficients())
    if coefficients.size == 0:
        raise ValueError(f"{label} requires at least one coefficient.")
    if np.iscomplexobj(coefficients):
        if not np.allclose(coefficients.imag, 0.0):
            raise ValueError(f"{label} requires real coefficients.")
        coefficients = coefficients.real
    coefficients = coefficients.astype(float, copy=False)

    determinants = wavefunction.get_active_determinants()
    num_bits = wavefunction.get_configuration_set().num_modes() * determinants[0].bits_per_mode()
    num_qubits = max(num_bits, 1)
    if num_qubits > MAX_DENSE_QUBITS:
        raise ValueError(f"{label} is only supported for up to {MAX_DENSE_QUBITS} qubits.")

    dense = np.zeros(1 << num_qubits, dtype=float)
    for coefficient, determinant in zip(coefficients, determinants, strict=True):
        index = 0
        for position, bit in enumerate(determinant.to_bits(num_bits)):
            index |= bit << position
        dense[index] += coefficient
    return dense, num_qubits


class StatePreparationSettings(Settings):
    """Deprecated settings container for state preparation algorithms.

    .. deprecated::
        Each state preparation algorithm now owns its settings, and the transpilation keys
        below belong to the Qiskit-backed algorithms that actually honour them. Kept so that
        existing imports keep working; it is no longer used by any algorithm in this package.
    """

    def __init__(self):
        """Initialize the StatePreparationSettings."""
        warnings.warn(
            "'StatePreparationSettings' is deprecated and will be removed in a future release; "
            "use the settings class of the specific state preparation algorithm instead "
            "(e.g. 'SparseIsometryStatePreparationSettings').",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self._set_default("basis_gates", "vector<string>", ["x", "y", "z", "cx", "cz", "id", "h", "s", "sdg", "rz"])
        self._set_default("transpile", "bool", True)
        self._set_default("transpile_optimization_level", "int", 0)


class StatePreparation(Algorithm):
    """Abstract base class for state preparation algorithms.

    .. note::
        **Current Limitation**: All state preparation algorithms currently only support
        the Jordan-Wigner encoding for fermion-to-qubit mapping. The returned :class:`~qdk_chemistry.data.Circuit`
        will have its ``encoding`` attribute set to ``"jordan-wigner"``.

        If you use the state preparation circuit with a :class:`~qdk_chemistry.data.QubitOperator`
        that uses a different encoding (e.g., ``"bravyi-kitaev"`` or ``"parity"``), the
        encodings will be incompatible and may lead to incorrect results.

        **Recommended workflow**:
            1. Create a :class:`~qdk_chemistry.data.QubitOperator` using Jordan-Wigner encoding
            2. Use state preparation to create a :class:`~qdk_chemistry.data.Circuit`
            3. Both will have ``encoding="jordan-wigner"`` and will be compatible

        Support for additional encodings is planned for future releases.

    """

    def __init__(self):
        """Initialize the StatePreparation with default settings."""
        super().__init__()

    def type_name(self) -> str:
        """Return the algorithm type name as state_prep."""
        return "state_prep"

    def run(self, wavefunction: Wavefunction) -> Circuit:
        """Prepare a quantum circuit that encodes the given wavefunction.

        Args:
            wavefunction: The target wavefunction to prepare.

        Returns:
            A Circuit object containing an OpenQASM3 string of the quantum circuit that prepares the wavefunction.

        """
        return super().run(wavefunction)


class StatePreparationFactory(AlgorithmFactory):
    """Factory class for creating StatePreparation instances."""

    def __init__(self):
        """Initialize the StatePreparationFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as state_prep."""
        return "state_prep"

    def default_algorithm_name(self) -> str:
        """Return the sparse_isometry as default algorithm name."""
        return "sparse_isometry"
