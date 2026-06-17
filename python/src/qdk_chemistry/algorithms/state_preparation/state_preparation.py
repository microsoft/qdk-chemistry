"""QDK/Chemistry state preparation abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, Settings, Wavefunction

__all__: list[str] = []


class StatePreparationSettings(Settings):
    """Settings for state preparation algorithms."""

    def __init__(self):
        """Initialize the StatePreparationSettings."""
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

        If you use the state preparation circuit with a :class:`~qdk_chemistry.data.QubitHamiltonian`
        that uses a different encoding (e.g., ``"bravyi-kitaev"`` or ``"parity"``), the
        encodings will be incompatible and may lead to incorrect results.

        **Recommended workflow**:
            1. Create a :class:`~qdk_chemistry.data.QubitHamiltonian` using Jordan-Wigner encoding
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
        """Return the sparse_isometry_gf2x as default algorithm name."""
        return "sparse_isometry_gf2x"
