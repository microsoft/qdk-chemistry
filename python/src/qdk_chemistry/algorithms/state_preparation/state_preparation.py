"""QDK/Chemistry state preparation abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings
from typing import Any

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, Settings, Wavefunction


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

    def num_system_qubits(self, wavefunction: Wavefunction) -> int:
        """Return the width of the index register a block encoding's SELECT controls on.

        Args:
            wavefunction: The wavefunction that will be prepared.

        Returns:
            The number of qubits carrying index information.

        """
        return wavefunction.get_orbitals().num_modes()

    def num_entangled_ancillas(self, wavefunction: Wavefunction) -> int:
        r"""Return the scratch width this oracle leaves entangled with the index.

        Zero for any state preparation that produces a pure state on the index register,
        which is every one that is not explicitly a PREPARE oracle. When it is non-zero
        the block ancilla register is wider than the index, and a qubitization walk
        reflects about all of it because ``PREPARE``\ :sup:`†` returns all of it to
        :math:`|0\rangle`.

        Args:
            wavefunction: The wavefunction that will be prepared.

        Returns:
            The number of scratch qubits beyond the index register.

        """
        del wavefunction
        return 0

    def num_phase_gradient_ancillas(self, wavefunction: Wavefunction) -> int:
        r"""Return the width of the phase gradient register this oracle reads.

        The gradient is an eigenstate of the addition it drives, so it comes back
        unchanged and one register can serve the whole circuit. It is left in
        :math:`|\phi\rangle` rather than :math:`|0\rangle` between uses, so the caller
        allocates and prepares it outside the block encoding and a qubitization walk
        must not reflect about it.

        Args:
            wavefunction: The wavefunction that will be prepared.

        Returns:
            The number of phase gradient qubits the caller must supply, zero if the
            oracle does not use one.

        """
        del wavefunction
        return 0

    def prepare_oracle(self, wavefunction: Wavefunction) -> Any:
        """Return the Q# PREPARE callable to embed in a block encoding.

        Separate from :meth:`run` because an oracle may want a different implementation
        when it is embedded than when it is a standalone circuit — QROM state preparation
        reads a caller-supplied phase gradient here rather than allocating its own.

        Args:
            wavefunction: The wavefunction that will be prepared.

        Returns:
            The Q# callable acting on the register described by this oracle's widths.

        """
        return self.run(wavefunction)._qsharp_op  # noqa: SLF001


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
