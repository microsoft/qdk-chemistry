"""QDK/Chemistry state preparation abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings
from dataclasses import dataclass
from typing import Any

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, Settings, Wavefunction

__all__: list[str] = ["PrepareLayout"]


@dataclass(frozen=True)
class PrepareLayout:
    r"""Register widths a PREPARE oracle needs inside a block encoding.

    A block encoding hands PREPARE one register and SELECT another, and for simple
    state preparations those are the same qubits. They are not always: an oracle may
    need scratch beyond the index it produces, and it may want to share a resource
    across the whole circuit rather than re-create it per call. This splits the three
    cases apart.

    Attributes:
        num_select_qubits: Width of the index SELECT controls on. Only these qubits
            carry index information.
        num_block_ancillas: Total width PREPARE owns, index plus any garbage left
            entangled with it. ``PREPARE``\ :sup:`†` returns all of it to
            :math:`|0\rangle`, and a qubitization walk reflects about exactly this
            register.
        num_shared_ancillas: Width of ancilla prepared once for the whole circuit and
            left in a non-zero state between uses, such as a phase gradient. These are
            deliberately excluded from the walk reflection.

    """

    num_select_qubits: int
    num_block_ancillas: int
    num_shared_ancillas: int = 0


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

    def prepare_layout(self, wavefunction: Wavefunction) -> PrepareLayout:
        """Return the register widths this oracle needs inside a block encoding.

        The default assumes the oracle produces a pure state on the index register and
        needs nothing else, which is true of every state preparation that is not
        explicitly a PREPARE oracle.

        Args:
            wavefunction: The wavefunction that will be prepared.

        Returns:
            PrepareLayout: The index width, repeated as the block ancilla width, with no
            shared ancilla.

        """
        num_index_qubits = wavefunction.get_orbitals().num_modes()
        return PrepareLayout(
            num_select_qubits=num_index_qubits,
            num_block_ancillas=num_index_qubits,
        )

    def prepare_oracle(self, wavefunction: Wavefunction) -> tuple[Any, PrepareLayout]:
        """Return the Q# PREPARE callable to embed in a block encoding, and its layout.

        Separate from :meth:`run` because an oracle may want a different implementation
        when it is embedded than when it is a standalone circuit — QROM state preparation
        takes its phase gradient from the caller here rather than allocating one.

        Args:
            wavefunction: The wavefunction that will be prepared.

        Returns:
            The Q# callable and the register layout it expects.

        """
        return self.run(wavefunction)._qsharp_op, self.prepare_layout(wavefunction)  # noqa: SLF001


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
