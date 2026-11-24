"""QDK/Chemistry circuit mappers that synthesize unitary matrices and isometries."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry._core.utils.unitary_synthesis import (
    block_sparse_isometry_synthesis,
    dense_isometry_right_factor,
    dense_isometry_synthesis,
)
from qdk_chemistry.data import Settings
from qdk_chemistry.data.circuit import Circuit, CircuitMetadata, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.unitary_matrix import UnitaryMatrixContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import CircuitMapper

if TYPE_CHECKING:
    from qdk_chemistry._core.utils.unitary_synthesis import GivensDecomposition
    from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation

__all__: list[str] = [
    "BlockSparseUnitarySynthesisMapper",
    "BlockSparseUnitarySynthesisMapperSettings",
    "DenseUnitarySynthesisMapper",
    "DenseUnitarySynthesisMapperSettings",
    "HouseholderUnitarySynthesisMapper",
    "HouseholderUnitarySynthesisMapperSettings",
    "UnitarySynthesisMapper",
    "UnitarySynthesisMapperSettings",
]

_HASH_CHARS = 64  # Full SHA-256 digest, so a reused right factor cannot belong to another matrix.


class UnitarySynthesisMapperSettings(Settings):
    """Settings shared by the unitary synthesis mappers."""

    def __init__(self):
        """Initialize the shared unitary synthesis settings."""
        super().__init__()
        self._set_default(
            "rotation_bit_precision",
            "int",
            10,
            "Number of bits for rotation angle precision, from 1 to 30 inclusive.",
            (1, 30),
        )
        self._set_default(
            "allocate_phase_gradient",
            "bool",
            True,
            "Whether the operation allocates its own phase gradient; when False it acts on "
            "[target | phase gradient] and the caller prepares the trailing rotation_bit_precision qubits.",
        )
        self._set_default(
            "num_input_qubits",
            "int",
            -1,
            "Width of the input register, the low target qubits, with the others starting in |0>; "
            "-1 selects the smallest allowed width.",
            (-1, 62),
        )
        self._set_default(
            "max_swap_address_bits",
            "int",
            -1,
            "Upper bound on the address width k of each QROAM swap network of the table lookups; "
            "-1 leaves each loader's cost model to choose, 0 forces plain select, "
            "and a positive value limits each loader to widths at or below it.",
            (-1, 30),
        )


class DenseUnitarySynthesisMapperSettings(UnitarySynthesisMapperSettings):
    """Settings for the DenseUnitarySynthesisMapper."""

    def __init__(self):
        """Initialize the settings for DenseUnitarySynthesisMapper."""
        super().__init__()
        self._set_default(
            "omit_right_factor",
            "bool",
            False,
            "Whether to leave the right factor V out, so the circuit realizes M @ V.T on its first columns.",
        )


class BlockSparseUnitarySynthesisMapperSettings(UnitarySynthesisMapperSettings):
    """Settings for the BlockSparseUnitarySynthesisMapper."""


class HouseholderUnitarySynthesisMapperSettings(UnitarySynthesisMapperSettings):
    """Settings for the HouseholderUnitarySynthesisMapper."""


class UnitarySynthesisMapper(CircuitMapper):
    r"""Base mapper that synthesizes a real isometry :math:`M` from a UnitaryMatrixContainer.

    All rotations use quantized angles applied by phase-gradient addition.
    """

    _qsharp_prefix: str
    _max_added_qubits: int | None = None

    def type_name(self) -> str:
        """Return ``circuit_mapper``."""
        return "circuit_mapper"

    @abstractmethod
    def _synthesize(self, container: UnitaryMatrixContainer, num_input_qubits: int) -> dict[str, Any]:
        """Return the fields of the Q# ``{_qsharp_prefix}Synthesis`` struct."""

    def _container(self, unitary: UnitaryRepresentation) -> UnitaryMatrixContainer:
        """Return the real UnitaryMatrixContainer of ``unitary``."""
        container = unitary.get_container()
        if not isinstance(container, UnitaryMatrixContainer):
            raise ValueError(
                f"The {unitary.get_container_type()} container type is not supported; "
                f"{self.name()} requires a unitary_matrix container."
            )
        if not container.is_real:
            raise ValueError(f"{self.name()} requires a real matrix, but the matrix is complex.")
        return container

    def _num_input_qubits(self, container: UnitaryMatrixContainer) -> int:
        """Return the input register width, validating the ``num_input_qubits`` setting."""
        num_qubits = container.num_qubits
        smallest = (container.num_columns - 1).bit_length()
        if self._max_added_qubits is not None:
            smallest = max(smallest, num_qubits - self._max_added_qubits)
        requested = self._settings.get("num_input_qubits")
        if requested == -1:
            return smallest
        if not smallest <= requested <= num_qubits:
            raise ValueError(
                f"num_input_qubits={requested} is invalid for {self.name()} on a {num_qubits}-qubit matrix with "
                f"{container.num_columns} columns. It must be at least {smallest} and at most {num_qubits}."
            )
        return requested

    def synthesize(self, unitary: UnitaryRepresentation) -> dict[str, Any]:
        """Return the fields of the Q# ``{_qsharp_prefix}Synthesis`` struct without building a Q# operation.

        Args:
            unitary: The unitary representation holding a UnitaryMatrixContainer.

        Returns:
            dict[str, Any]: The struct fields, with the register widths and ``maxSwapBits``, that ``run`` passes to Q#.

        Raises:
            ValueError: If the container or the settings are invalid for the matrix.

        """
        container = self._container(unitary)
        num_input_qubits = self._num_input_qubits(container)
        return {
            **self._synthesize(container, num_input_qubits),
            "maxSwapBits": self._settings.get("max_swap_address_bits"),
            "numInputQubits": num_input_qubits,
            "numAddedQubits": container.num_qubits - num_input_qubits,
            "rotationBitPrecision": self._settings.get("rotation_bit_precision"),
        }

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        """Synthesize the circuit of the isometry in ``unitary``.

        Args:
            unitary: The unitary representation holding a UnitaryMatrixContainer.

        Returns:
            Circuit: The circuit, whose qubits include the phase gradient when ``allocate_phase_gradient`` is False.

        Raises:
            ValueError: If the container or the settings are invalid for the matrix.

        """
        synthesis = self.synthesize(unitary)
        module, prefix = QSHARP_UTILS.IsometrySynthesis, self._qsharp_prefix
        qsharp_synthesis = getattr(module, f"{prefix}Synthesis")(**synthesis)
        if self._settings.get("allocate_phase_gradient"):
            qsharp_op = getattr(module, f"Make{prefix}Op")(qsharp_synthesis)
            num_gradient_ancillas = 0
        else:
            qsharp_op = getattr(module, f"Make{prefix}OpWithPhaseGradient")(qsharp_synthesis)
            num_gradient_ancillas = synthesis["rotationBitPrecision"]
        # An exported circuit has no caller to own the gradient, so it always allocates its own.
        program = getattr(module, f"Make{prefix}Circuit")
        return Circuit(
            qsharp_op=qsharp_op,
            qsharp_factory=QsharpFactoryData(program=program, parameter={"synthesis": synthesis}),
            num_qubits=synthesis["numInputQubits"] + synthesis["numAddedQubits"] + num_gradient_ancillas,
            metadata=CircuitMetadata(num_phase_gradient_ancillas=num_gradient_ancillas),
        )


class DenseUnitarySynthesisMapper(UnitarySynthesisMapper):
    """Isometry synthesis by cosine-sine decompositions :cite:`Berry2025,Rupprecht2026` that adds at most two qubits."""

    _qsharp_prefix = "DenseIsometry"
    _max_added_qubits = 2

    def __init__(self):
        """Initialize the DenseUnitarySynthesisMapper."""
        super().__init__()
        self._settings = DenseUnitarySynthesisMapperSettings()
        # V of the last run with its matrix hash and input width, which right_factor reuses.
        self._last_right_factor: tuple[str, int, np.ndarray] | None = None

    def name(self) -> str:
        """Return ``unitary_synthesis_dense``."""
        return "unitary_synthesis_dense"

    def right_factor(self, unitary: UnitaryRepresentation) -> np.ndarray:
        """Return the right factor ``V`` that ``omit_right_factor`` leaves out, reusing the last run's factor.

        Args:
            unitary: The unitary representation to map with the current settings.

        Returns:
            numpy.ndarray: Orthogonal matrix with one row and column per column of the matrix.

        Raises:
            ValueError: If the container or ``num_input_qubits`` is invalid for the matrix.

        """
        container = self._container(unitary)
        num_input_qubits = self._num_input_qubits(container)
        if self._last_right_factor is not None:
            matrix_hash, last_input_qubits, right_factor = self._last_right_factor
            if last_input_qubits == num_input_qubits and matrix_hash == container.content_hash(_HASH_CHARS):
                return right_factor.copy()
        return dense_isometry_right_factor(container.to_dense(), 1 << num_input_qubits)

    def _synthesize(self, container: UnitaryMatrixContainer, num_input_qubits: int) -> dict[str, Any]:
        """Return the fields of the Q# ``DenseIsometrySynthesis`` struct and cache the right factor."""
        synthesis = dense_isometry_synthesis(
            container.to_dense(), 1 << num_input_qubits, omit_right_factor=self._settings.get("omit_right_factor")
        )
        self._last_right_factor = (container.content_hash(_HASH_CHARS), num_input_qubits, synthesis.right_factor.copy())
        right_factor = synthesis.right_factor_givens
        return {
            "rotationAngles": synthesis.rotation_angles,
            "mixingGivens": [_givens_to_qsharp(givens) for givens in synthesis.mixing_givens],
            "blockGivens": _givens_to_qsharp(synthesis.block_givens),
            "rightFactorGivens": [] if right_factor is None else [_givens_to_qsharp(right_factor)],
        }


class BlockSparseUnitarySynthesisMapper(UnitarySynthesisMapper):
    """Isometry synthesis as QROAM permutations around a block-diagonal orthogonal matrix :cite:`Rupprecht2026`."""

    _qsharp_prefix = "SparseIsometry"

    def __init__(self):
        """Initialize the BlockSparseUnitarySynthesisMapper."""
        super().__init__()
        self._settings = BlockSparseUnitarySynthesisMapperSettings()

    def name(self) -> str:
        """Return ``unitary_synthesis_block_sparse``."""
        return "unitary_synthesis_block_sparse"

    def _synthesize(self, container: UnitaryMatrixContainer, num_input_qubits: int) -> dict[str, Any]:  # noqa: ARG002
        """Return the fields of the Q# ``SparseIsometrySynthesis`` struct."""
        # The permutations act on the whole register, so the input width only splits the Q# arguments.
        synthesis = block_sparse_isometry_synthesis(container.matrix)
        return {
            "columnPermutation": synthesis.column_permutation,
            "rowPermutation": synthesis.row_permutation,
            "blockGivens": _givens_to_qsharp(synthesis.block_givens),
        }


class HouseholderUnitarySynthesisMapper(UnitarySynthesisMapper):
    r"""Isometry synthesis as a product of Householder reflections, one per column :cite:`Low2024,Hennefarth2026`.

    Column :math:`j` contributes the reflection about :math:`(|0\rangle|j\rangle - |1\rangle|v_j\rangle)/\sqrt{2}` on a
    reflection ancilla and the target, where :math:`|v_j\rangle` is prepared by a state preparation controlled on the
    ancilla, with QROAM-loaded rotation angles. The cost grows with the number of columns times their length, and the
    method adds any number of qubits.
    """

    _qsharp_prefix = "HouseholderIsometry"

    def __init__(self):
        """Initialize the HouseholderUnitarySynthesisMapper."""
        super().__init__()
        self._settings = HouseholderUnitarySynthesisMapperSettings()

    def name(self) -> str:
        """Return ``unitary_synthesis_householder``."""
        return "unitary_synthesis_householder"

    @staticmethod
    def column_preparation(amplitudes: np.ndarray) -> tuple[list[list[float]], list[bool]]:
        """Return the state-preparation data of a real vector, padded to a power-of-two length.

        Args:
            amplitudes: Real amplitudes of the state to prepare.

        Returns:
            The Ry angles 2θ per level, the most significant bit first, with cos θ = sqrt(p_low / p) of the
            probability tree; and the sign of each amplitude, empty when none is negative.

        """
        num_bits = (len(amplitudes) - 1).bit_length()
        padded = np.zeros(1 << num_bits)
        padded[: len(amplitudes)] = amplitudes
        probabilities = np.square(padded)
        angles = []
        for level in range(num_bits):
            tree = probabilities.reshape(1 << level, 2, -1).sum(axis=2)
            total = tree.sum(axis=1)
            ratio = np.divide(tree[:, 0], total, out=np.ones_like(total), where=total > 0)
            angles.append((2.0 * np.arccos(np.sqrt(np.clip(ratio, 0.0, 1.0)))).tolist())
        negative = padded < 0
        return angles, negative.tolist() if negative.any() else []

    def _synthesize(self, container: UnitaryMatrixContainer, num_input_qubits: int) -> dict[str, Any]:  # noqa: ARG002
        """Return the fields of the Q# ``HouseholderIsometrySynthesis`` struct."""
        columns = [self.column_preparation(column) for column in container.to_dense().T]
        return {
            "columnAngles": [angles for angles, _ in columns],
            "columnSigns": [signs for _, signs in columns],
        }


def _givens_to_qsharp(givens: GivensDecomposition) -> dict[str, Any]:
    """Return the fields of the Q# ``GivensDecomposition`` struct."""
    return {"layerAngles": givens.layer_angles, "layerShifted": givens.layer_shifted, "phases": givens.phases}
