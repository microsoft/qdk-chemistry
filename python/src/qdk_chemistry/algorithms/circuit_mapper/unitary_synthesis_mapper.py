"""QDK/Chemistry circuit mapper that synthesizes unitary matrices and isometries."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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
    import numpy as np

    from qdk_chemistry._core.utils.unitary_synthesis import (
        DenseIsometrySynthesis,
        GivensDecomposition,
        SparseIsometrySynthesis,
    )
    from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation

__all__: list[str] = ["UnitarySynthesisMapper", "UnitarySynthesisMapperSettings"]

_HASH_CHARS = 64  # Full SHA-256 digest, so a reused right factor cannot belong to another matrix.


class _SynthesisMethod(ABC):
    """A synthesis method of :class:`UnitarySynthesisMapper`.

    The mapper validates the container and the input register width, calls :meth:`synthesize`,
    and wraps the result in the Q# entry points ``{qsharp_prefix}Params``,
    ``Make{qsharp_prefix}Op``, ``Make{qsharp_prefix}OpWithPhaseGradient`` and
    ``Make{qsharp_prefix}Circuit`` of ``QDKChemistry.Utils.IsometrySynthesis``, whose header
    documents their signatures. To add a method, subclass this, add an instance to
    ``_METHODS``, and add those Q# entry points.
    """

    name: str
    """Value of the ``method`` setting."""
    description: str
    """Short description for the ``method`` setting."""
    qsharp_prefix: str
    """Prefix of the Q# entry points."""
    max_added_qubits: int | None = None
    """Most qubits added to the input register, or None for no limit."""
    supports_complex: bool = False
    """Whether the method synthesizes matrices with complex entries."""
    has_right_factor: bool = False
    """Whether the method can leave a right factor out of the circuit, see :meth:`right_factor`."""

    @abstractmethod
    def synthesize(
        self, container: UnitaryMatrixContainer, num_input_qubits: int, omit_right_factor: bool
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        """Return the fields of the Q# ``{qsharp_prefix}Synthesis`` struct and the right factor, if any."""

    def right_factor(self, container: UnitaryMatrixContainer, num_input_qubits: int) -> np.ndarray:
        """Return the right factor that ``omit_right_factor`` leaves out, when ``has_right_factor`` holds."""
        raise NotImplementedError(f"The '{self.name}' method has no right factor.")


class _DenseMethod(_SynthesisMethod):
    name = "dense"
    description = "cosine-sine decomposition"
    qsharp_prefix = "DenseIsometry"
    max_added_qubits = 2
    has_right_factor = True

    def synthesize(
        self, container: UnitaryMatrixContainer, num_input_qubits: int, omit_right_factor: bool
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        synthesis = dense_isometry_synthesis(
            container.to_dense(), 1 << num_input_qubits, omit_right_factor=omit_right_factor
        )
        return _dense_to_qsharp(synthesis), synthesis.right_factor

    def right_factor(self, container: UnitaryMatrixContainer, num_input_qubits: int) -> np.ndarray:
        return dense_isometry_right_factor(container.to_dense(), 1 << num_input_qubits)


class _BlockSparseMethod(_SynthesisMethod):
    name = "block_sparse"
    description = "permutations around a block-diagonal orthogonal matrix"
    qsharp_prefix = "SparseIsometry"

    def synthesize(
        self,
        container: UnitaryMatrixContainer,
        num_input_qubits: int,  # noqa: ARG002
        omit_right_factor: bool,  # noqa: ARG002
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        # The permutations act on the whole register, so the input width only splits the Q# arguments.
        return _sparse_to_qsharp(block_sparse_isometry_synthesis(container.matrix)), None


_METHODS: dict[str, _SynthesisMethod] = {method.name: method for method in (_DenseMethod(), _BlockSparseMethod())}


class UnitarySynthesisMapperSettings(Settings):
    """Settings for the UnitarySynthesisMapper."""

    def __init__(self):
        """Initialize the settings for UnitarySynthesisMapper."""
        super().__init__()
        self._set_default(
            "method",
            "string",
            "dense",
            "Synthesis method: "
            + ", ".join(f"'{method.name}' ({method.description})" for method in _METHODS.values())
            + ".",
            list(_METHODS),
        )
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
            "Whether the operation allocates its own phase gradient register. When False, the "
            "operation acts on [target | phase gradient] and the caller prepares and keeps the "
            "trailing rotation_bit_precision qubits.",
        )
        self._set_default(
            "omit_right_factor",
            "bool",
            False,
            "Whether the dense method leaves the right factor V out of the circuit. The circuit then "
            "realizes M @ V.T on its first columns; UnitarySynthesisMapper.right_factor returns V.",
        )
        self._set_default(
            "num_input_qubits",
            "int",
            -1,
            "Input register width, the low qubits of the target; the other qubits are added in |0>. "
            "The input register must hold the columns, and the dense method adds at most two qubits. "
            "-1 selects the smallest allowed width.",
            (-1, 62),
        )


class UnitarySynthesisMapper(CircuitMapper):
    r"""Circuit mapper that synthesizes a unitary matrix or isometry.

    The input is a :class:`~qdk_chemistry.data.UnitaryRepresentation` holding a
    :class:`~qdk_chemistry.data.UnitaryMatrixContainer` with matrix :math:`M` of :math:`2^n`
    rows and :math:`m` orthonormal columns. The circuit acts on :math:`n` little-endian qubits
    and realizes a unitary :math:`U` whose first :math:`m` columns equal :math:`M`. The low
    ``num_input_qubits`` qubits form the input register, which must hold the :math:`m`
    columns, and the remaining qubits start in :math:`|0\rangle`.

    Methods:
        - ``"dense"``: isometry synthesis by cosine-sine decompositions :cite:`Rupprecht2026`,
          which adds at most two qubits to the input register. The first decomposition leaves
          a right factor :math:`V` on the input register; with ``omit_right_factor`` the
          circuit drops it and realizes :math:`M V^T` on its first :math:`m` columns, and
          :meth:`right_factor` returns :math:`V`.
        - ``"block_sparse"``: :math:`U = P_r B P_c` with row and column permutations
          :math:`P_r, P_c` loaded by QROAM :cite:`Berry2025` and a block-diagonal orthogonal
          :math:`B` that couples only columns with a common nonzero row.

    Both methods require a real matrix. All orthogonal factors are applied as layers of Givens
    rotations :cite:`Clements2016` whose angles, like the multiplexed rotations, are quantized
    to ``rotation_bit_precision`` bits and applied by phase-gradient addition.

    The returned operation acts on the :math:`n` target qubits. With
    ``allocate_phase_gradient`` disabled it acts on
    :math:`[\text{target} \mid \text{phase gradient}]` instead, whose last
    ``rotation_bit_precision`` qubits the caller prepares with ``PreparePhaseGradientState`` and
    may share with other operations.

    """

    def __init__(self):
        """Initialize the UnitarySynthesisMapper."""
        super().__init__()
        self._settings = UnitarySynthesisMapperSettings()
        # V of the last run with its matrix hash and input width, which right_factor reuses.
        self._last_right_factor: tuple[str, int, np.ndarray] | None = None

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: ``"unitary_synthesis"``.

        """
        return "unitary_synthesis"

    def type_name(self) -> str:
        """Return the algorithm type name.

        Returns:
            str: ``"circuit_mapper"``.

        """
        return "circuit_mapper"

    def right_factor(self, unitary: UnitaryRepresentation) -> np.ndarray:
        """Return the right factor ``V`` that ``omit_right_factor`` leaves out of the dense circuit.

        After ``run()`` on the same matrix and input width, this returns the factor that the
        synthesis already computed instead of factorizing the matrix again.

        Args:
            unitary: The unitary representation that will be mapped, with the current settings.

        Returns:
            numpy.ndarray: Orthogonal matrix with one row and column per column of the matrix. The
            dense circuit with ``omit_right_factor`` realizes ``M @ V.T`` on its first columns. It
            is the identity when no qubit is added.

        Raises:
            ValueError: If the container is not a :class:`~qdk_chemistry.data.UnitaryMatrixContainer`,
                the method does not support it or has no right factor, or ``num_input_qubits`` is
                invalid for the matrix.

        """
        container, method = self._container_and_method(unitary)
        if not method.has_right_factor:
            raise ValueError(f"The '{method.name}' method has no right factor.")
        num_input_qubits = self._num_input_qubits(container, method)
        if self._last_right_factor is not None:
            matrix_hash, last_input_qubits, right_factor = self._last_right_factor
            if last_input_qubits == num_input_qubits and matrix_hash == container.content_hash(_HASH_CHARS):
                return right_factor.copy()
        return method.right_factor(container, num_input_qubits)

    def _container_and_method(self, unitary: UnitaryRepresentation) -> tuple[UnitaryMatrixContainer, _SynthesisMethod]:
        """Return the container of ``unitary`` and the selected method, which must support it."""
        container = unitary.get_container()
        if not isinstance(container, UnitaryMatrixContainer):
            raise ValueError(
                f"The {unitary.get_container_type()} container type is not supported; "
                "unitary_synthesis requires a unitary_matrix container."
            )
        method = _METHODS[self._settings.get("method")]
        if not container.is_real and not method.supports_complex:
            raise ValueError(f"The '{method.name}' method requires a real matrix, but the matrix is complex.")
        return container, method

    def _num_input_qubits(self, container: UnitaryMatrixContainer, method: _SynthesisMethod) -> int:
        """Return the input register width.

        Raises:
            ValueError: If the requested width cannot hold the columns, exceeds the register, or
                leaves more qubits to add than the method supports.

        """
        num_qubits = container.num_qubits
        smallest = (container.num_columns - 1).bit_length()
        if method.max_added_qubits is not None:
            smallest = max(smallest, num_qubits - method.max_added_qubits)
        requested = self._settings.get("num_input_qubits")
        if requested == -1:
            return smallest
        if not smallest <= requested <= num_qubits:
            raise ValueError(
                f"num_input_qubits={requested} is invalid for the '{method.name}' method on a "
                f"{num_qubits}-qubit matrix with {container.num_columns} columns. It must be at least "
                f"{smallest} and at most {num_qubits}."
            )
        return requested

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        """Synthesize the circuit of a unitary matrix or isometry.

        Args:
            unitary: The unitary representation holding a
                :class:`~qdk_chemistry.data.UnitaryMatrixContainer`.

        Returns:
            Circuit: The synthesized circuit. Its ``num_qubits`` and
            ``metadata.num_phase_gradient_ancillas`` include the caller-owned phase gradient
            when ``allocate_phase_gradient`` is disabled.

        Raises:
            ValueError: If the container is not a :class:`~qdk_chemistry.data.UnitaryMatrixContainer`,
                the method does not support it, the settings are inconsistent with the matrix, or
                ``omit_right_factor`` is set for a method without a right factor.

        """
        container, method = self._container_and_method(unitary)
        omit_right_factor = self._settings.get("omit_right_factor")
        if omit_right_factor and not method.has_right_factor:
            raise ValueError(
                f"omit_right_factor is not supported by the '{method.name}' method, which has no right factor."
            )
        num_input_qubits = self._num_input_qubits(container, method)
        rotation_bit_precision = self._settings.get("rotation_bit_precision")

        synthesis, right_factor = method.synthesize(container, num_input_qubits, omit_right_factor)
        if right_factor is not None:
            self._last_right_factor = (container.content_hash(_HASH_CHARS), num_input_qubits, right_factor.copy())
        # The order matches the Q# Params structs and the positional arguments of Make*Circuit.
        parameter: dict[str, Any] = {
            "synthesis": synthesis,
            "numInputQubits": num_input_qubits,
            "numAddedQubits": container.num_qubits - num_input_qubits,
            "rotationBitPrecision": rotation_bit_precision,
        }
        module, prefix = QSHARP_UTILS.IsometrySynthesis, method.qsharp_prefix
        params = getattr(module, f"{prefix}Params")(**parameter)
        if self._settings.get("allocate_phase_gradient"):
            qsharp_op = getattr(module, f"Make{prefix}Op")(params)
            num_gradient_ancillas = 0
        else:
            qsharp_op = getattr(module, f"Make{prefix}OpWithPhaseGradient")(params)
            num_gradient_ancillas = rotation_bit_precision
        # An exported circuit has no caller to own the gradient, so it always allocates its own.
        return Circuit(
            qsharp_op=qsharp_op,
            qsharp_factory=QsharpFactoryData(program=getattr(module, f"Make{prefix}Circuit"), parameter=parameter),
            num_qubits=container.num_qubits + num_gradient_ancillas,
            metadata=CircuitMetadata(num_phase_gradient_ancillas=num_gradient_ancillas),
        )


def _givens_to_qsharp(givens: GivensDecomposition) -> dict[str, Any]:
    """Return the fields of the Q# ``GivensDecomposition`` struct."""
    return {"layerAngles": givens.layer_angles, "layerShifted": givens.layer_shifted, "phases": givens.phases}


def _dense_to_qsharp(synthesis: DenseIsometrySynthesis) -> dict[str, Any]:
    """Return the fields of the Q# ``DenseIsometrySynthesis`` struct."""
    right_factor = synthesis.right_factor_givens
    return {
        "rotationAngles": synthesis.rotation_angles,
        "mixingGivens": [_givens_to_qsharp(givens) for givens in synthesis.mixing_givens],
        "blockGivens": _givens_to_qsharp(synthesis.block_givens),
        "rightFactorGivens": [] if right_factor is None else [_givens_to_qsharp(right_factor)],
    }


def _sparse_to_qsharp(synthesis: SparseIsometrySynthesis) -> dict[str, Any]:
    """Return the fields of the Q# ``SparseIsometrySynthesis`` struct."""
    return {
        "columnPermutation": synthesis.column_permutation,
        "rowPermutation": synthesis.row_permutation,
        "blockGivens": _givens_to_qsharp(synthesis.block_givens),
    }
