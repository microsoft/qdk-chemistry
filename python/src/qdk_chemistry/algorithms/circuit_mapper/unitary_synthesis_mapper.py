"""QDK/Chemistry circuit mapper for real orthogonal matrices and isometries."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qdk_chemistry._core.utils.unitary_synthesis import (
    block_sparse_isometry_synthesis,
    dense_isometry_right_factor,
    dense_isometry_synthesis,
)
from qdk_chemistry.data import Settings
from qdk_chemistry.data.circuit import Circuit, CircuitMetadata, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.real_unitary import RealUnitaryContainer
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

_MAX_ADDED_QUBITS = 2  # Most qubits the dense synthesis adds to its input register.


class UnitarySynthesisMapperSettings(Settings):
    """Settings for the UnitarySynthesisMapper."""

    def __init__(self):
        """Initialize the settings for UnitarySynthesisMapper."""
        super().__init__()
        self._set_default(
            "method",
            "string",
            "dense",
            "Synthesis method: 'dense' (cosine-sine decomposition) or 'block_sparse' (permutations "
            "around a block-diagonal orthogonal matrix).",
            ["dense", "block_sparse"],
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
            "Input register width of the dense method, the low qubits of the target. The other "
            "zero to two qubits are added in |0>. -1 selects the smallest register that holds the "
            "columns.",
            (-1, 62),
        )


class UnitarySynthesisMapper(CircuitMapper):
    r"""Circuit mapper that synthesizes a real orthogonal matrix or isometry.

    The input is a :class:`~qdk_chemistry.data.UnitaryRepresentation` holding a
    :class:`~qdk_chemistry.data.RealUnitaryContainer` with matrix :math:`M` of :math:`2^n` rows
    and :math:`m` orthonormal columns. The circuit acts on :math:`n` little-endian qubits and
    realizes an orthogonal :math:`U` whose first :math:`m` columns equal :math:`M`.

    Methods:
        - ``"dense"``: isometry synthesis by cosine-sine decompositions :cite:`Rupprecht2026`.
          The low ``num_input_qubits`` qubits form the input register and the remaining
          :math:`k \le 2` qubits start in :math:`|0\rangle`, so
          :math:`m \le 2^{\text{num\_input\_qubits}}`. The first decomposition leaves a right
          factor :math:`V` on the input register; with ``omit_right_factor`` the circuit drops it
          and realizes :math:`M V^T` on its first :math:`m` columns, and :meth:`right_factor`
          returns :math:`V`.
        - ``"block_sparse"``: :math:`U = P_r B P_c` with row and column permutations
          :math:`P_r, P_c` loaded by QROAM :cite:`Berry2025` and a block-diagonal orthogonal
          :math:`B` that couples only columns with a common nonzero row.

    All orthogonal factors are applied as layers of Givens rotations :cite:`Clements2016` whose
    angles, like the multiplexed rotations, are quantized to ``rotation_bit_precision`` bits and
    applied by phase-gradient addition.

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

        Args:
            unitary: The unitary representation that will be mapped, with the current settings.

        Returns:
            numpy.ndarray: Orthogonal matrix with one row and column per column of the matrix. The
            dense circuit with ``omit_right_factor`` realizes ``M @ V.T`` on its first columns. It
            is the identity when no qubit is added.

        Raises:
            ValueError: If the container is not a :class:`RealUnitaryContainer`, the method is not
                ``"dense"``, or ``num_input_qubits`` is invalid for the matrix.

        """
        container = self._container(unitary)
        if self._settings.get("method") != "dense":
            raise ValueError("Only the dense method has a right factor.")
        num_input_qubits = self._num_input_qubits(container)
        return dense_isometry_right_factor(container.to_dense(), 1 << num_input_qubits)

    @staticmethod
    def _container(unitary: UnitaryRepresentation) -> RealUnitaryContainer:
        """Return the container of ``unitary``, which must be a :class:`RealUnitaryContainer`."""
        container = unitary.get_container()
        if not isinstance(container, RealUnitaryContainer):
            raise ValueError(
                f"The {unitary.get_container_type()} container type is not supported; "
                "unitary_synthesis requires a real_unitary container."
            )
        return container

    def _num_input_qubits(self, container: RealUnitaryContainer) -> int:
        """Return the input register width of the dense method.

        Raises:
            ValueError: If the requested width cannot hold the columns or leaves more than two
                added qubits.

        """
        num_qubits = container.num_qubits
        min_input_qubits = (container.num_columns - 1).bit_length()
        requested = self._settings.get("num_input_qubits")
        if requested == -1:
            return max(min_input_qubits, num_qubits - _MAX_ADDED_QUBITS)
        if requested > num_qubits or requested < num_qubits - _MAX_ADDED_QUBITS or requested < min_input_qubits:
            raise ValueError(
                f"num_input_qubits={requested} is invalid for a {num_qubits}-qubit matrix with "
                f"{container.num_columns} columns. It must be at least "
                f"{max(min_input_qubits, num_qubits - _MAX_ADDED_QUBITS)} and at most {num_qubits}."
            )
        return requested

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        """Synthesize the circuit of a real orthogonal matrix or isometry.

        Args:
            unitary: The unitary representation holding a :class:`RealUnitaryContainer`.

        Returns:
            Circuit: The synthesized circuit. Its ``num_qubits`` and
            ``metadata.num_phase_gradient_ancillas`` include the caller-owned phase gradient
            when ``allocate_phase_gradient`` is disabled.

        Raises:
            ValueError: If the container is not a :class:`RealUnitaryContainer`, the settings are
                inconsistent with the matrix, or ``omit_right_factor`` is set for the
                ``"block_sparse"`` method.

        """
        container = self._container(unitary)
        method = self._settings.get("method")
        rotation_bit_precision = self._settings.get("rotation_bit_precision")
        num_qubits = container.num_qubits
        module = QSHARP_UTILS.IsometrySynthesis

        if method == "dense":
            num_input_qubits = self._num_input_qubits(container)
            synthesis = dense_isometry_synthesis(
                container.to_dense(),
                1 << num_input_qubits,
                omit_right_factor=self._settings.get("omit_right_factor"),
            )
            parameter: dict[str, Any] = {
                "synthesis": _dense_to_qsharp(synthesis),
                "numInputQubits": num_input_qubits,
                "numAddedQubits": num_qubits - num_input_qubits,
                "rotationBitPrecision": rotation_bit_precision,
            }
            program, params_type = module.MakeDenseIsometryCircuit, module.DenseIsometryParams
            make_op, make_op_with_phase_gradient = (
                module.MakeDenseIsometryOp,
                module.MakeDenseIsometryOpWithPhaseGradient,
            )
        elif method == "block_sparse":
            if self._settings.get("omit_right_factor"):
                raise ValueError("omit_right_factor applies only to the dense method.")
            synthesis = block_sparse_isometry_synthesis(container.matrix)
            parameter = {
                "synthesis": _sparse_to_qsharp(synthesis),
                "numQubits": num_qubits,
                "rotationBitPrecision": rotation_bit_precision,
            }
            program, params_type = module.MakeSparseIsometryCircuit, module.SparseIsometryParams
            make_op, make_op_with_phase_gradient = (
                module.MakeSparseIsometryOp,
                module.MakeSparseIsometryOpWithPhaseGradient,
            )
        else:
            raise ValueError(f"Unsupported unitary synthesis method '{method}'.")

        if self._settings.get("allocate_phase_gradient"):
            qsharp_op = make_op(params_type(**parameter))
            num_gradient_ancillas = 0
        else:
            qsharp_op = make_op_with_phase_gradient(params_type(**parameter))
            num_gradient_ancillas = rotation_bit_precision
        # An exported circuit has no caller to own the gradient, so it always allocates its own.
        return Circuit(
            qsharp_op=qsharp_op,
            qsharp_factory=QsharpFactoryData(program=program, parameter=parameter),
            num_qubits=num_qubits + num_gradient_ancillas,
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
