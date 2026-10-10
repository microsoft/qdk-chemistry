"""QDK/Chemistry sequence structure controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.circuit_mapper.pauli_sequence_mapper import _pauli_evolution_parameters
from qdk_chemistry.data import Settings
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper, ControlledCircuitMapperSettings

__all__: list[str] = ["ControlledPauliSequenceMapper", "ControlledPauliSequenceMapperSettings"]


class ControlledPauliSequenceMapperSettings(ControlledCircuitMapperSettings):
    r"""Settings for :class:`ControlledPauliSequenceMapper`.

    Attributes:
        max_hamming_weight_phasing_batch_size: Largest tower of equal-angle rotations in a declared
            layer phased through a single Hamming-weight register, or ``-1`` for no cap. A cap below
            8 (e.g. ``1``) turns Hamming-weight phasing off, so every term is its own rotation.
            Defaults to ``1``, which leaves phasing off until a cap of at least 8 or ``-1`` is set.

    """

    def __init__(self):
        """Initialize the settings, adding the Hamming-weight phasing batch cap."""
        super().__init__()
        self._set_default(
            "max_hamming_weight_phasing_batch_size",
            "int",
            1,
            "Largest tower of equal-angle rotations in a declared layer phased through a single "
            "Hamming-weight register. A shorter batch releases its adder-tree scratch sooner, so the "
            "peak ancilla count follows the batch rather than the whole tower, at the cost of one "
            "extra set of place-value rotations per batch. A cap below 8, such as the default 1, turns "
            "Hamming-weight phasing off, so every term is its own rotation. Set to -1 for no cap.",
        )


def _max_hamming_weight_phasing_batch_size(settings: Settings) -> int:
    """Return the validated Hamming-weight phasing batch cap.

    Args:
        settings: Settings holding ``max_hamming_weight_phasing_batch_size``.

    Returns:
        The cap, ``-1`` meaning no cap.

    Raises:
        ValueError: If the cap is neither -1 nor positive.

    """
    max_batch_size = int(settings.get("max_hamming_weight_phasing_batch_size"))
    if max_batch_size != -1 and max_batch_size < 1:
        raise ValueError(
            f"max_hamming_weight_phasing_batch_size must be -1 or a positive integer. Got {max_batch_size}."
        )
    return max_batch_size


class ControlledPauliSequenceMapper(ControlledCircuitMapper):
    r"""Controlled evolution circuit mapper using Pauli product formula term sequences.

    Given a time-evolution operator expressed as a Pauli product formula
    :math:`U(t) \approx \left[ U_{\mathrm{step}}(t / r) \right]^{r}`, this mapper constructs
    a controlled version of :math:`U(t)` using the following pattern:

    1. Each Pauli operator :math:`P_j` is basis-rotated into the :math:`Z` basis.
    2. Qubits involved in :math:`P_j` are entangled into a sequence using CNOT gates.
    3. A controlled :math:`R_z` rotation implements
        :math:`e^{-i\,\theta_j\,P_j} \;\rightarrow\; \text{CRZ}(2 \theta_j)`.
    4. The basis rotations and entangling operations are uncomputed.

    When the formula declares disjoint layers, their controlled rotations share two
    rotation rounds per layer. The declared boundaries are used without regrouping;
    formulas without layer metadata retain term-by-term controlled evolution.

    Within a declared layer, terms that share a rotation angle form a tower once at least 8 of
    them agree, the break-even of the adder tree. Each tower is synthesized with Hamming-weight
    phasing :cite:`Kan2025`: an adder tree writes the Hamming weight of the rotated qubits into a
    scratch register, and one controlled rotation per place value replaces the per-term
    rotations. ``max_hamming_weight_phasing_batch_size`` caps the tower phased through one
    register; a cap below 8, such as the default ``1``, turns phasing off.

    Notes:
        * Currently supports only single-control-qubit scenarios.
        * Requires a ``PauliProductFormulaContainer`` for the time evolution unitary.

    """

    def __init__(self):
        """Initialize the PauliSequenceMapper."""
        super().__init__()
        self._settings = ControlledPauliSequenceMapperSettings()

    def name(self) -> str:
        """Return the algorithm name."""
        return "pauli_sequence"

    def type_name(self) -> str:
        """Return controlled_circuit_mapper as the algorithm type name."""
        return "controlled_circuit_mapper"

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct a quantum circuit implementing the controlled unitary.

        Args:
            unitary: The unitary representation containing the Hamiltonian
                and evolution parameters. Control and target indices are
                read from settings.

        Returns:
            Circuit: A quantum circuit implementing the controlled unitary :math:`U`
            where :math:`U` is the time evolution operator :math:`\exp(-i H t)`.

        Raises:
            ValueError: If the unitary container type is not supported.
            ValueError: If multiple control qubits are provided.
            ValueError: If ``max_hamming_weight_phasing_batch_size`` is neither -1 nor positive.

        """
        unitary_container = unitary.get_container()
        if not isinstance(unitary_container, PauliProductFormulaContainer):
            raise ValueError(
                f"The {unitary.get_container_type()} container type is not supported. "
                "PauliSequenceMapper only supports PauliProductFormula container for the unitary."
            )

        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("PauliSequenceMapper currently only supports a single control qubit.")

        max_batch_size = _max_hamming_weight_phasing_batch_size(self._settings)
        target_indices = self._get_target_indices(unitary)

        evo_params = QSHARP_UTILS.PauliExp.RepPauliExpParams(**_pauli_evolution_parameters(unitary_container))
        layer_offsets = list(unitary_container.layer_offsets or ())

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.ControlledPauliExp.MakeRepControlledPauliExpCircuit,
            parameter={
                "params": evo_params,
                "layerOffsets": layer_offsets,
                "maxBatchSize": max_batch_size,
                "control": control_indices[0],
                "systems": target_indices,
            },
        )

        controlled_unitary_op = QSHARP_UTILS.ControlledPauliExp.MakeRepControlledPauliExpOp(
            evo_params, layer_offsets, max_batch_size
        )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=controlled_unitary_op)
