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
        use_hamming_weight_phasing: Whether to phase each declared layer's equal-angle rotations
            through Hamming-weight registers. Pays off only when layers hold many equal angles.
            Defaults to ``False``.
        max_hamming_weight_phasing_batch_size: Largest batch phased through one register, or ``-1``
            (default) for no cap. A smaller cap needs fewer scratch qubits but more rotations; use
            at least 8, as smaller batches save too few rotations. Ignored when phasing is off.

    """

    def __init__(self):
        """Initialize the settings with the Hamming-weight phasing switch and batch cap."""
        super().__init__()
        self._set_default(
            "use_hamming_weight_phasing",
            "bool",
            False,
            "Whether to phase each declared layer's equal-angle rotations through Hamming-weight "
            "registers. Pays off only when layers hold many equal angles.",
        )
        self._set_default(
            "max_hamming_weight_phasing_batch_size",
            "int",
            -1,
            "Largest batch phased through one register, or -1 for no cap. A smaller cap needs fewer "
            "scratch qubits but more rotations; use at least 8, as smaller batches save too few "
            "rotations. Ignored when phasing is off.",
        )


def _hamming_weight_phasing_parameters(
    settings: Settings, container: PauliProductFormulaContainer
) -> tuple[bool, int, list[int]]:
    """Return the phasing switch, batch cap and layer offsets, validated only when phasing is on.

    Args:
        settings: The mapper settings.
        container: The product formula to map.

    Returns:
        Whether phasing is on, the cap (``-1`` meaning no cap) and the declared layer offsets.

    Raises:
        ValueError: If phasing is on and the cap is neither -1 nor positive.
        ValueError: If phasing is on and the formula declares no ``layer_offsets``.

    """
    use_phasing = bool(settings.get("use_hamming_weight_phasing"))
    max_batch_size = int(settings.get("max_hamming_weight_phasing_batch_size"))
    layer_offsets = list(container.layer_offsets or ())
    if use_phasing and max_batch_size != -1 and max_batch_size < 1:
        raise ValueError(
            f"max_hamming_weight_phasing_batch_size must be -1 or a positive integer. Got {max_batch_size}."
        )
    if use_phasing and not layer_offsets:
        raise ValueError(
            "use_hamming_weight_phasing needs a product formula with layer_offsets, but this formula declares none."
        )
    return use_phasing, max_batch_size, layer_offsets


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

    With ``use_hamming_weight_phasing``, each declared layer's equal-angle rotations are phased
    together with Hamming-weight phasing :cite:`Kan2025`. Phasing is off by default.

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
            ValueError: If phasing is on and the batch cap is neither -1 nor positive.
            ValueError: If phasing is on and the formula declares no ``layer_offsets``.

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

        use_hamming_weight_phasing, max_batch_size, layer_offsets = _hamming_weight_phasing_parameters(
            self._settings, unitary_container
        )
        target_indices = self._get_target_indices(unitary)

        evo_params = QSHARP_UTILS.PauliExp.RepPauliExpParams(**_pauli_evolution_parameters(unitary_container))

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.ControlledPauliExp.MakeRepControlledPauliExpCircuit,
            parameter={
                "params": evo_params,
                "layerOffsets": layer_offsets,
                "useHammingWeightPhasing": use_hamming_weight_phasing,
                "maxBatchSize": max_batch_size,
                "control": control_indices[0],
                "systems": target_indices,
            },
        )

        controlled_unitary_op = QSHARP_UTILS.ControlledPauliExp.MakeRepControlledPauliExpOp(
            evo_params, layer_offsets, use_hamming_weight_phasing, max_batch_size
        )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=controlled_unitary_op)
