"""QDK/Chemistry CSWAP-sandwich controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import qsharp

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.utils.pauli_commutation import do_pauli_maps_commute
from qdk_chemistry.utils.pauli_qubit_flip import pauli_map_zero_state_action
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper, ControlledCircuitMapperSettings

__all__: list[str] = [
    "ControlledSwapPauliSequenceMapper",
    "ControlledSwapPauliSequenceMapperSettings",
    "vacuum_preserving_blocks",
]


def vacuum_preserving_blocks(
    terms: list[tuple[dict[int, str], float]],
    *,
    atol: float = 1e-9,
) -> list[tuple[int, ...]] | None:
    r"""Split an *ordered* sequence of exponentiated Pauli terms into vacuum-preserving blocks.

    A product :math:`\prod_j e^{-i\theta_j P_j}` fixes :math:`|0\ldots0\rangle` (up to a
    phase) when it can be cut into contiguous blocks in which

    #. all Pauli strings pairwise commute, so the block equals
       :math:`\exp(-i\sum_j \theta_j P_j)`, and
    #. the block generator :math:`\sum_j \theta_j P_j` maps the all-zero state onto a
       multiple of itself, i.e. amplitudes cancel for every non-empty flipped-qubit set.

    Blocks are cut as early as possible, giving the finest valid partition: any coarser
    block contains a finer one and so inherits its non-commuting pairs.

    Args:
        terms: Ordered ``(pauli_map, angle)`` pairs, one per exponential factor.
        atol: Absolute tolerance used when testing amplitude cancellation.

    Returns:
        A list of blocks, each a tuple of indices into *terms*, or ``None`` when no such
        split exists (i.e. the ordering does not preserve the vacuum).

    """
    blocks: list[tuple[int, ...]] = []
    current: list[int] = []
    residual: dict[frozenset[int], complex] = {}

    for index, (pauli_map, angle) in enumerate(terms):
        flipped, amplitude = pauli_map_zero_state_action(pauli_map)
        current.append(index)
        if flipped:
            residual[flipped] = residual.get(flipped, 0j) + angle * amplitude
            if abs(residual[flipped]) <= atol:
                del residual[flipped]

        if residual:
            continue

        # The accumulated generator maps the vacuum onto a multiple of itself;
        # the block is only usable if its factors also commute.
        if not _pairwise_commuting(terms, current):
            return None
        blocks.append(tuple(current))
        current = []

    if current or residual:
        return None
    return blocks


def _pairwise_commuting(terms: list[tuple[dict[int, str], float]], indices: list[int]) -> bool:
    """Check that every pair of Pauli terms referenced by *indices* commutes."""
    # Explicit "I" axes are dropped: do_pauli_maps_commute miscounts them in sparse maps.
    maps = {i: {q: axis for q, axis in terms[i][0].items() if axis != "I"} for i in indices}
    for position, i in enumerate(indices):
        for j in indices[position + 1 :]:
            if not do_pauli_maps_commute(maps[i], maps[j]):
                return False
    return True


class ControlledSwapPauliSequenceMapperSettings(ControlledCircuitMapperSettings):
    """Settings for the :class:`ControlledSwapPauliSequenceMapper`.

    Attributes:
        vacuum_preservation_tolerance: Absolute tolerance used when checking that the
            Pauli amplitudes within a block cancel on the vacuum.

    """

    def __init__(self):
        """Initialize the settings for ControlledSwapPauliSequenceMapper."""
        super().__init__()
        self._set_default(
            "vacuum_preservation_tolerance",
            "double",
            1e-9,
            "Absolute tolerance for the vacuum-preservation validation of the input product formula.",
        )


class ControlledSwapPauliSequenceMapper(ControlledCircuitMapper):
    r"""Controlled evolution circuit mapper using a CSWAP-sandwich construction.

    Given a time evolution as a Pauli product formula
    :math:`U(t) \approx \left[ U_{\mathrm{step}}(t / r) \right]^{r}`, this mapper builds a
    controlled :math:`U(t)` without controlling every gate.  An internally allocated
    ``vacuum`` register (:math:`|0\ldots0\rangle`) is conditionally swapped with the system,
    the *uncontrolled* evolution runs on the vacuum (``step_reps`` times), and the swap is
    uncomputed.  On the :math:`|0\rangle` branch the evolution hits the vacuum and the system
    is untouched; on :math:`|1\rangle` the system is parked in the vacuum and evolved, so the
    eigenphase accumulates on the :math:`|1\rangle` branch as usual.  The cost of controlling
    every gate is traded for one layer of controlled-:math:`\mathrm{SWAP}`.

    **Hamiltonian restriction.** Exact **only if** :math:`U|0\ldots0\rangle = \lambda|0\ldots0\rangle`
    with :math:`|\lambda| = 1`; otherwise the vacuum leaks, the control decoheres, and the phase
    is lost.  The :math:`|0\rangle` branch picks up a residual phase
    :math:`\varphi_0 = \arg\lambda = -E_0 t`, where
    :math:`E_0 = \langle 0\ldots0|H|0\ldots0\rangle`.  All fermion-to-qubit mappers supported
    here exclude the core energy, so :math:`E_0 = 0` and that phase vanishes.

    **Grouping requirement.** Each fermionic term annihilates the vacuum, but its individual
    Pauli strings cannot (they are unitary) — only their weighted sum does.  A Trotterised
    :math:`U` therefore fixes the vacuum only when strings from the same fermionic term are
    exponentiated as one contiguous, mutually commuting block:

    .. math::

        U|0\ldots0\rangle = e^{-it\sum_i P_i}|0\ldots0\rangle
        \approx \prod_i e^{-it P_i}|0\ldots0\rangle = |0\ldots0\rangle .

    Building the product formula from a Hamiltonian grouped with the ``qubit_flip`` term
    grouper (:class:`~qdk_chemistry.algorithms.term_grouper.QubitFlipTermGrouper`) guarantees
    this.  The incoming formula is validated and rejected when the ordering is not vacuum
    preserving, since the sandwich would otherwise return a silently decohered result.
    Interleaving cancellation partners — say ``XX, Z0, YY, I`` for
    :math:`H = \tfrac12(XX + YY) + \tfrac12(I - Z_0)` — leaks half the vacuum amplitude.

    Notes:
        * Currently supports only single-control-qubit scenarios.
        * Requires a ``PauliProductFormulaContainer`` for the time evolution unitary.
        * The vacuum register is allocated internally by the Q# operation.

    """

    def __init__(self):
        """Initialize the ControlledSwapPauliSequenceMapper."""
        super().__init__()
        self._settings = ControlledSwapPauliSequenceMapperSettings()

    def name(self) -> str:
        """Return the algorithm name."""
        return "cswap_pauli_sequence"

    def type_name(self) -> str:
        """Return controlled_circuit_mapper as the algorithm type name."""
        return "controlled_circuit_mapper"

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct a quantum circuit implementing the controlled unitary.

        Args:
            unitary: The unitary representation containing the Hamiltonian and evolution parameters.
            Control and target indices are read from settings.

        Returns:
            Circuit: A quantum circuit implementing the controlled unitary :math:`U` via the CSWAP sandwich,
            where :math:`U` is the time evolution operator :math:`\exp(-i H t)`.

        Raises:
            ValueError: If the unitary container type is not supported.
            ValueError: If multiple control qubits are provided.
            ValueError: If the product formula ordering is not vacuum preserving.

        """
        unitary_container = unitary.get_container()
        if not isinstance(unitary_container, PauliProductFormulaContainer):
            raise ValueError(
                f"The {unitary.get_container_type()} container type is not supported. "
                "ControlledSwapPauliSequenceMapper only supports PauliProductFormula container for the unitary."
            )

        self._validate_vacuum_preserving(unitary_container)

        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("ControlledSwapPauliSequenceMapper currently only supports a single control qubit.")

        target_indices = self._get_target_indices(unitary)

        pauli_terms: list[list[qsharp.Pauli]] = []
        angles: list[float] = []
        for term in unitary_container.step_terms:
            base_terms = [qsharp.Pauli.I] * unitary_container.num_qubits
            for index, pauli in term.pauli_term.items():
                base_terms[index] = getattr(qsharp.Pauli, pauli)
            pauli_terms.append(base_terms.copy())
            angles.append(term.angle)

        controlled_evo_params = QSHARP_UTILS.ControlledSwapPauliExp.ControlledSwapPauliExpParams(
            pauliExponents=pauli_terms,
            pauliCoefficients=angles,
            repetitions=unitary_container.step_reps,
            control=control_indices[0],
            systems=target_indices,
        )

        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.ControlledSwapPauliExp.MakeRepControlledSwapPauliExpCircuit,
            parameter=vars(controlled_evo_params),
        )

        controlled_unitary_op = QSHARP_UTILS.ControlledSwapPauliExp.MakeRepControlledSwapPauliExpOp(
            controlled_evo_params
        )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=controlled_unitary_op)

    def _validate_vacuum_preserving(self, container: PauliProductFormulaContainer) -> None:
        r"""Check that the product formula leaves the vacuum invariant.

        The sandwich evolves an internally allocated vacuum register and resets it at
        the end.  If the evolution moves amplitude out of :math:`|0\ldots0\rangle`, the
        vacuum stays entangled with the control and the reset destroys the control
        coherence, silently corrupting the phase.  Safe when the terms split into
        contiguous, mutually commuting blocks whose generators map the vacuum onto a
        multiple of itself.

        Args:
            container: The Pauli product formula to validate.

        Raises:
            ValueError: If no such split exists.

        """
        terms = [(term.pauli_term, term.angle) for term in container.step_terms]
        tolerance = self._settings.get("vacuum_preservation_tolerance")
        if vacuum_preserving_blocks(terms, atol=tolerance) is None:
            raise ValueError(
                "ControlledSwapPauliSequenceMapper requires a vacuum-preserving product formula: the "
                "Pauli terms could not be split into contiguous, mutually commuting blocks that leave "
                "|0...0> invariant, so the CSWAP sandwich would leak the vacuum and decohere the control. "
                "Group the Hamiltonian with the 'qubit_flip' term grouper before building the "
                "unitary, e.g. "
                "registry.create('term_grouper', 'qubit_flip').run(qubit_hamiltonian)."
            )
