"""QDK/Chemistry CSWAP-sandwich controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

from qdk import qsharp

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper, ControlledCircuitMapperSettings

__all__: list[str] = [
    "ControlledSwapPauliSequenceMapper",
    "ControlledSwapPauliSequenceMapperSettings",
]


def _vacuum_eigenphase(terms: list[tuple[dict[int, str], float]], atol: float) -> float | None:
    r"""Return the phase an *ordered* product :math:`\prod_j e^{-i\theta_j P_j}` imprints on the vacuum.

    That is :math:`\varphi_0` in
    :math:`\prod_j e^{-i\theta_j P_j}|0\ldots0\rangle = e^{i\varphi_0}|0\ldots0\rangle`.
    The vacuum stays an eigenstate when the product splits into contiguous blocks of pairwise
    commuting strings, so each block equals :math:`\exp(-i\sum_j \theta_j P_j)`, whose
    generator maps the all-zero state onto a multiple of itself, i.e. amplitudes cancel for every
    non-empty flipped-qubit set.  Exact cancellations close a block immediately; otherwise a block
    closes at the first non-commuting boundary or at the end, where its residual is charged against
    ``atol``.  The phase then comes from the diagonal (:math:`I`/:math:`Z`) terms alone.

    Args:
        terms: Ordered ``(pauli_map, angle)`` pairs, one per exponential factor.
        atol: Absolute tolerance on the amplitude leaked out of the vacuum, aggregated over every
            flipped-qubit set the product formula closes.

    Returns:
        The imprinted phase, or ``None`` when the vacuum is not an eigenstate of the product.

    """
    block: list[tuple[int, int]] = []  # symplectic (X, Z) bit masks of the terms in the open block
    contributions: dict[int, list[complex]] = {}
    leaked: list[float] = []
    diagonal: list[float] = []

    def close_block() -> bool:
        """Charge whatever the open block still leaves outstanding against the shared budget."""
        for pending in contributions.values():
            residual = complex(math.fsum(c.real for c in pending), math.fsum(c.imag for c in pending))
            leaked.append(abs(residual))
        return math.fsum(leaked) <= atol

    for pauli_map, angle in terms:
        x_mask = z_mask = 0
        for qubit, axis in pauli_map.items():
            if axis in ("X", "Y"):
                x_mask |= 1 << qubit
            if axis in ("Y", "Z"):
                z_mask |= 1 << qubit

        # Two Pauli strings commute iff they anticommute on an even number of qubits.
        if any(((x_mask & z) ^ (z_mask & x)).bit_count() % 2 for x, z in block):
            if not close_block():
                return None
            block.clear()
            contributions.clear()
        block.append((x_mask, z_mask))

        if x_mask:
            # P|0...0> = i^{n_Y}|b>, with b the bit pattern of the X/Y support.
            pending = contributions.setdefault(x_mask, [])
            pending.append(angle * 1j ** (x_mask & z_mask).bit_count())
            residual = complex(math.fsum(c.real for c in pending), math.fsum(c.imag for c in pending))
            # Tolerance is charged when the complete block closes, not at a prefix boundary.
            if residual == 0.0:
                del contributions[x_mask]
        else:
            # Diagonal strings act as +1 on |0...0>, so they contribute phase only.
            diagonal.append(angle)

        if not contributions:
            block.clear()

    if not close_block():
        return None
    return -math.fsum(diagonal)


class ControlledSwapPauliSequenceMapperSettings(ControlledCircuitMapperSettings):
    """Settings for the :class:`ControlledSwapPauliSequenceMapper`.

    Attributes:
        vacuum_preservation_tolerance: Absolute tolerance on the amplitude leaked out of the vacuum,
            aggregated over every flipped-qubit set and over all ``step_reps`` repetitions.

    """

    def __init__(self):
        """Initialize the settings for ControlledSwapPauliSequenceMapper."""
        super().__init__()
        self._set_default(
            "vacuum_preservation_tolerance",
            "double",
            1e-9,
            "Absolute tolerance on the total amplitude the repeated evolution may leak out of the vacuum.",
        )


class ControlledSwapPauliSequenceMapper(ControlledCircuitMapper):
    r"""Controlled evolution circuit mapper using a CSWAP-sandwich construction.

    Given a time evolution as a Pauli product formula
    :math:`U(t) \approx \left[ U_{\mathrm{step}}(t / r) \right]^{r}`, this mapper builds a
    controlled :math:`U(t)` without controlling every gate.  An internally allocated
    ``vacuum`` register (:math:`|0\ldots0\rangle`) is conditionally swapped with the system,
    the *uncontrolled* evolution runs on the vacuum (``step_reps`` times), and the swap is
    uncomputed.  The eigenphase accumulates on the :math:`|1\rangle` control branch, as with
    a directly controlled evolution, for the cost of one layer of controlled-:math:`\mathrm{SWAP}`.

    **Vacuum phase.** The :math:`|0\rangle` branch acquires
    :math:`U|0\ldots0\rangle = e^{i\varphi_0}|0\ldots0\rangle` with :math:`\varphi_0 = -E_0 t`
    and :math:`E_0 = \langle 0\ldots0|H|0\ldots0\rangle`.  Only the diagonal (:math:`I`/:math:`Z`)
    terms of the product formula contribute, so :math:`\varphi_0` is known classically and is
    cancelled by an :math:`R_1(\varphi_0)` on the control.  The circuit is then a genuine
    :math:`C\text{-}U` up to a global phase for any :math:`E_0`.

    **Grouping requirement.** The vacuum must stay an eigenstate, which is what particle
    conservation buys: :math:`H` cannot connect :math:`|0\ldots0\rangle` to any other occupation
    number.  Leaked amplitude entangles the vacuum register with the control, and the final reset
    destroys the control coherence.  A fermionic term annihilates the vacuum only through the
    weighted sum of its Pauli strings, so a Trotterised :math:`U` preserves the vacuum only when
    those strings are exponentiated as one contiguous, mutually commuting block:

    .. math::

        U|0\ldots0\rangle = e^{-it\sum_i P_i}|0\ldots0\rangle
        \approx \prod_i e^{-it P_i}|0\ldots0\rangle = |0\ldots0\rangle .

    Grouping the Hamiltonian with the ``vacuum_annihilating`` term grouper
    (:class:`~qdk_chemistry.algorithms.term_grouper.VacuumAnnihilatingTermGrouper`) produces that ordering.
    The incoming formula is validated and rejected otherwise; interleaving cancellation partners,
    say ``XX, Z0, YY, I`` for :math:`H = \tfrac12(XX + YY) + \tfrac12(I - Z_0)`, leaks half the
    vacuum amplitude.

    Notes:
        * Applies to particle-conserving Hamiltonians.
        * The requirement is on the mapped operator, not the encoding: after qubit tapering the
          all-zero state belongs to the retained sector, which the Hamiltonian need not annihilate.
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

        vacuum_phase = self._vacuum_phase(unitary_container)

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

        controlled_evo_params = QSHARP_UTILS.ControlledSwapPauliExp.RepControlledSwapPauliExpParams(
            pauliExponents=pauli_terms,
            pauliCoefficients=angles,
            repetitions=unitary_container.step_reps,
            vacuumPhase=vacuum_phase,
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

    def _vacuum_phase(self, container: PauliProductFormulaContainer) -> float:
        r"""Return the phase the repeated evolution imprints on the vacuum register.

        The vacuum must remain an eigenstate: leaked amplitude entangles the vacuum register
        with the control, and the final reset then destroys the control coherence.

        Args:
            container: The Pauli product formula to validate.

        Returns:
            The phase accumulated over all ``step_reps`` repetitions.

        Raises:
            ValueError: If the vacuum is not an eigenstate of the product formula.

        """
        terms = [(term.pauli_term, term.angle) for term in container.step_terms]
        # A residual left by one step leaks again on every repetition, so the per-step budget shrinks.
        atol = self._settings.get("vacuum_preservation_tolerance") / container.step_reps
        phase = _vacuum_eigenphase(terms, atol)
        if phase is None:
            raise ValueError(
                "ControlledSwapPauliSequenceMapper requires a vacuum-preserving product formula: the "
                "Pauli terms could not be split into contiguous, mutually commuting blocks that leave "
                "|0...0> invariant, so the CSWAP sandwich would leak the vacuum and decohere the control. "
                "The mapper applies to particle-conserving Hamiltonians; group such a Hamiltonian with "
                "the 'vacuum_annihilating' term grouper before building the unitary, e.g. "
                "registry.create('term_grouper', 'vacuum_annihilating').run(qubit_hamiltonian)."
            )
        return container.step_reps * phase
