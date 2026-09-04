r"""Plaquette Trotterization of a uniform Fermi-Hubbard model on a square lattice.

Term-by-term Trotterization pays one synthesized rotation per Pauli, and a hopping
bond costs two of them under Jordan-Wigner. A four-site plaquette therefore costs
eight, even though the plaquette is a *quadratic* operator whose exact evolution
needs only two.

This builder exploits that. A plaquette's single-particle hopping matrix is
diagonalized, :math:`T = V \Lambda V^{T}`, which lifts to

.. math::
    e^{-i\tau H_\mathrm{plaq}}
        = U_V \, e^{-i\tau \sum_m \lambda_m n_m} \, U_V^{\dagger}

in Fock space. A uniform four-cycle has :math:`\Lambda = \{-2t, 2t, 0, 0\}`, so only
two modes carry a phase, and for a *uniform* cycle :math:`V` is the four-point
Fourier transform whose Givens network is three rotations at a **fixed**
:math:`\pi/4`. Fixed angles compile to Clifford+T once rather than being synthesized
per timestep, so they do not count against the rotation budget.

Both properties are needed. Emitting the network in cycle order is what keeps the
angles fixed: sorting the modes first permutes the cycle, destroys the Fourier
structure, and leaves six *arbitrary* angles that cost more than the eight Pauli
rotations this replaces.

References:
    Campbell, E. T. "Early fault-tolerant simulations of the Hubbard model."
    *Quantum Science & Technology* 7.1 (2022): 015007. arXiv:2012.09238, Appendix E.

    Kivlichan, I. D., et al. "Quantum Simulation of Electronic Structure with Linear
    Depth and Connectivity." *Physical Review Letters* 120.11 (2018): 110501.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter, TrotterSettings
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import ExponentiatedPauliTerm
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import QubitOperator

__all__: list[str] = ["PlaquetteTrotter", "PlaquetteTrotterSettings", "plaquette_sections"]

#: Radix-2 butterfly pairs of the four-point Fourier transform, in cycle-local indices.
_BUTTERFLY = ((0, 2), (1, 3), (0, 1))

#: Every butterfly rotation of a uniform four-cycle sits at this fixed angle.
_QUARTER = -math.pi / 4


def plaquette_sections(width: int, height: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Tile a periodic square lattice with two sections of vertex-disjoint four-cycles.

    The two sections are a unit plaquette translated over the 2x2 sublattice, and the
    same plaquette shifted by ``(-1, -1)``. Together they cover every bond exactly
    once, and no two cycles within a section share a site, so the cycles of one
    section commute and that section is exact.

    Args:
        width: Number of lattice columns.
        height: Number of lattice rows.

    Returns:
        Two lists of four-tuples of site indices in cycle order, where sites are
        numbered row-major as ``row * width + col``.

    Raises:
        ValueError: If either side is odd or smaller than four.

    """
    if width % 2 or height % 2:
        raise ValueError(f"Plaquette tiling requires even side lengths, got {width}x{height}.")
    if width < 4 or height < 4:
        # Below four, the periodic wrap folds the two sections onto the same cycles,
        # so every bond would be covered twice and the hopping applied twice over.
        raise ValueError(
            f"Plaquette tiling requires both sides to be at least four, got {width}x{height}. "
            "Smaller periodic lattices wrap onto themselves and cannot be tiled."
        )

    def cycle(row: int, col: int) -> tuple[int, ...]:
        def site(r: int, c: int) -> int:
            return (r % height) * width + (c % width)

        return (site(row, col), site(row, col + 1), site(row + 1, col + 1), site(row + 1, col))

    anchors = [(r, c) for r in range(0, height, 2) for c in range(0, width, 2)]
    return (
        [cycle(r, c) for r, c in anchors],
        [cycle(r + 1, c + 1) for r, c in anchors],
    )


def _givens_terms(mode_i: int, mode_j: int, theta: float) -> list[ExponentiatedPauliTerm]:
    r"""Return :math:`\exp(\theta (a_i^{\dagger} a_j - a_j^{\dagger} a_i))` as Pauli exponentials.

    The Jordan-Wigner parity string between the two modes is carried explicitly, so
    they need not be adjacent and no fermionic swap network is required. The two
    Pauli terms commute, so the product splits exactly.

    Args:
        mode_i: First mode.
        mode_j: Second mode.
        theta: Rotation angle.

    Returns:
        Two terms under the container's convention, where an ``angle`` of ``a``
        means the circuit applies :math:`\exp(-i a P)`.

    """
    low, high = min(mode_i, mode_j), max(mode_i, mode_j)
    orientation = 1.0 if mode_i < mode_j else -1.0
    string = dict.fromkeys(range(low + 1, high), "Z")
    half = orientation * theta / 2.0
    return [
        ExponentiatedPauliTerm(pauli_term={**string, low: "X", high: "Y"}, angle=-half),
        ExponentiatedPauliTerm(pauli_term={**string, low: "Y", high: "X"}, angle=half),
    ]


def _plaquette_terms(sites: tuple[int, ...], hopping: float, time: float) -> list[ExponentiatedPauliTerm]:
    r"""Return the Pauli exponentials evolving one plaquette for *time*.

    Args:
        sites: The plaquette's four modes in cycle order.
        hopping: Hopping amplitude :math:`t`.
        time: Evolution time.

    Returns:
        Fourteen terms: twelve carry the network's fixed ``pi/8`` angles and cost one
        T gate each, and two carry the eigenvalue phases and are the only ones needing
        rotation synthesis.

    """
    # The target is U_V D U_V^dagger with U_V = G1 G2 G3 as a matrix product, so the
    # factor applied first is G1^dagger: the network runs forwards inverted, then the
    # eigenvalue layer, then the network backwards.
    head: list[ExponentiatedPauliTerm] = []
    for local_i, local_j in _BUTTERFLY:
        head += _givens_terms(sites[local_i], sites[local_j], -_QUARTER)

    # exp(-i alpha n) with n = (I - Z)/2 splits into a Z rotation and an identity
    # phase. The two extremal eigenvalues are equal and opposite, so their identity
    # phases cancel exactly and only the Z rotations survive.
    kappa = 2.0 * hopping * time
    middle = [
        ExponentiatedPauliTerm(pauli_term={sites[0]: "Z"}, angle=kappa / 2.0),
        ExponentiatedPauliTerm(pauli_term={sites[1]: "Z"}, angle=-kappa / 2.0),
    ]

    tail: list[ExponentiatedPauliTerm] = []
    for local_i, local_j in reversed(_BUTTERFLY):
        tail += _givens_terms(sites[local_i], sites[local_j], _QUARTER)

    return head + middle + tail


class PlaquetteTrotterSettings(TrotterSettings):
    """Settings for the plaquette Trotter builder."""

    def __init__(self):
        """Initialize the settings, adding the lattice shape to the Trotter defaults."""
        super().__init__()
        self._set_default("lattice_width", "int", 0, "Number of lattice columns. Required.")
        self._set_default("lattice_height", "int", 0, "Number of lattice rows. Required.")


class PlaquetteTrotter(Trotter):
    """Trotter builder that evolves hopping terms a plaquette at a time.

    Only the hopping part of the Hamiltonian is re-expressed; every diagonal term
    (the on-site interaction, any chemical potential, and the identity) is emitted
    unchanged, since those already commute and cost the same either way.

    The hopping amplitude is read from the Hamiltonian rather than configured, so a
    mismatch between the operator and the declared lattice is an error rather than a
    silently wrong circuit.

    Note:
        This expects a Jordan-Wigner encoded, spin-ordered uniform Fermi-Hubbard
        model on a periodic square lattice of even side length at least four, with
        spin-up modes occupying the first half of the register. A non-uniform
        hopping, an open boundary, or a term that does not match a lattice bond
        raises rather than being approximated.

    """

    def __init__(
        self,
        order: int = 1,
        *,
        lattice_width: int = 0,
        lattice_height: int = 0,
        time: float = 0.0,
        target_accuracy: float = 0.0,
        num_divisions: int = 0,
        error_bound: str = "commutator",
        accuracy_metric: str = "unitary",
        weight_threshold: float = 1e-12,
        power: int = 1,
        power_strategy: str = "repeat",
    ):
        """Initialize the builder.

        Args:
            order: Trotter-Suzuki order. Defaults to 1.
            lattice_width: Number of lattice columns. Required at run time.
            lattice_height: Number of lattice rows. Required at run time.
            time: The evolution time. Defaults to 0.0.
            target_accuracy: Target accuracy for auto step computation. Use 0.0 to disable.
            num_divisions: Divisions per Trotter step. Max of this and the auto value is used.
            error_bound: Error bound strategy: ``"commutator"`` (default) or ``"naive"``.
            accuracy_metric: What *target_accuracy* measures, ``"unitary"`` or ``"energy"``.
            weight_threshold: Threshold for filtering small coefficients.
            power: The power to raise the unitary to. Defaults to 1.
            power_strategy: Strategy for ``U^power``: ``"rescale"`` or ``"repeat"``.

        """
        super().__init__(
            order,
            time=time,
            target_accuracy=target_accuracy,
            num_divisions=num_divisions,
            error_bound=error_bound,
            accuracy_metric=accuracy_metric,
            weight_threshold=weight_threshold,
            power=power,
            power_strategy=power_strategy,
        )
        settings = PlaquetteTrotterSettings()
        settings.set("time", time)
        settings.set("power", power)
        settings.set("power_strategy", power_strategy)
        settings.set("order", order)
        settings.set("target_accuracy", target_accuracy)
        settings.set("num_divisions", num_divisions)
        settings.set("error_bound", error_bound)
        settings.set("accuracy_metric", accuracy_metric)
        settings.set("weight_threshold", weight_threshold)
        settings.set("lattice_width", lattice_width)
        settings.set("lattice_height", lattice_height)
        self._settings = settings

    def name(self) -> str:
        """Return ``plaquette`` as the algorithm name."""
        return "plaquette"

    def _decompose_trotter_step(
        self,
        qubit_hamiltonian: QubitOperator,
        time: float,
        atol: float = 1e-12,
    ) -> list[ExponentiatedPauliTerm]:
        """Return one Trotter step, with the hopping expressed as plaquettes.

        Args:
            qubit_hamiltonian: The Hamiltonian to decompose.
            time: Duration of the step.
            atol: Threshold below which coefficients are dropped.

        Returns:
            The step's exponentiated Pauli terms.

        Raises:
            ValueError: If the lattice shape is unset or inconsistent with the
                Hamiltonian, or if the hopping is not uniform.

        """
        width = self._settings.get("lattice_width")
        height = self._settings.get("lattice_height")
        if width <= 0 or height <= 0:
            raise ValueError(
                "PlaquetteTrotter requires lattice_width and lattice_height, for example "
                "create('hamiltonian_unitary_builder', 'plaquette', lattice_width=4, lattice_height=4)."
            )

        num_sites = width * height
        if qubit_hamiltonian.num_qubits != 2 * num_sites:
            raise ValueError(
                f"A {width}x{height} spinful lattice needs {2 * num_sites} qubits, but the "
                f"Hamiltonian has {qubit_hamiltonian.num_qubits}."
            )

        hopping, diagonal = self._split_hopping(qubit_hamiltonian, atol)
        section_a, section_b = plaquette_sections(width, height)
        order = self._settings.get("order")

        def hop_layer(section, fraction):
            terms: list[ExponentiatedPauliTerm] = []
            for spin_offset in (0, num_sites):
                for cycle in section:
                    shifted = tuple(site + spin_offset for site in cycle)
                    terms += _plaquette_terms(shifted, hopping, time * fraction)
            return terms

        def diagonal_layer(fraction):
            return [
                ExponentiatedPauliTerm(pauli_term=dict(term.pauli_term), angle=term.angle * fraction)
                for term in diagonal
            ]

        if order == 1:
            return hop_layer(section_a, 1.0) + hop_layer(section_b, 1.0) + diagonal_layer(time)

        # Strang ordering, with the cheap diagonal layer at the midpoint.
        return (
            hop_layer(section_a, 0.5)
            + hop_layer(section_b, 0.5)
            + diagonal_layer(time)
            + hop_layer(section_b, 0.5)
            + hop_layer(section_a, 0.5)
        )

    def _split_hopping(self, qubit_hamiltonian, atol):
        """Separate the uniform hopping amplitude from the diagonal terms.

        Args:
            qubit_hamiltonian: The Hamiltonian to inspect.
            atol: Threshold below which coefficients are dropped.

        Returns:
            ``(hopping, diagonal)`` where *hopping* is the amplitude ``t`` and
            *diagonal* holds every non-hopping term as an
            :class:`ExponentiatedPauliTerm` scaled to unit time.

        Raises:
            ValueError: If no hopping is present or it is not uniform.

        """
        magnitudes: set[float] = set()
        diagonal: list[ExponentiatedPauliTerm] = []
        for label, coeff in qubit_hamiltonian.get_real_coefficients(tolerance=atol):
            mapping = self._pauli_label_to_map(label)
            axes = [position for position, axis in mapping.items() if axis in "XY"]
            if len(axes) == 2:
                # A Jordan-Wigner hopping bond contributes two Paulis of weight t/2.
                magnitudes.add(round(abs(coeff), 12))
            else:
                diagonal.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=coeff))

        if not magnitudes:
            raise ValueError("The Hamiltonian carries no hopping terms; nothing to tile into plaquettes.")
        if len(magnitudes) > 1:
            raise ValueError(
                f"PlaquetteTrotter requires a uniform hopping amplitude, but found "
                f"{len(magnitudes)} distinct magnitudes: {sorted(magnitudes)}."
            )

        hopping = 2.0 * next(iter(magnitudes))
        Logger.debug(f"PlaquetteTrotter: hopping t={hopping}, {len(diagonal)} diagonal terms.")
        return hopping, diagonal
