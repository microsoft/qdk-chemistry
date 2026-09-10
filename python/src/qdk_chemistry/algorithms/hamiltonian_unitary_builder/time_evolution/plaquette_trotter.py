r"""Second-order plaquette Trotterization for the uniform Fermi-Hubbard model.

The builder reads a Jordan-Wigner encoded :class:`~qdk_chemistry.data.QubitOperator`
and splits it into diagonal terms and two tilings of the hopping terms by disjoint
four-site plaquettes. One step applies the symmetric product

.. math::
    e^{-isH_D/2} e^{-isH_A/2} e^{-isH_B} e^{-isH_A/2} e^{-isH_D/2}.

Each plaquette hopping evolution is exact. The builder diagonalizes its
single-particle matrix, :math:`T = V \Lambda V^\dagger`, and applies

.. math::
    e^{-isH_\square} = U_V e^{-is\sum_m \lambda_m n_m} U_V^\dagger.

Thus the circuit switches to the plaquette momentum basis, applies two nonzero
eigenvalue phases, and switches back. Trotter error comes only from splitting the
three Hamiltonian layers. The lattice dimensions are supplied as settings; the
builder derives the plaquettes from them and verifies that they match the input
Hamiltonian.

References:
    Campbell, E. T. "Early fault-tolerant simulations of the Hubbard model."
    *Quantum Science & Technology* 7.1 (2022): 015007. arXiv:2012.09238v4.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter, TrotterSettings
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    MIN_USEFUL_BATCH,
    ExponentiatedPauliTerm,
)
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import QubitOperator

__all__: list[str] = [
    "PlaquetteTrotter",
    "PlaquetteTrotterSettings",
]


class PlaquetteTrotter(Trotter):
    """Build a second-order product formula from exact plaquette evolutions.

    The builder derives the hopping and diagonal layers from a normal
    :class:`~qdk_chemistry.data.QubitOperator`.
    The declared lattice is checked against the Hamiltonian before the
    plaquette tilings are constructed.

    Note:
        This expects a Jordan-Wigner encoded, spin-blocked uniform Fermi-Hubbard
        model on a periodic square lattice whose sides are even and either both at
        least four or exactly 2x2, with spin-up modes first. Incompatible geometry,
        ordering, or hopping raises a :class:`ValueError`.

    """

    def __init__(
        self,
        order: int = 2,
        *,
        lattice_width: int = 0,
        lattice_height: int = 0,
        time: float = 0.0,
        target_accuracy: float = 0.0,
        num_divisions: int = 0,
        error_bound: str = "commutator",
        weight_threshold: float = 1e-12,
        power: int = 1,
        power_strategy: str = "repeat",
    ):
        """Initialize the builder.

        Args:
            order: Trotter-Suzuki order. Only 2 is supported.
            lattice_width: Number of lattice columns. Required at run time.
            lattice_height: Number of lattice rows. Required at run time.
            time: The evolution time. Defaults to 0.0.
            target_accuracy: Target accuracy for auto step computation. Use 0.0 to disable.
            num_divisions: Divisions per Trotter step. Max of this and the auto value is used.
            error_bound: Error bound strategy: ``"commutator"`` (default) or ``"naive"``.
            weight_threshold: Threshold for filtering small coefficients.
            power: The power to raise the unitary to. Defaults to 1.
            power_strategy: Strategy for ``U^power``: ``"rescale"`` or ``"repeat"``.

        Raises:
            ValueError: If *order* is anything other than 2.

        """
        if order != 2:
            raise ValueError(f"PlaquetteTrotter supports order 2 only, got {order}.")
        super().__init__(
            order,
            time=time,
            target_accuracy=target_accuracy,
            num_divisions=num_divisions,
            error_bound=error_bound,
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
        settings.set("weight_threshold", weight_threshold)
        settings.set("lattice_width", lattice_width)
        settings.set("lattice_height", lattice_height)
        self._settings = settings

    def name(self) -> str:
        """Return ``plaquette`` as the algorithm name."""
        return "plaquette"

    def _resolve_num_divisions(self, qubit_hamiltonian, time: float) -> int:
        """Determine the step count from Campbell's plaquette-specific error constant.

        derived for this
        exact splitting (arXiv:2012.09238v4, Eq. (20) and App. D).

        Args:
            qubit_hamiltonian: The Hamiltonian being evolved.
            time: Duration of the evolution.

        Returns:
            The number of Trotter steps, at least one.

        """
        num_divisions = self._settings.get("num_divisions")
        manual = num_divisions if num_divisions > 0 else 1

        target_accuracy = self._settings.get("target_accuracy")
        if target_accuracy <= 0.0:
            return manual

        width = self._settings.get("lattice_width")
        height = self._settings.get("lattice_height")
        if width <= 0 or height <= 0:
            raise ValueError(
                "PlaquetteTrotter needs lattice_width and lattice_height to size its "
                "error bound; set them, or set num_divisions and leave target_accuracy at 0."
            )

        hopping, _, _ = self._split_hopping(qubit_hamiltonian, width * height, self._settings.get("weight_threshold"))
        num_sites = width * height
        interaction = 0.0
        for label, coeff in qubit_hamiltonian.get_real_coefficients(tolerance=1e-12):
            positions = [index for index, axis in enumerate(reversed(label)) if axis != "I"]
            if len(positions) == 2 and all(label[::-1][position] == "Z" for position in positions):
                low, high = positions
                if high - low == num_sites:
                    interaction = abs(coeff) * 4.0
                    break
        automatic = self._plaquette_trotter_steps(
            width=width,
            height=height,
            hopping=hopping,
            interaction=interaction,
            time=time,
            target_accuracy=target_accuracy,
        )
        Logger.debug(f"PlaquetteTrotter: bound gives r={automatic}, manual is {manual}.")
        return max(manual, automatic)

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

        declared_order = getattr(qubit_hamiltonian, "fermion_mode_order", None)
        if declared_order is not None and str(declared_order) != str(FermionModeOrder.BLOCKED):
            raise ValueError(
                f"PlaquetteTrotter reads the register as spin-blocked (spin-up modes first), but "
                f"the Hamiltonian declares {declared_order!s} ordering. Re-map it with "
                f"{FermionModeOrder.BLOCKED!s} ordering before building the unitary."
            )

        hopping, diagonal, observed_bonds = self._split_hopping(qubit_hamiltonian, num_sites, atol)
        section_a, section_b = self._plaquette_sections(width, height)
        order = self._settings.get("order")
        if order != 2:
            raise ValueError(
                f"PlaquetteTrotter supports order 2 only, got {order}. Campbell's W_PLAQ is a "
                "second-order constant and would understate a first-order product's error."
            )

        tiled_bonds = {
            frozenset((cycle[index], cycle[(index + 1) % 4])) for cycle in section_a + section_b for index in range(4)
        }
        if observed_bonds != tiled_bonds:
            missing = len(tiled_bonds - observed_bonds)
            extra = len(observed_bonds - tiled_bonds)
            raise ValueError(
                f"The Hamiltonian's hopping graph does not match a periodic {width}x{height} "
                f"square lattice: {missing} lattice bond(s) absent from the operator and "
                f"{extra} operator bond(s) outside the lattice. Check the lattice dimensions, "
                "the boundary conditions, and that sites are numbered row-major."
            )

        # Campbell's Eq. (D2) (arXiv:2012.09238v4, App. D) ordering: the interaction is
        # halved at the two ends and the second hopping section runs at full time in
        # the middle.
        #
        # Batch identifiers unique across the whole step: the two
        # diagonal half-layers and each of the three section applications hoist and batch
        # their own equal-angle families from a first identifier past all already issued.
        batch = 1
        constant = [term for term in diagonal if not term.pauli_term]
        diagonal = [term for term in diagonal if term.pauli_term]
        merged = (
            [ExponentiatedPauliTerm(pauli_term={}, angle=sum(term.angle for term in constant) * time)]
            if constant
            else []
        )
        # Step 1: half the interaction.
        opening = self._diagonal_layer(diagonal, time * 0.5, first_batch=batch)
        batch = self._next_batch(opening, batch)
        # Step 2: half of hopping section A.
        hop_a_open = self._hop_layer(
            section_a,
            num_sites=num_sites,
            hopping=hopping,
            time=time * 0.5,
            first_batch=batch,
        )
        batch = self._next_batch(hop_a_open, batch)
        # Step 3: all of hopping section B.
        hop_b = self._hop_layer(
            section_b,
            num_sites=num_sites,
            hopping=hopping,
            time=time,
            first_batch=batch,
        )
        batch = self._next_batch(hop_b, batch)
        # Step 4: the remaining half of section A.
        hop_a_close = self._hop_layer(
            section_a,
            num_sites=num_sites,
            hopping=hopping,
            time=time * 0.5,
            first_batch=batch,
        )
        batch = self._next_batch(hop_a_close, batch)
        # Step 5: the remaining half of the interaction.
        closing = self._diagonal_layer(diagonal, time * 0.5, first_batch=batch)
        return merged + opening + hop_a_open + hop_b + hop_a_close + closing

    def _split_hopping(self, qubit_hamiltonian, num_sites, atol):
        """Separate the uniform hopping amplitude, the diagonal terms, and the bond graph.

        Args:
            qubit_hamiltonian: The Hamiltonian to inspect.
            num_sites: Number of lattice sites, used to fold the two spin blocks together.
            atol: Threshold below which coefficients are dropped.

        Returns:
            ``(hopping, diagonal, bonds)`` where *hopping* is the amplitude ``t``,
            *diagonal* holds every non-hopping term as an
            :class:`ExponentiatedPauliTerm` scaled to unit time, and *bonds* is the
            set of site pairs the hopping terms connect.

        Raises:
            ValueError: If no hopping is present, it is not uniform, a hopping term
                connects the two spin blocks, or the two spin sectors disagree.

        """
        magnitudes: set[float] = set()
        diagonal: list[ExponentiatedPauliTerm] = []
        # Tracked per spin sector: a bond present for one spin but not the other would
        # otherwise be hidden by folding the two blocks together, and the tiling applies
        # every plaquette to both spins.
        bonds: dict[int, set[frozenset[int]]] = {0: set(), 1: set()}
        for label, coeff in qubit_hamiltonian.get_real_coefficients(tolerance=atol):
            mapping = self._pauli_label_to_map(label)
            axes = [position for position, axis in mapping.items() if axis in "XY"]
            if len(axes) != 2:
                diagonal.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=coeff))
                continue

            # A Jordan-Wigner hopping bond contributes two Paulis of weight t/2, whose
            # X/Y endpoints are the two modes it connects.
            magnitudes.add(round(abs(coeff), 12))
            first, second = axes
            if (first < num_sites) != (second < num_sites):
                raise ValueError(
                    f"The term {label!r} hops between the spin-up and spin-down blocks. "
                    "PlaquetteTrotter tiles each spin sector separately and cannot express "
                    "spin-flip hopping."
                )
            spin = 0 if first < num_sites else 1
            bonds[spin].add(frozenset((first % num_sites, second % num_sites)))

        if not magnitudes:
            raise ValueError("The Hamiltonian carries no hopping terms; nothing to tile into plaquettes.")
        if len(magnitudes) > 1:
            raise ValueError(
                f"PlaquetteTrotter requires a uniform hopping amplitude, but found "
                f"{len(magnitudes)} distinct magnitudes: {sorted(magnitudes)}."
            )
        if bonds[0] != bonds[1]:
            raise ValueError(
                f"The two spin sectors carry different hopping graphs "
                f"({len(bonds[0])} and {len(bonds[1])} bonds). PlaquetteTrotter applies the same "
                "tiling to both spins."
            )

        hopping = 2.0 * next(iter(magnitudes))
        Logger.debug(
            f"PlaquetteTrotter: hopping t={hopping}, {len(bonds[0])} bonds per spin, {len(diagonal)} diagonal terms."
        )
        return hopping, diagonal, bonds[0]

    @staticmethod
    def _plaquette_sections(width: int, height: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Tile a periodic square lattice with two sections of vertex-disjoint four-cycles."""
        if width % 2 or height % 2:
            raise ValueError(f"Plaquette tiling requires even side lengths, got {width}x{height}.")
        if (width < 4 or height < 4) and (width, height) != (2, 2):
            raise ValueError(
                f"Plaquette tiling requires both sides to be at least four, or exactly 2x2, "
                f"got {width}x{height}. Other periodic lattices with a side below four wrap "
                "onto themselves and cannot be tiled."
            )

        section_a: list[tuple[int, ...]] = []
        section_b: list[tuple[int, ...]] = []
        shifted_sections = ((section_a, 0),) if (width, height) == (2, 2) else ((section_a, 0), (section_b, 1))
        for section, shift in shifted_sections:
            for row in range(0, height, 2):
                for col in range(0, width, 2):
                    top = (row + shift) % height
                    bottom = (row + shift + 1) % height
                    left = (col + shift) % width
                    right = (col + shift + 1) % width
                    section.append(
                        (
                            top * width + left,
                            top * width + right,
                            bottom * width + right,
                            bottom * width + left,
                        )
                    )
        return section_a, section_b

    def _hop_layer(
        self,
        section: list[tuple[int, ...]],
        *,
        num_sites: int,
        hopping: float,
        time: float,
        first_batch: int,
    ) -> list[ExponentiatedPauliTerm]:
        """Evolve every plaquette of *section* for both spin sectors.

        Args:
            section: The section's four-cycles, or empty for the 2x2 lattice.
            num_sites: Number of sites in one spin sector.
            hopping: Uniform hopping amplitude.
            time: Duration of this section evolution.
            first_batch: First batch identifier free for this layer.

        Returns:
            The section factors in hoisted order.

        """
        heads: list[ExponentiatedPauliTerm] = []
        phases: list[ExponentiatedPauliTerm] = []
        tails_per_plaquette: list[list[ExponentiatedPauliTerm]] = []
        for spin_offset in (0, num_sites):
            for cycle in section:
                shifted = tuple(site + spin_offset for site in cycle)
                head, middle, tail = self._plaquette_parts(shifted, hopping, time)
                heads += head
                phases += middle
                tails_per_plaquette.append(tail)

        tails: list[ExponentiatedPauliTerm] = []
        for tail in reversed(tails_per_plaquette):
            tails += tail
        phases = self._batch_equal_angles(phases, first_batch=first_batch)
        return heads + phases + tails

    @staticmethod
    def _plaquette_parts(
        sites: tuple[int, ...], hopping: float, time: float
    ) -> tuple[list[ExponentiatedPauliTerm], list[ExponentiatedPauliTerm], list[ExponentiatedPauliTerm]]:
        r"""Return one plaquette's evolution split into its three structural blocks.

        The evolution is the conjugation :math:`U_V\, D\, U_V^{\dagger}`. Written out in
        full that is a six-factor Givens network, two eigenvalue phases, and the adjoint
        network. Campbell's Appendix E instead absorbs the innermost butterfly into the
        phases, using

        .. math::

            G_{01}^{\dagger}\,
            e^{i\alpha n_0} e^{-i\alpha n_1}\,
            G_{01}
            = e^{i(\alpha/2) XX} e^{i(\alpha/2) YY},

        which holds because :math:`G_{01}^{\dagger}(Z_1 - Z_0)G_{01} = XX + YY`. The two
        factors that produced the phases then disappear from both the network and its
        adjoint, so the plaquette costs four fixed factors instead of six:

        * ``head`` -- four ``pi/8`` butterfly factors (``needs_control=False``).
        * ``middle`` -- the two fused arbitrary-angle ``XX``/``YY`` rotations.
        * ``tail`` -- four factors, the exact reverse-and-negate of ``head``.

        That is eight T gates and two synthesized rotations per plaquette rather than
        twelve and two, which is what takes the step from :math:`18 L^2` to Campbell's
        published :math:`12 L^2` (arXiv:2012.09238v4, App. E Eqs. (E11)--(E14)).

        Keeping the blocks separate lets :meth:`PlaquetteTrotter._decompose_trotter_step`
        hoist the phases of a whole section together so equal-angle families can be
        Hamming-weight phased; see that method for why the hoisting is exact.

        Args:
            sites: The plaquette's four modes in cycle order.
            hopping: Hopping amplitude :math:`t`.
            time: Evolution time.

        Returns:
            ``(head, middle, tail)``. ``tail`` equals ``head`` reversed with every angle
            negated, so concatenating heads in one order and tails in the exactly reversed
            order keeps the uncontrolled factors cancelling strictly last-in-first-out.

        """
        # The target is U_V D U_V^dagger with U_V = G1 G2 G3 as a matrix product, so the
        # factor applied first is G1^dagger: the network runs forwards inverted, then the
        # eigenvalue layer, then the network backwards. G3 = G_{01} is fused into the
        # eigenvalue layer below and so is absent here.
        head: list[ExponentiatedPauliTerm] = []
        # Radix-2 butterfly pairs of the four-point Fourier transform, in cycle-local
        # indices. Every butterfly of a uniform four-cycle sits at the same pi/4 angle, so
        # each factor below is a fixed pi/8 rotation costing one T gate.
        for local_i, local_j in ((0, 2), (1, 3)):
            mode_i, mode_j = sites[local_i], sites[local_j]
            low, high = min(mode_i, mode_j), max(mode_i, mode_j)
            # The Jordan-Wigner parity string is carried inside the rotation, so the modes
            # need not be adjacent and no fermionic swap network is required.
            string = dict.fromkeys(range(low + 1, high), "Z")
            half = math.pi / 8.0 if mode_i < mode_j else -math.pi / 8.0
            # exp(theta (a_i^dag a_j - a_j^dag a_i)) as its two commuting Pauli factors.
            head += [
                ExponentiatedPauliTerm(pauli_term={**string, low: "X", high: "Y"}, angle=-half, needs_control=False),
                ExponentiatedPauliTerm(pauli_term={**string, low: "Y", high: "X"}, angle=half, needs_control=False),
            ]

        # The fused eigenvalue layer. Both rotations carry the same angle and act on the
        # plaquette's first bond, which is horizontal and therefore adjacent in row-major
        # order for every cycle except the wrapping ones, where the parity string below is
        # what keeps the identity exact.
        kappa = 2.0 * hopping * time
        low, high = min(sites[0], sites[1]), max(sites[0], sites[1])
        string = dict.fromkeys(range(low + 1, high), "Z")
        middle = [
            ExponentiatedPauliTerm(pauli_term={**string, low: "X", high: "X"}, angle=-kappa / 2.0),
            ExponentiatedPauliTerm(pauli_term={**string, low: "Y", high: "Y"}, angle=-kappa / 2.0),
        ]

        # Reversing and negating the head is what makes the conjugating factors undo each
        # other strictly last-in-first-out, which the builder's tests verify.
        tail = [
            ExponentiatedPauliTerm(
                pauli_term=dict(term.pauli_term), angle=-term.angle, needs_control=term.needs_control
            )
            for term in reversed(head)
        ]

        return head, middle, tail

    @classmethod
    def _diagonal_layer(
        cls,
        diagonal: list[ExponentiatedPauliTerm],
        fraction: float,
        first_batch: int = 1,
    ) -> list[ExponentiatedPauliTerm]:
        """Rescale diagonal terms and mark equal-angle batches."""
        scaled = [
            ExponentiatedPauliTerm(pauli_term=dict(term.pauli_term), angle=term.angle * fraction) for term in diagonal
        ]
        return cls._batch_equal_angles(scaled, first_batch=first_batch)

    @staticmethod
    def _batch_equal_angles(
        terms: list[ExponentiatedPauliTerm],
        min_batch: int = MIN_USEFUL_BATCH,
        max_batch: int = 0,
        first_batch: int = 1,
    ) -> list[ExponentiatedPauliTerm]:
        """Mark disjoint diagonal terms with equal angles as rotation batches."""
        families: dict[tuple[float, int], list[ExponentiatedPauliTerm]] = {}
        loose: list[ExponentiatedPauliTerm] = []
        for term in terms:
            if not term.pauli_term:
                loose.append(term)
                continue
            key = (round(term.angle, 12), len(term.pauli_term))
            families.setdefault(key, []).append(term)

        batched: list[ExponentiatedPauliTerm] = []
        next_batch = first_batch
        for _, family in sorted(families.items()):
            groups: list[list[ExponentiatedPauliTerm]] = []
            supports: list[set[int]] = []
            for term in family:
                support = set(term.pauli_term)
                for index, taken in enumerate(supports):
                    if not (taken & support):
                        groups[index].append(term)
                        supports[index] = taken | support
                        break
                else:
                    groups.append([term])
                    supports.append(support)

            for group in groups:
                width = len(group) if max_batch <= 0 else min(max_batch, len(group))
                for start in range(0, len(group), width):
                    chunk = group[start : start + width]
                    if len(chunk) < min_batch:
                        loose.extend(chunk)
                        continue
                    batched.extend(
                        ExponentiatedPauliTerm(
                            pauli_term=term.pauli_term,
                            angle=term.angle,
                            needs_control=term.needs_control,
                            batch=next_batch,
                        )
                        for term in chunk
                    )
                    next_batch += 1
        return batched + loose

    @staticmethod
    def _next_batch(terms: list[ExponentiatedPauliTerm], current: int) -> int:
        """Return the first batch identifier free after emitting *terms*."""
        return max((term.batch for term in terms if term.batch), default=current - 1) + 1

    @classmethod
    def _plaquette_trotter_steps(
        cls,
        width: int,
        height: int,
        hopping: float,
        interaction: float,
        time: float,
        target_accuracy: float,
    ) -> int:
        r"""Return the number of plaquette Trotter steps meeting *target_accuracy*.

        ``target_accuracy`` is a ground-state ENERGY tolerance, so the step count comes
        from Campbell's Eq. (F2) (arXiv:2012.09238v4, App. F): a step of duration
        :math:`s` biases the energy by at most :math:`W s^2`. That bias does not
        accumulate over the repetitions. Section V of the same paper gives the reason --
        the repeated unitary is :math:`\exp(i H_\mathrm{eff} s)` for one fixed
        :math:`H_\mathrm{eff}` with :math:`\|H - H_\mathrm{eff}\| \le W s^2`, and
        phase estimation reads an eigenvalue of that :math:`H_\mathrm{eff}`. Kivlichan
        et al. (arXiv:1902.10673v4, Eq. (8)) reach the same bound independently.

        So :math:`W (t/r)^2 = \epsilon` gives :math:`r = t \sqrt{W / \epsilon}`.

        The per-step *unitary* error is :math:`W s^3`, which would instead give
        :math:`r = t^{3/2}\sqrt{W/\epsilon}`. That is the wrong quantity here, and
        dimensionally so: with :math:`\hbar = 1`, :math:`[W] = E^3`, so :math:`W s^3`
        is dimensionless while an energy tolerance is not. Converting the accumulated
        norm error :math:`W t^3 / r^2` to an energy divides it by :math:`t`, which
        returns :math:`W t^2 / r^2` and the same formula as above.

        Args:
            width: Lattice columns.
            height: Lattice rows.
            hopping: Uniform hopping amplitude.
            interaction: On-site interaction strength.
            time: Total evolution time.
            target_accuracy: Ground-state energy tolerance.

        Returns:
            The number of steps, at least one.

        Raises:
            ValueError: If ``target_accuracy`` is not positive.

        """
        if target_accuracy <= 0.0:
            raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")
        constant = cls._plaquette_error_constant(width, height, hopping, interaction)
        if constant <= 0.0 or time == 0.0:
            return 1
        return max(1, math.ceil(abs(time) * math.sqrt(constant / target_accuracy)))

    @classmethod
    def _plaquette_error_constant(
        cls,
        width: int,
        height: int,
        hopping: float,
        interaction: float,
        commutator_norm_per_site: float = 3.229,
        exact_norm_max_sites: int = 1600,
    ) -> float:
        r"""Return Campbell's second-order error constant :math:`W_\mathrm{PLAQ}`.

        For a step of duration :math:`s` the *unitary* error is at most
        :math:`W_\mathrm{PLAQ}s^3` and the *energy* bias at most
        :math:`W_\mathrm{PLAQ}s^2` (arXiv:2012.09238v4, Table I caption and Eq. (F2));
        the latter is what sizes the step count, since phase estimation reads an
        energy. Here

        .. math::
            W_\mathrm{PLAQ} \le W_\mathrm{SO2}
                + \tfrac{3}{24}\, \lVert [[R_p, R_g], R_g] \rVert_1 ,

        .. math::
            W_\mathrm{SO2} \le \tfrac{u \tau^2}{6} L^2 (\sqrt5 + 8)
                + \tfrac{u^2}{24} \lVert R \rVert_1 ,

        Here :math:`R_p` and :math:`R_g` are the two tilings' single-particle hopping
        matrices. Their trace norms are evaluated exactly through 1600 sites; above that
        the extensive limits stand in, :math:`16/\pi^2` per site for
        :math:`\lVert R \rVert_1` and *commutator_norm_per_site* for the double
        commutator. That default is measured from the exact values below the cutoff and is
        flat to 0.2% from :math:`L = 32` onwards.

        Args:
            width: Lattice columns.
            height: Lattice rows.
            hopping: Hopping amplitude :math:`\tau`.
            interaction: On-site interaction :math:`u`.
            commutator_norm_per_site: Per-site double-commutator limit used beyond the exact cutoff.
            exact_norm_max_sites: Largest site count whose trace norms are evaluated exactly.

        Returns:
            The constant :math:`W_\mathrm{PLAQ}`.

        References:
            Campbell (2022), arXiv:2012.09238v4: Eq. (10); Eq. (20) (Sec. III);
            and App. D Eqs. (D6)-(D10).

        """
        import numpy as np  # noqa: PLC0415

        num_sites = width * height
        if num_sites <= exact_norm_max_sites:
            sections = []
            for cycles in cls._plaquette_sections(width, height):
                matrix = np.zeros((num_sites, num_sites))
                for cycle in cycles:
                    for index in range(4):
                        site_a, site_b = cycle[index], cycle[(index + 1) % 4]
                        matrix[site_a, site_b] = matrix[site_b, site_a] = -1.0
                sections.append(matrix)
            matrix_p, matrix_g = sections

            inner = matrix_p @ matrix_g - matrix_g @ matrix_p
            outer = inner @ matrix_g - matrix_g @ inner
            hopping_norm = float(np.linalg.svd(matrix_p + matrix_g, compute_uv=False).sum()) * hopping
            commutator_norm = float(np.linalg.svd(outer, compute_uv=False).sum()) * hopping**3
        else:
            hopping_norm = 16.0 / math.pi**2 * num_sites * hopping
            commutator_norm = commutator_norm_per_site * num_sites * hopping**3

        w_so2 = (
            interaction * hopping**2 / 6.0 * num_sites * (math.sqrt(5.0) + 8.0) + interaction**2 / 24.0 * hopping_norm
        )
        return w_so2 + (3.0 / 24.0) * commutator_norm


class PlaquetteTrotterSettings(TrotterSettings):
    """Settings for the plaquette Trotter builder."""

    def __init__(self):
        """Initialize the settings, adding the lattice shape to the Trotter defaults."""
        super().__init__()
        self._set_default("lattice_width", "int", 0, "Number of lattice columns. Required.")
        self._set_default("lattice_height", "int", 0, "Number of lattice rows. Required.")
