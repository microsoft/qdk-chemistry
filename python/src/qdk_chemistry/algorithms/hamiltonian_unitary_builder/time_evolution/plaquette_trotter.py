r"""Second-order plaquette Trotterization for the uniform Fermi-Hubbard model.

The builder reads a Jordan-Wigner encoded :class:`~qdk_chemistry.data.QubitOperator`
and follows Campbell's decomposition :math:`H=H_I+H_h^p+H_h^g`, where
:math:`H_I` is the particle-hole-shifted onsite interaction and :math:`p` and
:math:`g` denote the pink and gold hopping tilings. One step applies

.. math::
    e^{-isH_I/2} e^{-isH_h^p/2} e^{-isH_h^g} e^{-isH_h^p/2} e^{-isH_I/2}.

Each plaquette hopping evolution is exact. The builder diagonalizes its
single-particle matrix, :math:`T = V \Lambda V^\dagger`, and applies

.. math::
    e^{-isH_\square} = U_V e^{-is\sum_m \lambda_m n_m} U_V^\dagger.

Thus the circuit switches to the plaquette momentum basis, applies two nonzero
eigenvalue phases, and switches back. Trotter error comes only from splitting
:math:`H_I`, :math:`H_h^p`, and :math:`H_h^g`. The lattice dimensions are supplied as settings; the
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

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter, TrotterSettings
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    BatchedExponentiatedPauliTerm,
    ConjugatedExponentiatedPauliTerm,
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
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

    The builder derives :math:`H_I`, :math:`H_h^p`, and :math:`H_h^g` from a normal
    :class:`~qdk_chemistry.data.QubitOperator`.
    The declared lattice is checked against the Hamiltonian before the
    pink and gold plaquette tilings are constructed.

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

    def _trotter(
        self, qubit_hamiltonian: QubitOperator, time: float, power_repetitions: int = 1
    ) -> UnitaryRepresentation:
        r"""Build Campbell's segmented plaquette product formula.

        Define :math:`D=e^{-isH_I}`, :math:`P=e^{-isH_h^p}`, and
        :math:`G=e^{-isH_h^g}`. Campbell's Eqs. (E1)--(E2) rewrite repeated symmetric steps
        :math:`D^{1/2} P^{1/2} G P^{1/2} D^{1/2}` as
        :math:`D^{1/2}(P^{1/2} G P^{1/2} D)^rD^{-1/2}` after ``r`` repetitions.
        :meth:`_decompose_trotter_step` emits the four-factor repeated body; this
        method adds the one-time boundary factors, which are left bare in a
        controlled circuit.

        Args:
            qubit_hamiltonian: The Hamiltonian being evolved.
            time: Total evolution time before applying the power strategy.
            power_repetitions: Number of times to repeat the full evolution.

        Returns:
            The segmented product-formula representation.

        """
        num_divisions = self._resolve_num_divisions(qubit_hamiltonian, time)
        delta = time / num_divisions
        step_terms = self._decompose_trotter_step(
            qubit_hamiltonian,
            time=delta,
            atol=self._settings.get("weight_threshold"),
        )
        num_sites = self._settings.get("lattice_width") * self._settings.get("lattice_height")
        _, diagonal, _ = self._split_hopping(qubit_hamiltonian, num_sites, self._settings.get("weight_threshold"))
        diagonal = [term for term in diagonal if term.pauli_term]
        boundary = self._diagonal_layer(diagonal, delta * 0.5)

        return UnitaryRepresentation(
            container=PauliProductFormulaContainer(
                step_terms=step_terms,
                step_reps=num_divisions * power_repetitions,
                num_qubits=qubit_hamiltonian.num_qubits,
                scale=time,
                conjugating_terms=boundary,
            )
        )

    def _resolve_num_divisions(self, qubit_hamiltonian, time: float) -> int:
        """Determine the step count from Campbell's plaquette-specific error constant.

        Derived for this exact splitting (Eq. (20) and App. D).

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
        hopping = abs(hopping)
        num_sites = width * height

        # Reverse-engineers the Hubbard interaction magnitude (|U|) from the Jordan-Wigner-mapped Hamiltonian.
        interaction = 0.0
        for label, coeff in qubit_hamiltonian.get_real_coefficients(tolerance=1e-12):
            positions = [index for index, axis in enumerate(reversed(label)) if axis != "I"]
            if len(positions) == 2 and all(label[::-1][position] == "Z" for position in positions):
                low, high = positions
                if high - low == num_sites:
                    interaction = abs(coeff) * 4.0
                    break

        # R_p and R_g are the one-spin, unit-hopping matrices for the pink and gold tilings.
        # Evaluate their trace norms exactly through 1600 sites; beyond that use the
        # thermodynamic limits 16/pi^2 and 3.229 per site, respectively.
        if num_sites <= 1600:
            section_matrices: list[np.ndarray] = []
            for cycles in self._plaquette_sections(width, height):
                matrix = np.zeros((num_sites, num_sites))
                for cycle in cycles:
                    for index in range(4):
                        site_a, site_b = cycle[index], cycle[(index + 1) % 4]
                        matrix[site_a, site_b] = matrix[site_b, site_a] = -1.0
                section_matrices.append(matrix)
            matrix_p, matrix_g = section_matrices

            inner = matrix_p @ matrix_g - matrix_g @ matrix_p
            outer = inner @ matrix_g - matrix_g @ inner
            hopping_norm = float(np.linalg.svd(matrix_p + matrix_g, compute_uv=False).sum()) * hopping
            commutator_norm = float(np.linalg.svd(outer, compute_uv=False).sum()) * hopping**3
        else:
            hopping_norm = 16.0 / math.pi**2 * num_sites * hopping
            commutator_norm = 3.229 * num_sites * hopping**3

        # 1. W_SO2 from Eq. (C3), in the dimensionful R convention of Eq. (10).
        w_so2 = (
            interaction * hopping**2 / 6.0 * num_sites * (math.sqrt(5.0) + 8.0) + interaction**2 / 24.0 * hopping_norm
        )
        # 2. The extra plaquette-splitting contribution from Eq. (D10).
        w_extra2 = 3.0 / 24.0 * commutator_norm
        # 3. The complete plaquette error constant from Eq. (D6).
        w_plaquette = w_so2 + w_extra2

        # 4. Eq. (F2) gives epsilon_TS <= W s^2. Requiring this upper bound to
        # meet target_accuracy gives s <= sqrt(target_accuracy / W).
        if w_plaquette <= 0.0 or time == 0.0:
            automatic = 1
        else:
            max_step_size = math.sqrt(target_accuracy / w_plaquette)
            automatic = max(1, math.ceil(abs(time) / max_step_size))
        Logger.debug(f"PlaquetteTrotter: bound gives r={automatic}, manual is {manual}.")
        return max(manual, automatic)

    def _decompose_trotter_step(
        self,
        qubit_hamiltonian: QubitOperator,
        time: float,
        atol: float = 1e-12,
    ) -> list[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm | ConjugatedExponentiatedPauliTerm]:
        r"""Return the repeated bulk body of Campbell's plaquette formula.

        With :math:`D=e^{-isH_I}`, :math:`P=e^{-isH_h^p}`, and
        :math:`G=e^{-isH_h^g}`, the returned terms implement
        :math:`P^{1/2} G P^{1/2} D`: two pink half-layers, one gold full layer,
        and one full interaction layer. :meth:`_trotter` supplies the one-time
        :math:`D^{1/2}` and :math:`D^{-1/2}` boundary layers.

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
        constant = [term for term in diagonal if not term.pauli_term]
        diagonal = [term for term in diagonal if term.pauli_term]
        identity_phase = (
            [ExponentiatedPauliTerm(pauli_term={}, angle=sum(term.angle for term in constant) * time)]
            if constant
            else []
        )
        # For each tile, we diagonalize R_plaq and realize the tile with 2 Z rotations and 4 F gates.
        # Repeated body 1: P^(1/2), the first pink hopping half-layer.
        hop_a_open = self._hop_layer(
            section_a,
            num_sites=num_sites,
            hopping=hopping,
            time=time * 0.5,
        )
        # Repeated body 2: G, the gold hopping layer.
        hop_b = self._hop_layer(
            section_b,
            num_sites=num_sites,
            hopping=hopping,
            time=time,
        )
        # Repeated body 3: P^(1/2), the second pink hopping half-layer.
        hop_a_close = self._hop_layer(
            section_a,
            num_sites=num_sites,
            hopping=hopping,
            time=time * 0.5,
        )
        # Repeated body 4: D, one full H_I layer. After the particle-hole shift,
        # each onsite interaction contributes one rotation.
        interaction = self._diagonal_layer(diagonal, time)
        hopping_layers = [layer for layer in (hop_a_open, hop_b, hop_a_close) if layer is not None]
        return hopping_layers + identity_phase + interaction

    def _split_hopping(self, qubit_hamiltonian, num_sites, atol):
        """Separate uniform hopping, the :math:`H_I` terms, and the bond graph.

        Args:
            qubit_hamiltonian: The Hamiltonian to inspect.
            num_sites: Number of lattice sites, used to fold the two spin blocks together.
            atol: Threshold below which coefficients are dropped.

        Returns:
            ``(hopping, diagonal, bonds)`` where *hopping* is the amplitude ``t``,
            *diagonal* holds the mapped :math:`H_I` terms and scalar offsets as
            :class:`ExponentiatedPauliTerm` scaled to unit time, and *bonds* is the
            set of site pairs the hopping terms connect.

        Raises:
            ValueError: If no hopping is present, it is not uniform, a hopping term
                connects the two spin blocks, or the two spin sectors disagree.

        """
        hopping_terms: dict[int, dict[frozenset[int], dict[str, float]]] = {0: {}, 1: {}}
        diagonal: list[ExponentiatedPauliTerm] = []
        for label, coeff in qubit_hamiltonian.get_real_coefficients(tolerance=atol):
            mapping = self._pauli_label_to_map(label)
            axes = [position for position, axis in mapping.items() if axis in "XY"]
            if not axes:
                diagonal.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=coeff))
                continue
            if len(axes) != 2:
                raise ValueError(
                    f"The term {label!r} is not a Jordan-Wigner hopping string: expected exactly two X/Y endpoints."
                )

            first, second = sorted(axes)
            if (first < num_sites) != (second < num_sites):
                raise ValueError(
                    f"The term {label!r} hops between the spin-up and spin-down blocks. "
                    "PlaquetteTrotter tiles each spin sector separately and cannot express "
                    "spin-flip hopping."
                )
            endpoint_axis = mapping[first]
            expected_support = set(range(first, second + 1))
            if (
                endpoint_axis not in "XY"
                or mapping[second] != endpoint_axis
                or set(mapping) != expected_support
                or any(mapping[position] != "Z" for position in range(first + 1, second))
            ):
                raise ValueError(f"The term {label!r} is not a canonical Jordan-Wigner XX/YY hopping string.")
            spin = 0 if first < num_sites else 1
            bond = frozenset((first % num_sites, second % num_sites))
            components = hopping_terms[spin].setdefault(bond, {})
            components[endpoint_axis] = components.get(endpoint_axis, 0.0) + coeff

        if not hopping_terms[0] and not hopping_terms[1]:
            raise ValueError("The Hamiltonian carries no hopping terms; nothing to tile into plaquettes.")
        if hopping_terms[0].keys() != hopping_terms[1].keys():
            raise ValueError(
                f"The two spin sectors carry different hopping graphs "
                f"({len(hopping_terms[0])} and {len(hopping_terms[1])} bonds). PlaquetteTrotter applies the same "
                "tiling to both spins."
            )

        coefficients: set[float] = set()
        for spin_terms in hopping_terms.values():
            for components in spin_terms.values():
                if components.keys() != {"X", "Y"} or not math.isclose(
                    components["X"], components["Y"], rel_tol=0.0, abs_tol=atol
                ):
                    raise ValueError(
                        "PlaquetteTrotter requires uniform hopping with matching XX and YY coefficients on every bond."
                    )
                coefficients.add(round(components["X"], 12))
        if len(coefficients) > 1:
            raise ValueError(
                f"PlaquetteTrotter requires a uniform hopping amplitude, but found "
                f"{len(coefficients)} distinct signed coefficients: {sorted(coefficients)}."
            )

        hopping = -2.0 * next(iter(coefficients))
        Logger.debug(
            f"PlaquetteTrotter: hopping t={hopping}, {len(hopping_terms[0])} bonds per spin, "
            f"{len(diagonal)} diagonal terms."
        )
        return hopping, diagonal, set(hopping_terms[0])

    @staticmethod
    def _plaquette_sections(width: int, height: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Tile a periodic square lattice with Campbell's pink and gold four-cycles."""
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
    ) -> ConjugatedExponentiatedPauliTerm | None:
        r"""Evolve every plaquette of *section* for both spin sectors.

        Each plaquette evolution is the conjugation :math:`U_V D U_V^\dagger`.
        Campbell's Appendix E absorbs the innermost Fourier-transform butterfly
        into the two eigenvalue phases, leaving four fixed factors on each side
        and two synthesized rotations in the middle (Eqs. (E11)--(E14)). The
        structural blocks are hoisted across each vertex-disjoint section so its
        equal-angle phase families can use Hamming-weight phasing.

        Args:
            section: One pink or gold tiling's four-cycles, or empty for the 2x2 lattice.
            num_sites: Number of sites in one spin sector.
            hopping: Uniform hopping amplitude.
            time: Duration of this section evolution.

        Returns:
            The hopping tiling as a structured basis-change conjugation, or ``None`` when empty.

        """
        heads: list[ExponentiatedPauliTerm] = []
        phases: list[ExponentiatedPauliTerm] = []
        for spin_offset in (0, num_sites):
            for cycle in section:
                sites = tuple(site + spin_offset for site in cycle)
                head: list[ExponentiatedPauliTerm] = []

                # These are the two remaining radix-2 butterflies after Campbell's
                # innermost one is fused into the phase layer. Their Jordan-Wigner
                # strings allow nonadjacent modes without a fermionic swap network.
                for local_i, local_j in ((0, 2), (1, 3)):
                    mode_i, mode_j = sites[local_i], sites[local_j]
                    low, high = min(mode_i, mode_j), max(mode_i, mode_j)
                    string = dict.fromkeys(range(low + 1, high), "Z")
                    half = math.pi / 8.0 if mode_i < mode_j else -math.pi / 8.0
                    head += [
                        ExponentiatedPauliTerm(pauli_term={**string, low: "X", high: "Y"}, angle=-half),
                        ExponentiatedPauliTerm(pauli_term={**string, low: "Y", high: "X"}, angle=half),
                    ]

                # G_01^dagger exp(i alpha n_0) exp(-i alpha n_1) G_01 becomes
                # the two equal-angle XX/YY rotations on the plaquette's first bond.
                kappa = 2.0 * hopping * time
                low, high = min(sites[0], sites[1]), max(sites[0], sites[1])
                string = dict.fromkeys(range(low + 1, high), "Z")
                middle = [
                    ExponentiatedPauliTerm(pauli_term={**string, low: "X", high: "X"}, angle=-kappa / 2.0),
                    ExponentiatedPauliTerm(pauli_term={**string, low: "Y", high: "Y"}, angle=-kappa / 2.0),
                ]
                heads += head
                phases += middle
        if not heads:
            return None
        return ConjugatedExponentiatedPauliTerm(
            within_terms=heads,
            apply_terms=self._batch_equal_angles(phases),
        )

    @classmethod
    def _diagonal_layer(
        cls,
        diagonal: list[ExponentiatedPauliTerm],
        fraction: float,
    ) -> list[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm]:
        """Rescale mapped :math:`H_I` terms and group equal-angle rotations."""
        scaled = [
            ExponentiatedPauliTerm(pauli_term=dict(term.pauli_term), angle=term.angle * fraction) for term in diagonal
        ]
        return cls._batch_equal_angles(scaled)

    @staticmethod
    def _batch_equal_angles(
        terms: list[ExponentiatedPauliTerm],
        max_batch: int = 0,
    ) -> list[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm]:
        """Group disjoint equal-angle terms for Hamming-weight phasing."""
        families: dict[tuple[float, int], list[ExponentiatedPauliTerm]] = {}
        loose: list[ExponentiatedPauliTerm] = []
        for term in terms:
            if not term.pauli_term:
                loose.append(term)
                continue
            key = (round(term.angle, 12), len(term.pauli_term))
            families.setdefault(key, []).append(term)

        batched: list[BatchedExponentiatedPauliTerm] = []
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
                    if len(chunk) < 2:
                        loose.extend(chunk)
                        continue
                    batched.append(
                        BatchedExponentiatedPauliTerm(
                            pauli_terms=[term.pauli_term for term in chunk],
                            angle=chunk[0].angle,
                        )
                    )
        return batched + loose


class PlaquetteTrotterSettings(TrotterSettings):
    """Settings for the plaquette Trotter builder."""

    def __init__(self):
        """Initialize the settings, adding the lattice shape to the Trotter defaults."""
        super().__init__()
        self._set_default("lattice_width", "int", 0, "Number of lattice columns. Required.")
        self._set_default("lattice_height", "int", 0, "Number of lattice rows. Required.")
