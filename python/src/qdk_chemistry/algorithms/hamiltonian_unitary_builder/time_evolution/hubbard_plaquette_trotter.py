r"""Second-order plaquette Trotterization for the uniform Fermi-Hubbard model.

The builder reads a :class:`~qdk_chemistry.data.QubitOperator` backed by a
:class:`~qdk_chemistry.data.qubit_operator.containers.lattice.LatticeContainer`
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

import numpy as np
import scipy.sparse

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter, TrotterSettings
from qdk_chemistry.data.qubit_operator import QubitOperator
from qdk_chemistry.data.qubit_operator.containers.lattice import LatticeContainer
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    BatchedExponentiatedPauliTerm,
    ConjugatedExponentiatedPauliTerm,
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils import Logger

__all__: list[str] = [
    "HubbardPlaquetteTrotter",
    "HubbardPlaquetteTrotterSettings",
]


class HubbardPlaquetteTrotter(Trotter):
    """Build a second-order product formula from exact plaquette evolutions.

    The builder takes a :class:`~qdk_chemistry.data.QubitOperator` wrapping a
    :class:`~qdk_chemistry.data.qubit_operator.containers.lattice.LatticeContainer`, which
    carries the lattice geometry. The model parameters
    ``t``, ``U``, and ``epsilon`` are settings.

    :math:`H_I`, :math:`H_h^p`, and :math:`H_h^g` are derived from the lattice bonds and
    construct the Jordan-Wigner image.

    Note:
        This expects a periodic square lattice whose sides are even and either both at
        least four or exactly 2x2, with uniform edge weights. Incompatible geometry
        raises a :class:`ValueError`.

    """

    def __init__(
        self,
        order: int = 2,
        *,
        t: float = 1.0,
        U: float = 0.0,  # noqa: N803  (standard Hubbard symbol, as in create_hubbard_hamiltonian)
        epsilon: float = 0.0,
        max_batch: int = 0,
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
            t: Uniform hopping amplitude of the Fermi-Hubbard model.
            U: Uniform on-site interaction of the Fermi-Hubbard model.
            epsilon: Uniform on-site energy. Use ``-U/2`` for the particle-hole-shifted model.
            max_batch: Largest Hamming-weight phasing batch, or 0 for unbounded.
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
            raise ValueError(f"HubbardPlaquetteTrotter supports order 2 only, got {order}.")
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
        settings = HubbardPlaquetteTrotterSettings()
        settings.set("time", time)
        settings.set("power", power)
        settings.set("power_strategy", power_strategy)
        settings.set("order", order)
        settings.set("target_accuracy", target_accuracy)
        settings.set("num_divisions", num_divisions)
        settings.set("error_bound", error_bound)
        settings.set("weight_threshold", weight_threshold)
        settings.set("t", t)
        settings.set("U", U)
        settings.set("epsilon", epsilon)
        settings.set("max_batch", max_batch)
        self._settings = settings

    def name(self) -> str:
        """Return ``plaquette`` as the algorithm name."""
        return "plaquette"

    @staticmethod
    def _lattice_geometry(qubit_hamiltonian: QubitOperator):
        """Return the lattice and its rectangular shape from a lattice-backed operator.

        Args:
            qubit_hamiltonian: The operator to inspect.

        Returns:
            A tuple of the :class:`~qdk_chemistry.data.LatticeGraph` and its ``(width, height)``.

        Raises:
            TypeError: If the operator does not wrap a
                :class:`~qdk_chemistry.data.qubit_operator.containers.lattice.LatticeContainer`.
            ValueError: If the lattice is not a periodic square grid with both sides above one.

        """
        if not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError("HubbardPlaquetteTrotter requires a QubitOperator containing a LatticeContainer")
        container = qubit_hamiltonian.get_container()
        if not isinstance(container, LatticeContainer):
            raise TypeError(
                f"HubbardPlaquetteTrotter requires a QubitOperator containing a LatticeContainer, but the "
                f"operator wraps a {container.type!r} container. Build it with "
                f"QubitOperator(container=LatticeContainer(LatticeGraph.square(width, height)))."
            )
        dims = tuple(int(d) for d in container.lattice.dims)
        if len(dims) != 2:
            raise ValueError(
                f"HubbardPlaquetteTrotter tiles a two-dimensional lattice, but the lattice reports "
                f"{list(dims) or 'no'} generating extents. Build it from "
                "LatticeGraph.square(width, height)."
            )
        width, height = dims
        return container.lattice, width, height

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> UnitaryRepresentation:
        r"""Build Campbell's segmented plaquette product formula.

        The pipeline runs in four stages:

        1. Read the lattice and its extents from the container, and derive the on-site
           layer :math:`H_I` from the ``U`` and ``epsilon`` settings.
        2. Read the uniform hopping from ``t``, tile the lattice into the pink and gold
           plaquette sections, and check that those tilings reproduce its bonds.
        3. Size the Trotter step count, reusing the hopping amplitude from stage 2.
        4. Assemble the merged formula. With :math:`D=e^{-isH_I}`, :math:`P=e^{-isH_h^p}`,
           and :math:`G=e^{-isH_h^g}`, repeated symmetric steps
           :math:`P^{1/2} D^{1/2} G D^{1/2} P^{1/2}` merge across repetitions into
           :math:`P^{-1/2}(D^{1/2} G D^{1/2} P)^r P^{1/2}` (arXiv:2609.05316
           Eqs. (16a)--(16b)). Merging the hopping layer rather than the interaction is
           what saves: the repeated body then carries two hopping layers instead of
           three, and hopping dominates the layer cost.

        Args:
            qubit_hamiltonian: Qubit operator wrapping a ``LatticeContainer``.

        Returns:
            UnitaryRepresentation: The segmented plaquette product formula.

        Raises:
            NotImplementedError: If the configured Trotter order is not 2.

        """
        order = self._settings.get("order")
        if order != 2:
            raise NotImplementedError(
                f"HubbardPlaquetteTrotter supports order 2 only, got {order}. Campbell's W_PLAQ is a "
                "second-order constant and would understate a first-order product's error."
            )
        atol = self._settings.get("weight_threshold")
        max_batch = self._settings.get("max_batch")

        # 1. Geometry, and the on-site layer it implies.
        lattice, width, height = self._lattice_geometry(qubit_hamiltonian)
        num_sites = width * height
        diagonal, identity_angle = self._diagonal_terms(num_sites, atol)

        # 2. Hopping amplitude and the two vertex-disjoint tilings covering every bond.
        hopping, bonds = self._uniform_hopping(lattice, atol)
        pink, gold = self._plaquette_sections(width, height)
        self._check_tiling(bonds, pink, gold, width, height)

        # 3. Step count, reusing the hopping amplitude resolved above.
        time, power_repetitions = self._resolve_power()
        num_divisions = self._step_count(hopping, (pink, gold), width, height, time)
        delta = time / num_divisions

        # 4. Repeated body D^(1/2) G D^(1/2) P. The interaction half-layers are
        # equal-angle families, so they are phased through a Hamming weight register;
        # see max_batch to bound its width.
        body: list[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm | ConjugatedExponentiatedPauliTerm] = []
        body.extend(self._diagonal_layer(diagonal, delta * 0.5, max_batch=max_batch))
        gold_layer = self._hop_layer(gold, num_sites=num_sites, hopping=hopping, time=delta)
        if gold_layer is not None:
            body.append(gold_layer)
        body.extend(self._diagonal_layer(diagonal, delta * 0.5, max_batch=max_batch))
        if identity_angle:
            body.append(ExponentiatedPauliTerm(pauli_term={}, angle=identity_angle * delta))
        pink_layer = self._hop_layer(pink, num_sites=num_sites, hopping=hopping, time=delta)
        if pink_layer is not None:
            body.append(pink_layer)

        # The one-time P^(1/2) boundary the merge leaves outside the repetitions.
        opening = self._hop_layer(pink, num_sites=num_sites, hopping=hopping, time=delta * 0.5)

        return UnitaryRepresentation(
            container=PauliProductFormulaContainer(
                step_terms=body,
                step_reps=num_divisions * power_repetitions,
                num_qubits=2 * num_sites,
                scale=time,
                conjugating_terms=[opening] if opening is not None else [],
            )
        )

    def _diagonal_terms(self, num_sites: int, atol: float) -> tuple[list[ExponentiatedPauliTerm], float]:
        r"""Return the on-site layer the model settings imply, scaled to unit time.

        The Jordan-Wigner image is constructed analytically rather than by mapping an
        operator. With spin-blocked modes (spin-up ``0..n-1``, spin-down ``n..2n-1``) and
        :math:`n_p = (I - Z_p)/2`, site :math:`i` with on-site energy :math:`\epsilon` and
        interaction :math:`U` contributes

        .. math::
            \epsilon (n_{i\uparrow} + n_{i\downarrow}) + U n_{i\uparrow} n_{i\downarrow}
            = \Bigl(\epsilon + \tfrac{U}{4}\Bigr) I
            - \Bigl(\tfrac{\epsilon}{2} + \tfrac{U}{4}\Bigr) (Z_i + Z_{i+n})
            + \tfrac{U}{4} Z_i Z_{i+n},

        so the particle-hole-shifted choice :math:`\epsilon = -U/2` cancels the
        single-qubit terms and leaves one equal-angle :math:`Z_i Z_{i+n}` family.

        Args:
            num_sites: Number of lattice sites.
            atol: Threshold below which coefficients are dropped.

        Returns:
            The non-identity on-site terms, and the scalar phase angle per unit time.

        """
        interaction = float(self._settings.get("U"))
        epsilon = float(self._settings.get("epsilon"))

        single_z = -(0.5 * epsilon + 0.25 * interaction)
        pair_z = 0.25 * interaction
        identity_angle = (epsilon + 0.25 * interaction) * num_sites

        terms: list[ExponentiatedPauliTerm] = []
        for site in range(num_sites):
            if abs(single_z) > atol:
                terms.append(ExponentiatedPauliTerm(pauli_term={site: "Z"}, angle=single_z))
                terms.append(ExponentiatedPauliTerm(pauli_term={site + num_sites: "Z"}, angle=single_z))
            if abs(pair_z) > atol:
                terms.append(ExponentiatedPauliTerm(pauli_term={site: "Z", site + num_sites: "Z"}, angle=pair_z))

        Logger.debug(f"HubbardPlaquetteTrotter: U={interaction}, epsilon={epsilon}, {len(terms)} on-site terms.")
        return terms, identity_angle if abs(identity_angle) > atol else 0.0

    def _uniform_hopping(self, lattice, atol: float) -> tuple[float, set[frozenset[int]]]:
        """Return the uniform hopping amplitude and the lattice's bonds.

        Each lattice edge carries the ``t`` setting scaled by its edge weight, so the
        weights must agree for the plaquette evolutions to be exact.

        Args:
            lattice: The lattice graph supplying the bonds.
            atol: Edge weights with magnitude at or below this are treated as absent.

        Returns:
            The hopping amplitude and the set of site pairs it connects.

        Raises:
            ValueError: If the lattice carries no bonds or its edge weights differ.

        """
        upper = scipy.sparse.triu(lattice.sparse_adjacency_matrix(), k=1, format="coo")
        bonds: set[frozenset[int]] = set()
        weights: set[float] = set()
        for row, col, value in zip(upper.row, upper.col, upper.data, strict=True):
            if abs(value) <= atol:
                continue
            bonds.add(frozenset((int(row), int(col))))
            weights.add(round(float(value), 12))

        if not bonds:
            raise ValueError("The lattice carries no bonds; nothing to tile into plaquettes.")
        if len(weights) > 1:
            raise ValueError(
                f"HubbardPlaquetteTrotter requires a uniform hopping amplitude, but the lattice carries "
                f"{len(weights)} distinct edge weights: {sorted(weights)}."
            )

        hopping = float(self._settings.get("t")) * next(iter(weights))
        Logger.debug(f"HubbardPlaquetteTrotter: hopping t={hopping} over {len(bonds)} bonds per spin.")
        return hopping, bonds

    @staticmethod
    def _check_tiling(
        bonds: set[frozenset[int]],
        pink: list[tuple[int, ...]],
        gold: list[tuple[int, ...]],
        width: int,
        height: int,
    ) -> None:
        """Check that the two tilings cover exactly the lattice's bonds.

        Args:
            bonds: Bonds observed on the lattice.
            pink: The pink tiling's four-cycles.
            gold: The gold tiling's four-cycles.
            width: Number of lattice columns.
            height: Number of lattice rows.

        Raises:
            ValueError: If the tilings and the lattice disagree on any bond.

        """
        tiled = {frozenset((cycle[i], cycle[(i + 1) % 4])) for cycle in pink + gold for i in range(4)}
        if bonds != tiled:
            raise ValueError(
                f"The lattice's bond graph does not match a periodic {width}x{height} "
                f"square lattice: {len(tiled - bonds)} lattice bond(s) absent from the graph and "
                f"{len(bonds - tiled)} graph bond(s) outside the tiling. Check the lattice "
                "dimensions, the boundary conditions, and that sites are numbered row-major."
            )

    def _resolve_num_divisions(self, qubit_hamiltonian: QubitOperator, time: float) -> int:
        """Return the step count the builder would use for this operator and duration.

        Args:
            qubit_hamiltonian: The lattice-backed operator being evolved.
            time: Duration of the evolution.

        Returns:
            The number of Trotter steps, at least one.

        """
        lattice, width, height = self._lattice_geometry(qubit_hamiltonian)
        hopping, _ = self._uniform_hopping(lattice, self._settings.get("weight_threshold"))
        sections = self._plaquette_sections(width, height)
        return self._step_count(hopping, sections, width, height, time)

    def _step_count(
        self,
        hopping: float,
        sections: tuple[list[tuple[int, ...]], list[tuple[int, ...]]],
        width: int,
        height: int,
        time: float,
    ) -> int:
        """Determine the step count from Campbell's plaquette-specific error constant.

        Derived for this exact splitting (Eq. (20) and App. D).

        Args:
            hopping: Uniform hopping amplitude.
            sections: The pink and gold tilings, reused from the caller.
            width: Number of lattice columns.
            height: Number of lattice rows.
            time: Duration of the evolution.

        Returns:
            The number of Trotter steps, at least one.

        """
        num_divisions = self._settings.get("num_divisions")
        manual = num_divisions if num_divisions > 0 else 1

        target_accuracy = self._settings.get("target_accuracy")
        if target_accuracy <= 0.0:
            return manual

        num_sites = width * height
        hopping = abs(hopping)
        interaction = abs(self._settings.get("U"))

        # R_p and R_g are the one-spin, unit-hopping matrices for the pink and gold tilings.
        # Evaluate their trace norms exactly through 1600 sites; beyond that use the
        # thermodynamic limits 16/pi^2 and 3.229 per site, respectively.
        if num_sites <= 1600:
            section_matrices: list[np.ndarray] = []
            for cycles in sections:
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

        # Campbell derives W_PLAQ for his own factor order, which puts the interaction
        # outside; this builder puts a hopping tiling outside instead (arXiv:2609.05316
        # Eq. (16a)) to merge the expensive layer across repetitions. Reusing his
        # constant is therefore an approximation. Measured on the periodic 2x2 lattice
        # over U/t in [1, 16], the swap changes the empirical constant by a factor of
        # 0.50x to 1.01x, i.e. it is more accurate everywhere except the strongest
        # coupling, where it is 0.6% worse; at the U/t = 8 benchmark point it is 0.965x.
        # Larger lattices, where the two tilings no longer commute, were not measured.

        # 4. Eq. (F2) gives epsilon_TS <= W s^2. Requiring this upper bound to
        # meet target_accuracy gives s <= sqrt(target_accuracy / W).
        if w_plaquette <= 0.0 or time == 0.0:
            automatic = 1
        else:
            max_step_size = math.sqrt(target_accuracy / w_plaquette)
            automatic = max(1, math.ceil(abs(time) / max_step_size))
        Logger.debug(f"HubbardPlaquetteTrotter: bound gives r={automatic}, manual is {manual}.")
        return max(manual, automatic)

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
            apply_terms=self._batch_equal_angles(phases, max_batch=self._settings.get("max_batch")),
        )

    @classmethod
    def _diagonal_layer(
        cls,
        diagonal: list[ExponentiatedPauliTerm],
        fraction: float,
        max_batch: int = 0,
    ) -> list[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm]:
        """Rescale mapped :math:`H_I` terms and group equal-angle rotations."""
        scaled = [
            ExponentiatedPauliTerm(pauli_term=dict(term.pauli_term), angle=term.angle * fraction) for term in diagonal
        ]
        return cls._batch_equal_angles(scaled, max_batch=max_batch)

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


class HubbardPlaquetteTrotterSettings(TrotterSettings):
    """Settings for the plaquette Trotter builder."""

    def __init__(self):
        """Initialize the settings, adding the model parameters to the Trotter defaults."""
        super().__init__()
        self._set_default("t", "float", 1.0, "Uniform hopping amplitude of the Fermi-Hubbard model.")
        self._set_default("U", "float", 0.0, "Uniform on-site interaction of the Fermi-Hubbard model.")
        self._set_default(
            "epsilon",
            "float",
            0.0,
            "Uniform on-site energy; use -U/2 for the particle-hole-shifted model.",
        )
        self._set_default(
            "max_batch",
            "int",
            0,
            "Largest Hamming-weight phasing batch, or 0 for unbounded, or 1 to phase every "
            "term individually. A batch of m equal-angle terms on disjoint qubits costs "
            "ceil(log2(m+1)) rotations and about m ancillas, so capping it trades rotations "
            "for width. Campbell's Table II budgets L^2/2 ancillas (arXiv:2012.09238v4); pass "
            "that to reproduce his qubit counts rather than the cheaper rotation count.",
        )
