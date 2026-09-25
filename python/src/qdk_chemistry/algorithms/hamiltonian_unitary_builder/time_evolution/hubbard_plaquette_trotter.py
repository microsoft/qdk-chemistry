"""Second-order plaquette Trotterization for the uniform Fermi-Hubbard model."""

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
from qdk_chemistry.data.unitary_representation.containers.hubbard_plaquette import HubbardPlaquetteContainer
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import get_qsharp_context

__all__: list[str] = [
    "HubbardPlaquetteTrotter",
    "HubbardPlaquetteTrotterSettings",
]


class HubbardPlaquetteTrotter(Trotter):
    """Build a second-order product formula from exact plaquette evolutions.

    The builder takes a :class:`~qdk_chemistry.data.QubitOperator` wrapping a
    :class:`~qdk_chemistry.data.qubit_operator.containers.lattice.LatticeContainer`, which
    carries the lattice geometry. The model parameters ``t``, ``u``, and ``epsilon`` are settings.

    The plaquette decomposition, its error constant, and the exact four-mode plaquette
    evolution are Campbell's :cite:`Campbell2022`. The factor ordering and the step-count
    rule follow the later compilation of the same algorithm in :cite:`Apel2026`.

    Note:
        This expects a periodic square lattice whose sides are even with uniform edge weights.
        Incompatible geometry raises a :class:`ValueError`.

    """

    def __init__(
        self,
        order: int = 2,
        *,
        t: float = 1.0,
        u: float = 0.0,
        epsilon: float = 0.0,
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
            u: Uniform on-site interaction of the Fermi-Hubbard model.
            epsilon: Uniform on-site energy. Use ``-u/2`` for the particle-hole-shifted model.
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
        settings.set("u", u)
        settings.set("epsilon", epsilon)
        self._settings = settings

    def name(self) -> str:
        """Return ``hubbard_plaquette`` as the algorithm name."""
        return "hubbard_plaquette"

    def _lattice_geometry(self, qubit_hamiltonian: QubitOperator):
        """Return the lattice, its shape, its edge weight, and its plaquette tilings.

        The bond graph is validated here, in the single pass that also reads the edge
        weights: the bonds must match the plaquette tiling the circuit applies, and their
        weights must be uniform, since Campbell's formula is derived for a single hopping
        amplitude on every bond.

        The returned weight is the value the lattice carries on each of those bonds, the
        ``t`` a :meth:`~qdk_chemistry.data.LatticeGraph.square` lattice was built with
        (1.0 by default). It is a per-bond multiplier on the model's ``t`` setting, not
        an amplitude in itself; the caller forms ``t * weight`` to get the hopping.

        Args:
            qubit_hamiltonian: The operator to inspect.

        The tilings are resolved from Q# to run the check, so they are handed back rather
        than evaluated a second time by the caller that needs them for the error bound.

        Returns:
            A tuple of the lattice, its ``(width, height)``, its edge weight, and its pink and gold tilings.

        Raises:
            TypeError: If the operator does not wrap a
                :class:`~qdk_chemistry.data.qubit_operator.containers.lattice.LatticeContainer`.
            ValueError: If the lattice is not a periodic square grid tiled by plaquettes with uniform edge weights.

        """
        if not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError("HubbardPlaquetteTrotter requires a QubitOperator containing a LatticeContainer")

        container = qubit_hamiltonian.get_container()
        if not isinstance(container, LatticeContainer):
            raise TypeError(
                f"HubbardPlaquetteTrotter requires a QubitOperator containing a LatticeContainer, but the "
                f"operator wraps a {container.type!r} container."
            )
        dims = tuple(int(d) for d in container.lattice.dims)
        if len(dims) != 2:
            raise ValueError(
                f"HubbardPlaquetteTrotter tiles a two-dimensional lattice, but the lattice reports "
                f"{list(dims) or 'no'}."
            )
        width, height = dims

        lattice = container.lattice
        atol = self._settings.get("weight_threshold")
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

        sections = self._plaquette_sections(width, height)
        pink, gold = sections
        tiled = {frozenset((cycle[i], cycle[(i + 1) % 4])) for cycle in pink + gold for i in range(4)}
        if bonds != tiled:
            raise ValueError(
                f"The lattice's bond graph does not match a periodic {width}x{height} "
                f"square lattice: {len(tiled - bonds)} lattice bond(s) absent from the graph and "
                f"{len(bonds - tiled)} graph bond(s) outside the tiling. Check the lattice "
                "dimensions, the boundary conditions, and that sites are numbered row-major."
            )

        weight = next(iter(weights))
        Logger.debug(f"HubbardPlaquetteTrotter: edge weight {weight} over {len(bonds)} bonds per spin.")
        return lattice, width, height, weight, sections

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> UnitaryRepresentation:
        r"""Build Campbell's segmented plaquette product formula.

        The builder reads a :class:`~qdk_chemistry.data.QubitOperator` backed by a
        :class:`~qdk_chemistry.data.qubit_operator.containers.lattice.LatticeContainer`
        and follows Campbell's decomposition :math:`H=H_I+H_h^p+H_h^g`, where
        :math:`H_I` is the particle-hole-shifted onsite interaction and :math:`p` and
        :math:`g` denote the pink and gold hopping tilings. One step applies

        .. math::
            e^{-isH_h^p/2} e^{-isH_I/2} e^{-isH_h^g} e^{-isH_I/2} e^{-isH_h^p/2},

        so adjacent steps merge their outer half-layers into one full-angle
        :math:`e^{-isH_h^p}`, and the whole evolution collapses to a single
        :math:`e^{-isH_h^p/2}` boundary wrapped around the merged bodies:

        .. math::
            e^{-isH_h^p/2}
            \left( e^{-isH_I/2} e^{-isH_h^g} e^{-isH_I/2} e^{-isH_h^p} \right)^{r-1}
            e^{-isH_I/2} e^{-isH_h^g} e^{-isH_I/2} e^{-isH_h^p/2}.

        Placing a hopping tiling outermost is the "PIG" ordering of Eqs. (16a)-(16b) of
        :cite:`Apel2026`. It is a deliberate deviation from Campbell's own Eq. (D2)
        :cite:`Campbell2022`, which puts :math:`H_I` outermost ("IPG"): PIG merges the
        more resource-intensive hopping layers across step boundaries, where IPG merges
        the cheaper interaction layers.

        Each plaquette hopping evolution is exact. The builder diagonalizes its
        single-particle matrix, :math:`T = V \Lambda V^\dagger`, and applies

        .. math::
            e^{-isH_\square} = U_V e^{-is\sum_m \lambda_m n_m} U_V^\dagger.

        Thus the circuit switches to the plaquette momentum basis, applies two nonzero
        eigenvalue phases, and switches back; the four-cycle hopping matrix has only two
        non-trivial eigenvalues, so two phase rotations suffice (App. E, Eqs. (E6)-(E10)
        of :cite:`Campbell2022`). Trotter error comes only from splitting
        :math:`H_I`, :math:`H_h^p`, and :math:`H_h^g`.

        Args:
            qubit_hamiltonian: Qubit operator wrapping a ``LatticeContainer``.

        Returns:
            UnitaryRepresentation: The segmented plaquette product formula.

        Raises:
            NotImplementedError: If the configured Trotter order is not 2.
            ValueError: If the lattice's bonds do not match a periodic square tiling with uniform hopping.

        """
        order = self._settings.get("order")
        if order != 2:
            raise ValueError(f"HubbardPlaquetteTrotter supports order 2 only, got {order}.")

        # 1. Geometry, and the model angles the settings imply. The lattice's edge weight
        # scales the hopping the settings carry.
        lattice, width, height, weight, sections = self._lattice_geometry(qubit_hamiltonian)
        num_sites = lattice.num_sites
        hopping = float(self._settings.get("t")) * weight

        interaction = float(self._settings.get("u"))
        epsilon = float(self._settings.get("epsilon"))
        single_z = -(0.5 * epsilon + 0.25 * interaction)
        pair_z = 0.25 * interaction
        identity = (epsilon + 0.25 * interaction) * num_sites

        # 2. Step count, reusing the hopping amplitude and tilings resolved above.
        time, power_repetitions = self._resolve_power()
        num_divisions = self._step_count(hopping, sections, width, height, time)
        delta = time / num_divisions

        return UnitaryRepresentation(
            container=HubbardPlaquetteContainer(
                width=width,
                height=height,
                interaction_angle=pair_z * delta,
                onsite_angle=single_z * delta,
                identity_angle=identity * delta,
                hopping_angle=2.0 * hopping * delta,
                step_reps=num_divisions * power_repetitions,
                scale=time,
            )
        )

    def _resolve_num_divisions(self, qubit_hamiltonian: QubitOperator, time: float) -> int:
        """Return the step count the builder would use for this operator and duration.

        Args:
            qubit_hamiltonian: The lattice-backed operator being evolved.
            time: Duration of the evolution.

        Returns:
            The number of Trotter steps, at least one.

        """
        _, width, height, weight, sections = self._lattice_geometry(qubit_hamiltonian)
        hopping = float(self._settings.get("t")) * weight
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

        The error constant ``W_PLAQ`` is derived for this exact splitting in
        :cite:`Campbell2022`: Eq. (20) states it in the main text as
        ``W_PLAQ <= W_SO2 + (3/24) ||[[H_h^p, H_h^g], H_h^g]||``, and App. D carries the
        derivation, restating the same result as Eq. (D6). The energy budget is converted
        into a step count with the exact form retained by Algorithm 1 of :cite:`Apel2026`
        rather than its small-angle limit; see the comments on the conversion below.

        The bound deliberately ignores the single-mode term ``single_z``, which is nonzero
        whenever ``epsilon != -U/2``. That term is a *uniform* on-site energy, so under the
        Jordan-Wigner image it is proportional to ``M - 2 N``, where ``N`` is the total
        number operator. Hopping and interaction both conserve particle number, so this
        layer commutes with every other factor and contributes no Trotter error at any
        ordering; only its scalar part matters, and that is carried by ``identity_angle``.

        Warning:
            ``W_PLAQ`` below is Campbell's closed form, which is derived for the "IPG"
            factor ordering (interaction outermost, Eq. (D2) of :cite:`Campbell2022`).
            The circuit this builder emits uses the "PIG" ordering instead, and Sec. 4.8.2
            of :cite:`Apel2026` states that changing the ordering requires recomputing the
            commutator bound: under PIG the pure-hopping block ``[[H_h^p, H_h^g], .]``
            that Campbell evaluates as ``W_extra2`` is replaced by a mixed
            interaction-hopping block, and the accumulated operator following the first
            factor is no longer purely hopping, so the ``W_SO2 + W_extra2`` split does not
            transfer. Apel report the PIG constant to be larger than Campbell's IPG value
            at every lattice size they plot (their Fig. 18, even ``L`` from 4 to 20; by
            about 10% for ``L >= 10`` and about 20% at ``L = 4``). Since ``r`` scales as
            ``sqrt(W_PLAQ)``, reusing the IPG constant here under-counts the steps by
            roughly 5% for ``L >= 10``, in the optimistic direction. Apel give no closed
            form for the PIG constant -- they evaluate it numerically (their Eq. (29) and
            App. C.1) and publish no table of values -- so it is not reproduced here.

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
        interaction = abs(self._settings.get("u"))

        # R_p and R_g are the one-spin, unit-hopping matrices for the pink and gold tilings.
        # Evaluate their trace norms exactly through 1600 sites; beyond that fall back on
        # per-site asymptotic values, 16/pi^2 for ||R_p + R_g||_1 and 3.229 for
        # ||[[R_p, R_g], R_g]||_1. These two constants are NOT tabulated by Campbell: App. D
        # of Campbell2022 states only the cruder bounds (3/2) L^2 and (10/3) L^2, and its
        # Table III tabulates extensive norms for L <= 32 rather than per-site limits.
        # 16/pi^2 = 1.62114 is the exact thermodynamic limit of ||R_p + R_g||_1 / L^2, since
        # the mean of |cos k_x + cos k_y| over the Brillouin zone is 8/pi^2. 3.229 is an
        # empirical per-site value for the nested commutator; it is accurate near L = 40 but
        # slightly below the limiting value (~3.24), so it is mildly optimistic as L grows.
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

        # 1. W_SO2, the split-operator error constant, from Eq. (10) of :cite:`Campbell2022`.
        # "SO2" indexes Campbell's second split-operator *ordering*, not the Trotter order:
        # both SO1 and SO2 are second-order formulas. Eq. (10) is the main-text statement;
        # its appendix restatement Eq. (C3) carries a spurious hopping factor in the
        # u^2/24 term, so Eq. (10) is the form reproduced here.
        w_so2 = (
            interaction * hopping**2 / 6.0 * num_sites * (math.sqrt(5.0) + 8.0) + interaction**2 / 24.0 * hopping_norm
        )
        # 2. The extra plaquette-splitting contribution, Eq. (D10) of :cite:`Campbell2022`.
        w_extra2 = 3.0 / 24.0 * commutator_norm
        # 3. The complete plaquette error constant, Eq. (D6) of :cite:`Campbell2022`.
        w_plaquette = w_so2 + w_extra2

        # 4. Convert the energy budget into a step count.
        #
        # Write T for the total evolution time. (Campbell reserves tau for the hopping
        # amplitude, which appears inside W_PLAQ itself; Apel use tau for this duration.)
        #
        # A single step of size s carries unitary error ||Delta U|| <= W_PLAQ s^3, which is
        # Eq. (F1) of :cite:`Campbell2022`. Summing r steps of size s = T/r by subadditivity
        # gives W_PLAQ T^3 / r^2; that accumulation step is not itself stated by Campbell,
        # who works per-step. The induced error in the estimated *energy* then satisfies
        # |Delta E| <= (2/T) arcsin(||Delta U|| / 2).
        #
        #     r = ceil(sqrt(W_PLAQ T^3 / (2 sin(eps_TS T / 2)))),
        #
        # which is the form retained by Algorithm 1 of :cite:`Apel2026`.
        #
        # Campbell Eq. (F2) instead states the linearized bound Delta_TS <= W s^2, giving
        # s <= sqrt(eps_TS / W). That is the small-angle limit of the above, since
        # 2 sin(x/2) -> x as x -> 0. The two differ by a factor sqrt(x / (2 sin(x / 2)))
        # in r, where x = eps_TS T is the angle for that round. Because sin x <= x the
        # linearized rule is always the optimistic one, so the exact form is used here.
        # For the QPE ladder in examples/benchmark/sample_hubbard_resources.py the largest
        # round runs at x ~ 0.78, worth about 1.3% on that round, and the change in the
        # summed step count is between 0 and 0.74% depending on lattice size -- largest for
        # small lattices and absorbed entirely by the integer ceiling for L >= 40.
        duration = abs(time)
        if w_plaquette <= 0.0 or duration == 0.0:
            automatic = 1
        else:
            # ||Delta U|| can never exceed 2, so the arcsine saturates at eps_TS T = pi.
            # Clamping there keeps the step count monotonic in the accuracy target; a looser
            # target than that is certified by any step count and simply yields the floor.
            phase = min(target_accuracy * duration, math.pi)
            automatic = max(1, math.ceil(math.sqrt(w_plaquette * duration**3 / (2.0 * math.sin(phase / 2.0)))))
        Logger.debug(f"HubbardPlaquetteTrotter: bound gives r={automatic}, manual is {manual}.")
        return max(manual, automatic)

    @staticmethod
    def _plaquette_sections(width: int, height: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Return Campbell's pink and gold four-cycle tilings of a periodic square lattice.

        The cycles are read from the Q# implementation that executes them, so the bond
        validation and the error bound are derived from the same tiling the circuit
        applies rather than a second copy of the construction.

        Args:
            width: Number of lattice columns.
            height: Number of lattice rows.

        Returns:
            The pink and gold tilings, each a list of four-cycles in cycle order, for one
            spin sector.

        Raises:
            ValueError: If the lattice cannot be tiled into vertex-disjoint four-cycles.

        """
        if width % 2 or height % 2:
            raise ValueError(f"Plaquette tiling requires even side lengths, got {width}x{height}.")
        if (width < 4 or height < 4) and (width, height) != (2, 2):
            raise ValueError(
                f"Plaquette tiling requires both sides to be at least four, or exactly 2x2, "
                f"got {width}x{height}. Other periodic lattices with a side below four wrap "
                "onto themselves and cannot be tiled."
            )

        sites = width * height
        sections = []
        for pink in ("true", "false"):
            cycles = get_qsharp_context().eval(
                f"QDKChemistry.Utils.HubbardPlaquette.PlaquetteSection({width}, {height}, {pink})"
            )
            # Q# emits both spin sectors; the caller works in one and offsets the other.
            sections.append([tuple(int(mode) for mode in cycle) for cycle in cycles if int(cycle[0]) < sites])
        return sections[0], sections[1]


class HubbardPlaquetteTrotterSettings(TrotterSettings):
    """Settings for the plaquette Trotter builder."""

    def __init__(self):
        """Initialize the settings, adding the model parameters to the Trotter defaults."""
        super().__init__()
        self._set_default("t", "float", 1.0, "Uniform hopping amplitude of the Fermi-Hubbard model.")
        self._set_default("u", "float", 0.0, "Uniform on-site interaction of the Fermi-Hubbard model.")
        self._set_default(
            "epsilon",
            "float",
            0.0,
            "Uniform on-site energy; use -u/2 for the particle-hole-shifted model.",
        )
