"""Second-order plaquette Trotterization for the uniform Fermi-Hubbard model."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder, TimeEvolutionSettings
from qdk_chemistry.data.qubit_operator import QubitOperator
from qdk_chemistry.data.qubit_operator.containers.lattice import LatticeContainer
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.hubbard_plaquette import HubbardPlaquetteContainer
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import get_qsharp_context

if TYPE_CHECKING:
    from qdk_chemistry.data import LatticeGeometry

__all__: list[str] = [
    "HubbardPlaquetteTrotter",
    "HubbardPlaquetteTrotterSettings",
]

#: Absolute tolerance for matching lattice site positions, whose spacing is one.
_GEOMETRY_TOLERANCE = 1e-9

#: Decimal places the site coordinates are rounded to before they are counted.
_GEOMETRY_DECIMALS = 9

#: The :class:`LatticeContainer` couplings this builder reads, and the only ones it accepts.
_COUPLING_NAMES = frozenset({"hopping", "interaction"})


class HubbardPlaquetteTrotterSettings(TimeEvolutionSettings):
    """Settings for the plaquette Trotter builder."""

    def __init__(self):
        """Initialize the plaquette-specific settings."""
        super().__init__()
        self._set_default("order", "int", 2, "The Trotter decomposition order; only 2 is supported.")
        self._set_default(
            "target_accuracy",
            "double",
            0.0,
            "Target accuracy for automatic plaquette step computation (0.0 means disabled).",
        )
        self._set_default(
            "num_divisions",
            "int",
            0,
            "Explicit number of plaquette Trotter steps (0 means automatic).",
        )
        self._set_default(
            "num_electrons",
            "int",
            -1,
            "Electron count for the scalar shift to the conventional model; -1 leaves it unshifted.",
        )


class HubbardPlaquetteTrotter(TimeEvolutionBuilder):
    r"""Plaquette Trotterization of the Fermi-Hubbard model on a periodic two-dimensional square lattice.

    The builder takes a :class:`~qdk_chemistry.data.QubitOperator` wrapping a
    :class:`~qdk_chemistry.data.qubit_operator.containers.lattice.LatticeContainer`, which
    carries the lattice geometry and the model couplings. It reads exactly two couplings,
    ``"hopping"`` (the uniform hopping amplitude :math:`t`) and ``"interaction"`` (the uniform
    on-site interaction :math:`U`), and rejects a container missing either or carrying any other.

    The interaction is taken in Campbell's particle-hole symmetric form
    :math:`U \sum_i (n_{i\uparrow} - 1/2)(n_{i\downarrow} - 1/2)`, whose Jordan-Wigner image is
    pure :math:`ZZ`: there is no single-mode :math:`Z` layer. The conventional
    :math:`U \sum_i n_{i\uparrow} n_{i\downarrow}` model differs by :math:`U\eta/2 - UM/4` on a
    state of :math:`\eta` electrons, with :math:`M` the site count. Setting ``num_electrons``
    records that offset so the phase-to-energy conversion reports the conventional energy;
    the Q# circuit omits the scalar identity term. Leaving it unset reports the symmetric model's
    energy directly.

    The plaquette decomposition, its error constant, and the exact four-mode plaquette
    evolution are Campbell's :cite:`Campbell2022`. The factor ordering and the step-count
    rule follow the later compilation of the same algorithm in :cite:`Apel2026`.
    """

    def __init__(
        self,
        order: int = 2,
        *,
        num_electrons: int | None = None,
        time: float = 0.0,
        target_accuracy: float = 0.0,
        num_divisions: int = 0,
        power: int = 1,
        power_strategy: str = "repeat",
    ):
        """Initialize the builder.

        Args:
            order: Trotter decomposition order. Only 2 is supported.
            num_electrons: Electron count for the classical shift to the conventional model; ``None`` skips it.
            time: The evolution time. Defaults to 0.0.
            target_accuracy: Target accuracy for auto Trotter step computation. Use 0.0 to disable.
            num_divisions: Number of Trotter steps. Max of this and the auto value is used.
            power: The power to raise the unitary to. Defaults to 1.
            power_strategy: Strategy for ``U^power``: ``"rescale"`` or ``"repeat"``.

        Raises:
            ValueError: If *order* is anything other than 2.

        """
        if order != 2:
            raise ValueError(f"HubbardPlaquetteTrotter supports order 2 only, got {order}.")

        super().__init__()
        settings = HubbardPlaquetteTrotterSettings()
        settings.set("time", time)
        settings.set("power", power)
        settings.set("power_strategy", power_strategy)
        settings.set("order", order)
        settings.set("target_accuracy", target_accuracy)
        settings.set("num_divisions", num_divisions)
        if num_electrons is not None:
            settings.set("num_electrons", int(num_electrons))
        self._settings = settings

    def name(self) -> str:
        """Return ``hubbard_plaquette`` as the algorithm name."""
        return "hubbard_plaquette"

    def type_name(self) -> str:
        """Return ``hamiltonian_unitary_builder`` as the algorithm type name."""
        return "hamiltonian_unitary_builder"

    def _lattice_geometry(self, qubit_hamiltonian: QubitOperator):
        """Return the lattice shape and its plaquette tilings.

        The geometry is validated here: it must be a unit-spaced square lattice periodic in
        both directions and numbered ``y * width + x``, and the tilings resolved from Q# must
        cover exactly its nearest-neighbor bonds. The tilings are resolved to run that check,
        so they are returned alongside the shape.

        Args:
            qubit_hamiltonian: The operator to inspect.

        Returns:
            A tuple of the lattice width, height, and its pink and gold tilings.

        Raises:
            TypeError: If the operator does not wrap a ``LatticeContainer``.
            ValueError: If the geometry is not a periodic square grid tiled by plaquettes.

        """
        if not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError("HubbardPlaquetteTrotter requires a QubitOperator containing a LatticeContainer")

        container = qubit_hamiltonian.get_container()
        if not isinstance(container, LatticeContainer):
            raise TypeError(
                f"HubbardPlaquetteTrotter requires a QubitOperator containing a LatticeContainer, but the "
                f"operator wraps a {container.type!r} container."
            )
        width, height = self._square_grid_shape(container.geometry)

        sections = self._plaquette_sections(width, height)
        pink, gold = sections
        # Walk each plaquette, pairing every corner with the next and wrapping the last to the first.
        tiled: set[frozenset[int]] = set()
        for cycle in pink + gold:
            for index, corner in enumerate(cycle):
                next_corner = cycle[(index + 1) % len(cycle)]
                tiled.add(frozenset((corner, next_corner)))

        bonds = self._periodic_square_bonds(width, height)
        if bonds != tiled:
            raise ValueError(
                f"The plaquette tiling does not match a periodic {width}x{height} square lattice: "
                f"{len(bonds - tiled)} lattice bond(s) absent from the tiling and "
                f"{len(tiled - bonds)} tiled bond(s) outside the lattice. "
            )

        Logger.debug(f"HubbardPlaquetteTrotter: periodic {width}x{height} square lattice, {len(bonds)} bonds per spin.")
        return width, height, sections

    @staticmethod
    def _model_couplings(container: LatticeContainer) -> tuple[float, float]:
        """Return the hopping amplitude and on-site interaction the container carries.

        Args:
            container: The lattice container to read.

        Returns:
            A tuple of the ``"hopping"`` and ``"interaction"`` couplings.

        Raises:
            ValueError: If either coupling is missing, or the container carries any other, which
                this builder would otherwise silently drop from the evolution.

        """
        couplings = container.couplings
        missing = sorted(_COUPLING_NAMES - couplings.keys())
        unexpected = sorted(couplings.keys() - _COUPLING_NAMES)
        if missing or unexpected:
            raise ValueError(
                "HubbardPlaquetteTrotter evolves the uniform Fermi-Hubbard model and needs exactly the "
                f"couplings {sorted(_COUPLING_NAMES)}; the lattice container is missing {missing} and "
                f"carries unsupported {unexpected}."
            )
        return couplings["hopping"], couplings["interaction"]

    @staticmethod
    def _square_grid_shape(geometry: LatticeGeometry) -> tuple[int, int]:
        """Return the width and height of a doubly periodic square lattice geometry.

        The shape is read back from the site positions, which is what makes it survive a
        round trip: the geometry stores no separate dimension field that could disagree
        with them. A geometry whose positions are not the canonical unit-spaced grid, such
        as a triangular or honeycomb patch, is rejected rather than silently mis-tiled.

        Args:
            geometry: The :class:`~qdk_chemistry.data.LatticeGeometry` to inspect.

        Returns:
            The number of columns and rows of the lattice.

        Raises:
            ValueError: If the geometry is not a square lattice periodic in both directions.

        """
        positions = np.asarray(geometry.positions, dtype=float)
        periods = geometry.periods
        directions = 0 if periods is None else int(np.asarray(periods).shape[0])
        if directions != 2:
            raise ValueError(
                "HubbardPlaquetteTrotter tiles a lattice periodic in both directions, but the geometry is "
                f"periodic in {directions} direction(s). Build it with "
                "LatticeGeometry.square(nx, ny, periodic_x=True, periodic_y=True)."
            )

        # Unit spacing makes the distinct coordinates along each axis the lattice dimensions.
        width = int(np.unique(positions[:, 0].round(_GEOMETRY_DECIMALS)).size)
        height = int(np.unique(positions[:, 1].round(_GEOMETRY_DECIMALS)).size)
        expected_positions = np.array(
            [(float(x), float(y)) for y in range(height) for x in range(width)],
            dtype=float,
        ).reshape(-1, 2)
        expected_periods = np.array([(float(width), 0.0), (0.0, float(height))], dtype=float)
        if (
            width * height != positions.shape[0]
            or not np.allclose(positions, expected_positions, rtol=0.0, atol=_GEOMETRY_TOLERANCE)
            or not np.allclose(np.asarray(periods, dtype=float), expected_periods, rtol=0.0, atol=_GEOMETRY_TOLERANCE)
        ):
            raise ValueError(
                "HubbardPlaquetteTrotter tiles a unit-spaced square lattice numbered y * width + x, which "
                f"the {positions.shape[0]}-site geometry does not match. Build it with "
                "LatticeGeometry.square(nx, ny, periodic_x=True, periodic_y=True)."
            )
        return width, height

    @staticmethod
    @cache
    def _bond_multiplicity(width: int, height: int) -> int:
        r"""Return how many times a periodic image joins each pair of neighboring sites.

        On a ring of two sites the neighbor is reached going either way around, so the pair is
        joined twice and its hopping amplitude is :math:`2t` rather than :math:`t`. Every longer
        ring joins a pair once. The tiling carries one hopping angle for the whole lattice, so a
        shape that doubles in one direction and not the other has no single amplitude and is
        rejected rather than silently evolved under the wrong Hamiltonian.

        ``_periodic_square_bonds`` returns a set, which collapses the doubled bond, so the
        multiplicity has to be recovered from the extents instead of counted from the bonds.

        Args:
            width: Number of sites along x.
            height: Number of sites along y.

        Returns:
            int: The shared bond multiplicity, 1 or 2.

        Raises:
            ValueError: If the two directions disagree, which no single angle can express.

        """
        along_x = 2 if width == 2 else 1
        along_y = 2 if height == 2 else 1
        if along_x != along_y:
            raise ValueError(
                f"A periodic {width}x{height} lattice joins its neighbors {along_x} time(s) along x but "
                f"{along_y} time(s) along y, so one hopping angle cannot describe both. Use extents that "
                "are both two or both greater than two."
            )
        return along_x

    @staticmethod
    @cache
    def _periodic_square_bonds(width: int, height: int) -> frozenset[frozenset[int]]:
        """Return the nearest-neighbor bonds of a periodic square lattice.

        Args:
            width: Number of lattice columns.
            height: Number of lattice rows.

        Returns:
            Each bond once, as an unordered pair of site indices.

        """
        return frozenset(
            frozenset(bond)
            for row in range(height)
            for col in range(width)
            for bond in (
                (row * width + col, row * width + (col + 1) % width),
                (row * width + col, ((row + 1) % height) * width + col),
            )
        )

    @staticmethod
    @cache
    def _plaquette_sections(width: int, height: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Return Campbell's pink and gold four-cycle tilings of a periodic square lattice.

        Args:
            width: Number of lattice columns.
            height: Number of lattice rows.

        Returns:
            The pink and gold tilings for one spin sector, each a list of four-site cycles in cycle order.

        Raises:
            ValueError: If the lattice cannot be tiled into vertex-disjoint four-cycles.

        """
        if width % 2 or height % 2:
            raise ValueError(f"Plaquette tiling requires even side lengths, got {width}x{height}.")

        if (width < 4 or height < 4) and (width, height) != (2, 2):
            raise ValueError(
                f"Plaquette tiling requires both sides to be at least four, or exactly 2x2, got {width}x{height}. "
            )

        sites = width * height
        sections = []
        for pink in ("true", "false"):
            cycles = get_qsharp_context().eval(
                f"QDKChemistry.Utils.HubbardPlaquette.PlaquetteSection({width}, {height}, {pink})"
            )
            # Q# emits both spin sectors; keep the one whose cycles start inside the first sector.
            sector = []
            for cycle in cycles:
                modes = tuple(int(mode) for mode in cycle)
                if modes[0] < sites:
                    sector.append(modes)
            sections.append(sector)
        return sections[0], sections[1]

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> UnitaryRepresentation:
        r"""Build the plaquette Trotter unitary representation for Fermi-Hubbard.

        For a periodic :math:`w \times h` square lattice of :math:`M = wh` sites, writing
        :math:`\langle ij \rangle` for its bonds, :math:`\sigma \in \{\uparrow, \downarrow\}` for spin,
        :math:`a_{i\sigma}` for the fermionic annihilation operator on site :math:`i` with spin
        :math:`\sigma`, and :math:`n_{i\sigma} = a^\dagger_{i\sigma} a_{i\sigma}` for the number operator,
        the Fermi-Hubbard Hamiltonian is

        .. math::
            H = -t \sum_{\langle ij \rangle, \sigma} \left( a^\dagger_{i\sigma} a_{j\sigma}
                + a^\dagger_{j\sigma} a_{i\sigma} \right)
              + U \sum_i \left( n_{i\uparrow} - \tfrac{1}{2} \right)
                         \left( n_{i\downarrow} - \tfrac{1}{2} \right),

        where :math:`t` is the hopping amplitude and :math:`U` the on-site interaction. The interaction
        is written in the particle-hole symmetric form of :cite:`Campbell2022`; the conventional
        :math:`U \sum_i n_{i\uparrow} n_{i\downarrow}` model differs by :math:`U N / 2 - U M / 4`, with
        :math:`N` the total number operator, so its energy follows from the simulated
        :math:`\tilde{E}` by the classical shift :math:`E = \tilde{E} + U\eta/2 - UM/4` on a state of
        :math:`\eta` electrons. The Hamiltonian is split into :math:`H_I`, the diagonal interaction
        above, and :math:`H_h^p`, :math:`H_h^g`, the pink and gold tilings: two sets of
        vertex-disjoint four-cycles that together cover every bond exactly once.

        Since :math:`n_m - 1/2 = -Z_m/2` under :math:`n_m = (1 - Z_m)/2`, the interaction is a pure
        :math:`ZZ` layer between a site and its spin partner,

        .. math::
            U \sum_i \left( n_{i\uparrow} - \tfrac{1}{2} \right)
                     \left( n_{i\downarrow} - \tfrac{1}{2} \right)
            = \frac{U}{4} \sum_i Z_{i} Z_{i+M},

        while a hopping bond becomes a two-local term dressed by a Jordan-Wigner string of :math:`Z`
        operators between its endpoints,

        .. math::
            -t \left( a^\dagger_m a_n + a^\dagger_n a_m \right)
            = -\frac{t}{2} \left( X_m Z_{m+1} \cdots Z_{n-1} X_n + Y_m Z_{m+1} \cdots Z_{n-1} Y_n \right).

        To avoid the Jordan-Wigner :math:`Z` strings, each hopping layer first uses a network of
        fermionic swaps to route the four modes of every plaquette into a contiguous block.
        Interacting fermionic sites become adjacent in the Jordan-Wigner ordering thus localizing the hopping
        Two radix-two fermionic Fourier butterflies then
        transform its four-cycle hopping matrix, whose spectrum is
        :math:`\mathrm{diag}(2t, 0, -2t, 0)`, into a single adjacent two-mode hopping term.
        The resulting equal-angle :math:`XX` and :math:`YY`
        rotations can be batched by Hamming-weight phasing, after which the Fourier
        butterflies and routing are uncomputed.
        Each hopping layer is therefore an fswap routing, a basis change, two nonzero eigenvalue phases,
        and the inverse basis change and routing, with every plaquette of the tiling handled in parallel.

        Thus, over a total evolution time :math:`T` split into :math:`r` steps of duration
        :math:`\delta = T/r`, the plaquette Trotterization gives:

        .. math::
            e^{-i\delta H_h^p/2}
            \left( e^{-i\delta H_I/2} e^{-i\delta H_h^g} e^{-i\delta H_I/2} e^{-i\delta H_h^p} \right)^{r-1}
            e^{-i\delta H_I/2} e^{-i\delta H_h^g} e^{-i\delta H_I/2} e^{-i\delta H_h^p/2}.

        Placing a hopping tiling outermost is the "PIG" ordering of Eqs. (16a)-(16b) of :cite:`Apel2026`.
        Trotter error comes only from splitting :math:`H_I`, :math:`H_h^p`, and :math:`H_h^g`; each layer
        is applied exactly.

        Args:
            qubit_hamiltonian: Qubit operator wrapping a ``LatticeContainer``.

        Returns:
            UnitaryRepresentation: The segmented plaquette product formula.

        Raises:
            ValueError: If the order is not 2, the geometry is not a periodic square lattice, or the
                container's couplings are not exactly ``"hopping"`` and ``"interaction"``.

        """
        order = self._settings.get("order")
        if order != 2:
            raise ValueError(f"HubbardPlaquetteTrotter supports order 2 only, got {order}.")

        # 1. Geometry
        width, height, _sections = self._lattice_geometry(qubit_hamiltonian)

        # 2. Model parameters
        hopping_amplitude, interaction = self._model_couplings(qubit_hamiltonian.get_container())
        hopping = hopping_amplitude * self._bond_multiplicity(width, height)
        pair_angle = 0.25 * interaction
        num_electrons = int(self._settings.get("num_electrons"))
        num_sites = width * height
        shift = 0.0 if num_electrons < 0 else interaction * (0.5 * num_electrons - 0.25 * num_sites)

        # 3. Step count
        time, power_repetitions = self._resolve_power()
        num_divisions = self._step_count(hopping, interaction, width, height, time)
        delta_time = time / num_divisions

        return UnitaryRepresentation(
            container=HubbardPlaquetteContainer(
                width=width,
                height=height,
                interaction_angle=pair_angle * delta_time,
                constant_shift=shift * delta_time,
                hopping_angle=2.0 * hopping * delta_time,
                step_reps=num_divisions * power_repetitions,
                scale=time,
            )
        )

    @staticmethod
    @cache
    def _hopping_trace_norm(width: int, height: int) -> float:
        """Return ``||H_h|| = ||R_p + R_g||_1`` for a unit hopping amplitude.

        The two tilings cover every bond exactly once, so their sum is the whole periodic
        lattice's single-particle matrix. That matrix is free-fermionic and the discrete
        Fourier transform diagonalizes it, giving eigenvalues ``2 cos(k_x) + 2 cos(k_y)``.
        Summing their magnitudes is exact at every lattice size and costs ``O(M)`` rather
        than the ``O(M^3)`` of a singular value decomposition.

        A side of two is the exception: its opposite neighbours are the same site, so the
        tiling carries one bond where a torus carries two and the closed form would
        double-count. That case falls back on the tiling matrices themselves.
        """
        if width < 4 or height < 4:
            sections = HubbardPlaquetteTrotter._plaquette_sections(width, height)
            matrices = []
            for cycles in sections:
                matrix = np.zeros((width * height, width * height))
                for cycle in cycles:
                    for index in range(4):
                        site_a, site_b = cycle[index], cycle[(index + 1) % 4]
                        matrix[site_a, site_b] = matrix[site_b, site_a] = -1.0
                matrices.append(matrix)
            matrix_p, matrix_g = matrices
            return float(np.abs(np.linalg.eigvalsh(matrix_p + matrix_g)).sum())

        momenta_x = 2.0 * math.pi * np.arange(width) / width
        momenta_y = 2.0 * math.pi * np.arange(height) / height
        spectrum = 2.0 * np.cos(momenta_x)[:, None] + 2.0 * np.cos(momenta_y)[None, :]
        return float(np.abs(spectrum).sum())

    @staticmethod
    @cache
    def _commutator_trace_norm(width: int, height: int) -> float:
        """Return ``||[[R_p, R_g], R_g]||_1`` for a unit hopping amplitude.

        The nested commutator is invariant under translations of the two-by-two plaquette
        supercell. A Fourier transform over those cells splits it into the four-by-four
        Hermitian blocks assembled below. The trace norm is therefore the sum of the
        magnitudes of their eigenvalues.

        Each block entry couples a site to its partner one bond away, in the same cell, and to
        its partner three bonds away, two cells over. The second coupling therefore carries the
        phase of a two-cell translation, ``exp(2ik)``. With two cells along a side the two
        partners coincide and cancel, which is why the commutator vanishes on a 4x4 lattice.
        """
        cells_x, cells_y = width // 2, height // 2
        momenta_x = 2.0 * math.pi * np.arange(cells_x) / cells_x
        momenta_y = 2.0 * math.pi * np.arange(cells_y) / cells_y
        phase_x = np.exp(2j * momenta_x)[:, None]
        phase_y = np.exp(2j * momenta_y)[None, :]

        fourier = np.zeros((cells_x, cells_y, 4, 4), dtype=complex)
        fourier[:, :, 0, 1] = -2.0 + 2.0 / phase_x
        fourier[:, :, 1, 0] = -2.0 + 2.0 * phase_x
        fourier[:, :, 0, 2] = -2.0 + 2.0 / phase_y
        fourier[:, :, 2, 0] = -2.0 + 2.0 * phase_y
        fourier[:, :, 1, 3] = -2.0 + 2.0 / phase_y
        fourier[:, :, 3, 1] = -2.0 + 2.0 * phase_y
        fourier[:, :, 2, 3] = -2.0 + 2.0 / phase_x
        fourier[:, :, 3, 2] = -2.0 + 2.0 * phase_x
        return float(np.abs(np.linalg.eigvalsh(fourier)).sum())

    def _step_count(
        self,
        hopping: float,
        interaction: float,
        width: int,
        height: int,
        time: float,
    ) -> int:
        """Determine the number of Trotter steps from the plaquette-specific error constant.

        ``W_PLAQ <= W_SO2 + W_extra2``, Eq. (20) of :cite:`Campbell2022`.

        This constant is derived for the "IPG" factor ordering (interaction outermost), while the circuit
        this builder emits uses the "PIG" ordering based on Sec. 4.8.2 of :cite:`Apel2026`.
        The pure-hopping block ``[[H_h^p, H_h^g], .]`` is replaced by a mixed interaction-hopping block,
        which is reported to make the constant slightly larger than the IPG one. The difference is neglected here.

        Here::

            W_extra2 = (3/24) ||[[R^p, R^g], R^g]||
            W_SO2 <= (uτ^2 / 6) * L^2 * (√5+8) + (u^2τ / 24) ||H_h||

        The number of Trotter steps is determined by::

            r = ceil(sqrt(W_PLAQ T^3 / (2 sin(eps_TS T / 2)))),

        where T is the total evolution time, s = T/r the duration of a single Trotter step, and eps_TS the
        energy error budget for the Suzuki-Trotter approximation.

        A single step of size s carries unitary error ``||Delta U|| <= W_PLAQ s^3``.
        Summing r steps of size s = T/r by subadditivity gives ``||Delta U|| <= W_PLAQ T^3 / r^2``.
        The induced error in the estimated energy then satisfies ``|Delta E| <= (2/T) arcsin(||Delta U|| / 2)``.

        Args:
            hopping: Uniform hopping amplitude.
            interaction: Uniform on-site interaction.
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
        interaction = abs(interaction)

        hopping_norm = self._hopping_trace_norm(width, height) * hopping
        commutator_norm = self._commutator_trace_norm(width, height) * hopping**3

        w_so2 = (
            interaction * hopping**2 / 6.0 * num_sites * (math.sqrt(5.0) + 8.0) + interaction**2 / 24.0 * hopping_norm
        )
        w_extra2 = 3.0 / 24.0 * commutator_norm
        w_plaquette = w_so2 + w_extra2

        duration = abs(time)
        if w_plaquette <= 0.0 or duration == 0.0:
            automatic = 1
        else:
            phase = min(target_accuracy * duration, math.pi)
            automatic = max(1, math.ceil(math.sqrt(w_plaquette * duration**3 / (2.0 * math.sin(phase / 2.0)))))
        Logger.debug(f"HubbardPlaquetteTrotter: bound gives r={automatic}, manual is {manual}.")
        return max(manual, automatic)
