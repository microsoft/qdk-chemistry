r"""Second-order plaquette Trotterization for the uniform Fermi-Hubbard model.

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
        :math:`H_I`, :math:`H_h^p`, and :math:`H_h^g`.

        Args:
            qubit_hamiltonian: Qubit operator wrapping a ``LatticeContainer``.

        Returns:
            UnitaryRepresentation: The segmented plaquette product formula.

        Raises:
            NotImplementedError: If the configured Trotter order is not 2.
            ValueError: If the lattice's bonds do not match a periodic square tiling.

        """
        order = self._settings.get("order")
        if order != 2:
            raise NotImplementedError(
                f"HubbardPlaquetteTrotter supports order 2 only, got {order}. Campbell's W_PLAQ is a "
                "second-order constant and would understate a first-order product's error."
            )
        atol = self._settings.get("weight_threshold")

        # 1. Geometry, and the on-site angles the model settings imply.
        lattice, width, height = self._lattice_geometry(qubit_hamiltonian)
        num_sites = width * height
        interaction = float(self._settings.get("U"))
        epsilon = float(self._settings.get("epsilon"))
        single_z = -(0.5 * epsilon + 0.25 * interaction)
        pair_z = 0.25 * interaction
        identity = (epsilon + 0.25 * interaction) * num_sites

        # 2. Hopping amplitude; the tilings themselves are derived in Q# from the shape.
        hopping, bonds = self._uniform_hopping(lattice, atol)
        pink, gold = self._plaquette_sections(width, height)
        tiled = {frozenset((cycle[i], cycle[(i + 1) % 4])) for cycle in pink + gold for i in range(4)}
        if bonds != tiled:
            raise ValueError(
                f"The lattice's bond graph does not match a periodic {width}x{height} "
                f"square lattice: {len(tiled - bonds)} lattice bond(s) absent from the graph and "
                f"{len(bonds - tiled)} graph bond(s) outside the tiling. Check the lattice "
                "dimensions, the boundary conditions, and that sites are numbered row-major."
            )

        # 3. Step count, reusing the hopping amplitude and tilings resolved above.
        time, power_repetitions = self._resolve_power()
        num_divisions = self._step_count(hopping, (pink, gold), width, height, time)
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

    def _uniform_hopping(self, lattice, atol: float) -> tuple[float, set[frozenset[int]]]:
        """Return the uniform hopping amplitude and the lattice's bonds.

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
        self._set_default("U", "float", 0.0, "Uniform on-site interaction of the Fermi-Hubbard model.")
        self._set_default(
            "epsilon",
            "float",
            0.0,
            "Uniform on-site energy; use -U/2 for the particle-hole-shifted model.",
        )
