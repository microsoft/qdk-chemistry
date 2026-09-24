r"""Second-order plaquette Trotterization for the uniform Fermi-Hubbard model.

The builder reads an unmapped lattice :class:`~qdk_chemistry.data.Hamiltonian`
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
import scipy.sparse

from qdk_chemistry.algorithms.hamiltonian_input import (
    HamiltonianInput,
    system_num_qubits,
    validate_hamiltonian_input,
)
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter, TrotterSettings
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    BatchedExponentiatedPauliTerm,
    ConjugatedExponentiatedPauliTerm,
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import Hamiltonian

__all__: list[str] = [
    "PlaquetteTrotter",
    "PlaquetteTrotterSettings",
]


class PlaquetteTrotter(Trotter):
    """Build a second-order product formula from exact plaquette evolutions.

    The builder takes an unmapped **lattice Hamiltonian**
    (:class:`~qdk_chemistry.data.Hamiltonian`, as built by
    :func:`~qdk_chemistry.utils.model_hamiltonians.create_hubbard_hamiltonian`) and
    derives :math:`H_I`, :math:`H_h^p`, and :math:`H_h^g` straight from its one- and
    two-body integrals, constructing the Jordan-Wigner image analytically rather than
    mapping the operator. The declared lattice is checked against those integrals before
    the pink and gold plaquette tilings are built.

    A :class:`~qdk_chemistry.data.QubitOperator` is rejected: the plaquette tilings are
    defined by the lattice bonds and the on-site interaction, which the mapped operator
    no longer carries explicitly.

    Note:
        This expects a uniform Fermi-Hubbard model on a periodic square lattice whose
        sides are even and either both at least four or exactly 2x2. Incompatible
        geometry or non-uniform hopping or interaction raises a :class:`ValueError`.

    """

    def __init__(
        self,
        order: int = 2,
        *,
        lattice_width: int = 0,
        lattice_height: int = 0,
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
            lattice_width: Number of lattice columns. Required at run time.
            lattice_height: Number of lattice rows. Required at run time.
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
        settings.set("max_batch", max_batch)
        self._settings = settings

    def name(self) -> str:
        """Return ``plaquette`` as the algorithm name."""
        return "plaquette"

    def accepts_lattice(self) -> bool:
        """Return ``True``; this builder reads a lattice Hamiltonian directly.

        Returns:
            bool: Always ``True``.

        """
        return True

    def accepts_qubit_operator(self) -> bool:
        """Return ``False``; the plaquette tiling needs structure a mapping discards.

        The tilings are built from the lattice's bonds and its on-site interaction, which
        a :class:`~qdk_chemistry.data.QubitOperator` no longer carries explicitly.

        Returns:
            bool: Always ``False``.

        """
        return False

    def _run_impl(self, qubit_hamiltonian: HamiltonianInput) -> UnitaryRepresentation:
        """Construct the plaquette product formula for a lattice Hamiltonian.

        Args:
            qubit_hamiltonian: The lattice Hamiltonian to decompose.

        Returns:
            UnitaryRepresentation: The segmented plaquette product formula.

        Raises:
            TypeError: If the input is not a lattice Hamiltonian.
            NotImplementedError: If the configured Trotter order is not 2.

        """
        validate_hamiltonian_input(
            qubit_hamiltonian,
            accepts_lattice=True,
            accepts_qubit_operator=False,
            algorithm_name=self.name(),
            algorithm_kind="unitary builder",
        )
        order = self._settings.get("order")
        if order != 2:
            raise NotImplementedError(f"PlaquetteTrotter supports order 2 only, got {order}.")
        effective_time, power_repetitions = self._resolve_power()
        return self._trotter(qubit_hamiltonian, effective_time, power_repetitions)

    def _trotter(
        self, qubit_hamiltonian: HamiltonianInput, time: float, power_repetitions: int = 1
    ) -> UnitaryRepresentation:
        r"""Build Campbell's segmented plaquette product formula.

        Define :math:`D=e^{-isH_I}`, :math:`P=e^{-isH_h^p}`, and
        :math:`G=e^{-isH_h^g}`. Repeated symmetric steps
        :math:`P^{1/2} D^{1/2} G D^{1/2} P^{1/2}` merge across repetitions into
        :math:`P^{-1/2}(D^{1/2} G D^{1/2} P)^r P^{1/2}`, following the ordering of
        arXiv:2609.05316 Eqs. (16a)--(16b). 
        :meth:`_decompose_trotter_step` emits the repeated body; this
        method adds the one-time boundary factors.

        Args:
            qubit_hamiltonian: The lattice hamiltonian for 2d hubbard model.
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
        hopping, _, _ = self._split_terms(qubit_hamiltonian, num_sites, self._settings.get("weight_threshold"))
        section_a, _ = self._plaquette_sections(
            self._settings.get("lattice_width"), self._settings.get("lattice_height")
        )
        opening = self._hop_layer(section_a, num_sites=num_sites, hopping=hopping, time=delta * 0.5)
        boundary = [opening] if opening is not None else []

        return UnitaryRepresentation(
            container=PauliProductFormulaContainer(
                step_terms=step_terms,
                step_reps=num_divisions * power_repetitions,
                num_qubits=system_num_qubits(qubit_hamiltonian),
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

        hopping, _, _ = self._split_terms(qubit_hamiltonian, width * height, self._settings.get("weight_threshold"))
        hopping = abs(hopping)
        num_sites = width * height

        interaction = self._interaction_strength(qubit_hamiltonian, num_sites)

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
        Logger.debug(f"PlaquetteTrotter: bound gives r={automatic}, manual is {manual}.")
        return max(manual, automatic)

    def _decompose_trotter_step(
        self,
        qubit_hamiltonian: HamiltonianInput,
        time: float,
        atol: float = 1e-12,
    ) -> list[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm | ConjugatedExponentiatedPauliTerm]:
        r"""Return the repeated bulk body of Campbell's plaquette formula.

        With :math:`D=e^{-isH_I}`, :math:`P=e^{-isH_h^p}`, and
        :math:`G=e^{-isH_h^g}`, the returned terms implement
        :math:`D^{1/2} G D^{1/2} P`: two batched interaction half-layers, one gold
        full layer, and one full pink layer. :meth:`_trotter` supplies the one-time
        :math:`P^{-1/2}` and :math:`P^{1/2}` boundary layers.

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
        num_qubits = system_num_qubits(qubit_hamiltonian)
        if num_qubits != 2 * num_sites:
            raise ValueError(
                f"A {width}x{height} spinful lattice needs {2 * num_sites} qubits, but the "
                f"Hamiltonian has {num_qubits}."
            )

        hopping, diagonal, observed_bonds = self._split_terms(qubit_hamiltonian, num_sites, atol)
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
        # Repeated body 1: D^(1/2), the first interaction half-layer. After the
        # particle-hole shift each onsite interaction contributes one rotation, and its
        # equal-angle family is phased through a single Hamming weight register by
        # default; see the max_batch setting to bound that width.
        open_interaction = self._diagonal_layer(diagonal, time * 0.5, max_batch=self._settings.get("max_batch"))
        # Repeated body 2: G, the gold hopping layer.
        hop_b = self._hop_layer(
            section_b,
            num_sites=num_sites,
            hopping=hopping,
            time=time,
        )
        # Repeated body 3: D^(1/2), the second interaction half-layer.
        close_interaction = self._diagonal_layer(diagonal, time * 0.5, max_batch=self._settings.get("max_batch"))
        # Repeated body 4: P, one full pink hopping layer. Carrying the merged pink
        # layer here rather than the interaction is what buys the saving: a hopping
        # layer costs far more than a batched interaction layer, so the body holds two
        # hopping layers instead of three.
        hop_a = self._hop_layer(
            section_a,
            num_sites=num_sites,
            hopping=hopping,
            time=time,
        )
        body: list[ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm | ConjugatedExponentiatedPauliTerm] = []
        body.extend(open_interaction)
        if hop_b is not None:
            body.append(hop_b)
        body.extend(close_interaction)
        body.extend(identity_phase)
        if hop_a is not None:
            body.append(hop_a)
        return body

    def _interaction_strength(self, hamiltonian: Hamiltonian, num_sites: int) -> float:
        r"""Return the onsite interaction magnitude :math:`|U|` used by the error bound.

        Args:
            hamiltonian: The lattice Hamiltonian to inspect.
            num_sites: Number of lattice sites.

        Returns:
            float: The uniform interaction magnitude, or ``0.0`` when there is none.

        """
        return abs(self._uniform_interaction(hamiltonian, num_sites, atol=1e-12))

    @staticmethod
    def _lattice_one_body(hamiltonian: Hamiltonian, num_sites: int) -> scipy.sparse.csr_matrix:
        """Return the one-body integrals of a lattice Hamiltonian as a CSR matrix.

        Args:
            hamiltonian: The lattice Hamiltonian.
            num_sites: Expected number of lattice sites.

        Returns:
            scipy.sparse.csr_matrix: The one-body integrals, shape ``(num_sites, num_sites)``.

        Raises:
            ValueError: If the Hamiltonian is spin-unrestricted, carries no one-body
                integrals, or its size disagrees with the declared lattice.

        """
        if not hamiltonian.has_one_body_integrals():
            raise ValueError("The lattice Hamiltonian carries no one-body integrals; there is no hopping to tile.")
        if not hamiltonian.is_restricted():
            raise ValueError(
                "PlaquetteTrotter applies the same tiling to both spin sectors, so it requires a "
                "spin-restricted lattice Hamiltonian."
            )

        alpha, _ = hamiltonian.get_one_body_integrals()
        one_body = scipy.sparse.csr_matrix(np.asarray(alpha))
        if one_body.shape != (num_sites, num_sites):
            raise ValueError(
                f"The declared lattice has {num_sites} sites, but the Hamiltonian's one-body "
                f"integrals are {one_body.shape[0]}x{one_body.shape[1]}."
            )
        return one_body

    @staticmethod
    def _uniform_interaction(
        hamiltonian: Hamiltonian,
        num_sites: int,
        *,
        atol: float,
        bonds: set[frozenset[int]] | None = None,
    ) -> float:
        r"""Return the uniform onsite interaction :math:`U` of a lattice Hamiltonian.

        Intersite two-body integrals are detected on the bonded pairs, which is where a
        density-density potential such as the PPP :math:`V_{ij}` always shows up first.

        Args:
            hamiltonian: The lattice Hamiltonian.
            num_sites: Number of lattice sites.
            atol: Threshold below which interaction values are treated as zero.
            bonds: Bonded site pairs to screen for intersite terms, or ``None`` to skip.

        Returns:
            float: The uniform onsite interaction, or ``0.0`` when there are no two-body integrals.

        Raises:
            ValueError: If the two-body integrals are not purely onsite, or the onsite
                interaction is not uniform across sites.

        """
        if not hamiltonian.has_two_body_integrals():
            return 0.0

        for bond in bonds or ():
            site_a, site_b = sorted(bond)
            if abs(hamiltonian.get_two_body_element(site_a, site_a, site_b, site_b)) > atol:
                raise ValueError(
                    f"PlaquetteTrotter models a Hubbard onsite interaction, but the Hamiltonian "
                    f"carries an off-site two-body integral at ({site_a}, {site_a}, {site_b}, {site_b}). "
                    "Intersite terms such as the PPP potential cannot be folded into the interaction layer."
                )

        values = {round(hamiltonian.get_two_body_element(site, site, site, site), 12) for site in range(num_sites)}
        if len(values) > 1:
            raise ValueError(
                f"PlaquetteTrotter requires a uniform onsite interaction, but found "
                f"{len(values)} distinct values: {sorted(values)}."
            )
        return next(iter(values)) if values else 0.0

    def _split_terms(self, hamiltonian: Hamiltonian, num_sites: int, atol: float):
        r"""Read the plaquette inputs straight from a lattice Hamiltonian's integrals.

        The Jordan-Wigner image of the model is constructed analytically rather than by
        mapping the operator. With spin-blocked modes (spin-up ``0..n-1``, spin-down
        ``n..2n-1``) and :math:`n_p = (I - Z_p)/2`, the onsite part of site :math:`i`
        with energy :math:`e_i` and interaction :math:`U_i` is

        .. math::
            e_i (n_{i\uparrow} + n_{i\downarrow}) + U_i n_{i\uparrow} n_{i\downarrow}
            = \Bigl(e_i + \tfrac{U_i}{4}\Bigr) I
            - \Bigl(\tfrac{e_i}{2} + \tfrac{U_i}{4}\Bigr) (Z_i + Z_{i+n})
            + \tfrac{U_i}{4} Z_i Z_{i+n},

        while an off-diagonal integral :math:`h_{ij}` becomes a Jordan-Wigner
        :math:`XZ\cdots ZX` / :math:`YZ\cdots ZY` pair of weight :math:`h_{ij}/2`, i.e.
        a hopping amplitude :math:`t = -h_{ij}`.

        Args:
            hamiltonian: The lattice Hamiltonian to read.
            num_sites: Number of lattice sites.
            atol: Threshold below which coefficients are dropped.

        Returns:
            ``(hopping, diagonal, bonds)`` where *hopping* is the amplitude ``t``,
            *diagonal* holds the :math:`H_I` terms and scalar offset as
            :class:`ExponentiatedPauliTerm` scaled to unit time, and *bonds* is the
            set of site pairs the hopping connects.

        Raises:
            ValueError: If the Hamiltonian carries no hopping or its hopping is not uniform.

        """
        one_body = self._lattice_one_body(hamiltonian, num_sites)

        offdiagonal = scipy.sparse.triu(one_body, k=1, format="coo")
        bonds: set[frozenset[int]] = set()
        amplitudes: set[float] = set()
        for row, col, value in zip(offdiagonal.row, offdiagonal.col, offdiagonal.data, strict=True):
            if abs(value) <= atol:
                continue
            bonds.add(frozenset((int(row), int(col))))
            amplitudes.add(round(float(value), 12))

        if not bonds:
            raise ValueError("The Hamiltonian carries no hopping terms; nothing to tile into plaquettes.")
        if len(amplitudes) > 1:
            raise ValueError(
                f"PlaquetteTrotter requires a uniform hopping amplitude, but found "
                f"{len(amplitudes)} distinct one-body couplings: {sorted(amplitudes)}."
            )

        hopping = -next(iter(amplitudes))
        interaction = self._uniform_interaction(hamiltonian, num_sites, atol=atol, bonds=bonds)

        diagonal: list[ExponentiatedPauliTerm] = []
        constant = hamiltonian.get_core_energy()
        onsite_energies = one_body.diagonal()
        single_z = -0.5 * onsite_energies - 0.25 * interaction
        pair_z = 0.25 * interaction
        constant += float(onsite_energies.sum()) + 0.25 * interaction * num_sites
        for site in range(num_sites):
            weight = float(single_z[site])
            if abs(weight) > atol:
                diagonal.append(ExponentiatedPauliTerm(pauli_term={site: "Z"}, angle=weight))
                diagonal.append(ExponentiatedPauliTerm(pauli_term={site + num_sites: "Z"}, angle=weight))
            if abs(pair_z) > atol:
                diagonal.append(ExponentiatedPauliTerm(pauli_term={site: "Z", site + num_sites: "Z"}, angle=pair_z))
        if abs(constant) > atol:
            diagonal.append(ExponentiatedPauliTerm(pauli_term={}, angle=constant))

        Logger.debug(
            f"PlaquetteTrotter: lattice hopping t={hopping}, U={interaction}, "
            f"{len(bonds)} bonds per spin, {len(diagonal)} diagonal terms."
        )
        return hopping, diagonal, bonds

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


class PlaquetteTrotterSettings(TrotterSettings):
    """Settings for the plaquette Trotter builder."""

    def __init__(self):
        """Initialize the settings, adding the lattice shape to the Trotter defaults."""
        super().__init__()
        self._set_default("lattice_width", "int", 0, "Number of lattice columns. Required.")
        self._set_default("lattice_height", "int", 0, "Number of lattice rows. Required.")
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
