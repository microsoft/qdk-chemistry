"""Matrix Product State (MPS) state preparation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from qdk_chemistry.algorithms.circuit_mapper.unitary_synthesis_mapper import (
    BlockSparseUnitarySynthesisMapper,
    DenseUnitarySynthesisMapper,
    HouseholderUnitarySynthesisMapper,
    UnitarySynthesisMapper,
)
from qdk_chemistry.data import (
    AlgorithmRef,
    Configuration,
    MPSContainer,
    MPSSite,
    Settings,
    UnitaryMatrixContainer,
    UnitaryRepresentation,
    Wavefunction,
)
from qdk_chemistry.data.circuit import Circuit, CircuitMetadata, QsharpFactoryData
from qdk_chemistry.data.symmetry import AxisName, SymmetryLabel
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation

__all__: list[str] = ["MatrixProductStatePreparation"]

_DESCRIPTION = "MPS state preparation"

# Supported physical bases by the number of physical states per site. A site with two states
# is one spinless mode on one qubit; a site with four states is one spatial orbital on two
# qubits, its alpha and beta modes.
_PHYSICAL_BASIS_LABELS = {2: ("0", "1"), 4: ("0", "u", "d", "2")}
_PHYSICAL_BASES = {
    2: [Configuration.from_bitstring(state) for state in _PHYSICAL_BASIS_LABELS[2]],
    4: [Configuration.from_spin_half_string(state) for state in _PHYSICAL_BASIS_LABELS[4]],
}


# Spin-adapted sites use one reduced state per multiplet, (N, 2S) = (0, 0), (1, 1), (2, 0).
_SPIN_ADAPTED_BASIS = [Configuration.from_spin_half_string(state) for state in ("0", "u", "2")]

# Fusion channels j of Def. C.1 of arXiv:2611.19728: left sector offset (Δn, Δ2S), physical sector
# (n_P, 2S_P) and channel code Γ = k + 2l of the (alpha, beta) qubits before the Clebsch-Gordan stage.
_CHANNELS = (
    ((0, 0), (0, 0), 0b00),
    ((-2, 0), (2, 0), 0b11),
    ((-1, -1), (1, 1), 0b01),
    ((-1, 1), (1, 1), 0b10),
)

# Q# struct prefix of each site synthesis method.
_QSHARP_NAMES = {"dense": "MPSSequential", "block_sparse": "MPSSparse", "householder": "MPSHouseholder"}

_ISOMETRY_TOLERANCE = 1e-8


class MatrixProductStatePreparationSettings(Settings):
    """Settings for matrix product state preparation."""

    def __init__(self):
        """Initialize the MatrixProductStatePreparationSettings."""
        super().__init__()
        self._set_default(
            "rotation_bit_precision",
            "int",
            10,
            "Size of the phase gradient register, which sets the precision of every rotation "
            "angle. The upper bound of 30 is a sanity limit as 2^-30 is far below chemical "
            "accuracy.",
            (1, 30),
        )
        self._set_default(
            "unitary_synthesis",
            "algorithm_ref",
            AlgorithmRef("circuit_mapper", "unitary_synthesis_dense"),
            "unitary_synthesis_dense, unitary_synthesis_block_sparse or unitary_synthesis_householder circuit "
            "mapper that synthesizes every site isometry; spin-adapted MPS require unitary_synthesis_householder. "
            "Its num_input_qubits is set to the bond register width, its "
            "rotation_bit_precision to this algorithm's, and the dense mapper's omit_right_factor to True. "
            "Its allocate_phase_gradient is unused, as all sites share this algorithm's phase gradient.",
        )
        self._set_default(
            "allocate_phase_gradient",
            "bool",
            True,
            "Whether to allocate and initialize the phase gradient register internally.",
        )


class MatrixProductStatePreparation(StatePreparation):
    r"""Matrix Product State (MPS) state preparation.

    Implements the MPS state preparation algorithm based on :cite:`Berry2025` and :cite:`Rupprecht2026`,
    which prepares the state one site at a time using an ancilla register that stores the virtual bond.
    Site zero is loaded as an initial state, and every later site is applied as an isometry from its
    left bond, held in the bond register, to its physical state and right bond. The nested
    ``unitary_synthesis`` circuit mapper synthesizes each isometry.

    Sites with the ``('0', 'u', 'd', '2')`` physical basis are spatial
    orbitals in the blocked Jordan-Wigner layout. Sites
    with the ``('0', '1')`` physical basis are spinless modes in the
    Jordan-Wigner layout.

    A charge- and spin-adapted (U(1) x SU(2)) MPS is loaded from its reduced tensors following
    :cite:`Hennefarth2026`, with the ``unitary_synthesis_householder`` mapper. Its sector labels carry a
    particle-number and a total-spin axis, every site uses the reduced basis ``('0', 'u', '2')`` of
    sectors :math:`(N, 2S) = (0, 0), (1, 1), (2, 0)`, and the tensors are left-canonical with the
    orthogonality center on the last site, in the Clebsch-Gordan convention
    :math:`T = A \otimes \langle S_L m_L; S_P m_P | S_R m_R \rangle` with ``'2'`` created as
    :math:`a^\dagger_\alpha a^\dagger_\beta`, which block2 SU(2) MPS follow. The sites are loaded from the last
    to the first. A bond register holds the multiplicity, :math:`N`, :math:`2S` and :math:`2M_S` of the bond.
    Each site applies one Householder reflection per reduced column, covering every :math:`M_S` through a
    comparator; the column is loaded over a contiguized index that comparators then decode into the left
    sector and physical channel (App. A.2), and a data-independent Clebsch-Gordan rotation (App. C) restores
    the spin projections, so the unfolded tensors are never loaded. The spin projection is
    :math:`M_S = (N_\alpha - N_\beta) / 2` from the active particle counts, or :math:`M_S = S` without them.
    """

    def __init__(self):
        """Initialize the matrix product state preparation algorithm."""
        super().__init__()
        self._settings = MatrixProductStatePreparationSettings()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"matrix_product_state"``

        """
        return "matrix_product_state"

    def _generate_preparation_data(self, mps: MPSContainer) -> tuple[str, dict[str, Any]]:
        """Synthesize the site isometries with the nested ``unitary_synthesis`` circuit mapper.

        Args:
            mps: Real MPS container with the same supported physical basis on every site.
                Its validated site-to-orbital order is preserved.

        Returns:
            The site synthesis method, ``"dense"``, ``"block_sparse"`` or ``"householder"``, and the fields
            of the Q# ``MPSSequentialParams``, ``MPSSparseParams`` or ``MPSHouseholderParams`` struct.

        Raises:
            TypeError: If the ``unitary_synthesis`` setting does not refer to a unitary synthesis mapper.
            ValueError: If the physical basis is unsupported, the MPS is complex,
                the initial state has zero norm, or a synthesized site is not isometric.

        """
        if mps.is_complex():
            raise ValueError(f"{_DESCRIPTION} currently supports only real-valued MPS tensors.")
        mps_sites = mps.sites()
        dimensions = {site.physical_dimension() for site in mps_sites}
        if len(dimensions) > 1:
            raise ValueError("MPS state preparation requires the same physical dimension on every site.")
        (dimension,) = dimensions
        if dimension not in _PHYSICAL_BASES:
            raise ValueError("MPS state preparation requires two or four physical states per site.")
        if any(site.physical_basis() != _PHYSICAL_BASES[dimension] for site in mps_sites):
            labels = ", ".join(f"'{label}'" for label in _PHYSICAL_BASIS_LABELS[dimension])
            raise ValueError(f"MPS state preparation requires physical basis ordering ({labels}).")
        # The bond register holds the largest bond, and at least one qubit.
        ancilla_bits = max(1, (mps.max_bond_dimension() - 1).bit_length())
        ancilla_dim = 1 << ancilla_bits
        mapper, method = self._create_site_mapper(ancilla_bits)

        # Sites are synthesized from the last one so that each dense right factor V, which the
        # circuit of its site leaves out, can be absorbed into the right bond of the site before.
        syntheses: list[dict[str, Any]] = []
        right_factor: np.ndarray | None = None
        for index in range(len(mps_sites) - 1, 0, -1):
            if method == "block_sparse":
                matrix: np.ndarray | sparse.csc_array = self._sparse_site_isometry(mps_sites[index], ancilla_dim)
            else:
                isometry = self._site_isometry(mps_sites[index], ancilla_dim)
                if right_factor is not None:
                    bond = right_factor.shape[0]
                    isometry[:, :bond] = right_factor @ isometry[:, :bond]
                matrix = isometry.reshape(-1, isometry.shape[2])
            try:
                container = UnitaryMatrixContainer(matrix)
            except ValueError as error:
                raise ValueError(f"{_DESCRIPTION} requires isometric sites; site {index} is not: {error}") from error
            unitary = UnitaryRepresentation(container)
            syntheses.append(mapper.synthesize(unitary))
            right_factor = mapper.right_factor(unitary) if isinstance(mapper, DenseUnitarySynthesisMapper) else None
        syntheses.reverse()

        # With a left bond of dimension one the packed matrix is (physical, right).
        first_site = np.asarray(mps_sites[0].to_dense())
        if right_factor is not None:
            first_site = first_site @ right_factor.T
        # Site zero's right bond only needs the low qubits of the bond register, so the initial state
        # is indexed by physical * initial_bond_dim + bond.
        initial_bond_dim = 1 << (first_site.shape[1] - 1).bit_length()
        padded = np.zeros((first_site.shape[0], initial_bond_dim))
        padded[:, : first_site.shape[1]] = first_site
        # Dividing by the largest amplitude first keeps the check and the norm independent of the overall scale.
        vector = padded.reshape(-1)
        scale = np.max(np.abs(vector))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("MPS initial state must contain finite amplitudes with nonzero norm.")
        vector = vector / scale

        # Site zero is the initial state, and each following site is a synthesized isometry.
        return method, {
            "initialStateVec": (vector / np.linalg.norm(vector)).tolist(),
            "numSites": mps.num_sites(),
            "numQubitsPerSite": (dimension - 1).bit_length(),
            "siteToOrbitalOrder": mps.site_to_orbital_order(),
            "rotationBitPrecision": self._settings.get("rotation_bit_precision"),
            "numAncillaQubits": ancilla_bits,
            "siteDecompositions": syntheses,
        }

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Return a circuit to prepare an MPS state.

        Args:
            wavefunction: The wavefunction to prepare.

        Returns:
            A Circuit object implementing the MPS state preparation.

        Raises:
            TypeError: If wavefunction is not an MPSContainer instance, or the ``unitary_synthesis``
                setting does not refer to a unitary synthesis mapper supported for the MPS.
            ValueError: If the MPS is complex, not right-canonical with orthogonality center zero (or,
                when spin-adapted, left-canonical with center on the last site), or does not use one of
                the ``('0', '1')``, ``('0', 'u', 'd', '2')`` and spin-adapted ``('0', 'u', '2')`` physical
                bases on every site.

        """
        container = wavefunction.get_container()
        if not isinstance(container, MPSContainer):
            raise TypeError(f"MatrixProductStatePreparation requires an MPSContainer, got {type(container)}.")
        if any(label.has(AxisName.TotalSpin) for site in container.sites() for label in site.physical_sector_order()):
            two_projection = None
            if container.has_active_num_particles():
                num_alpha, num_beta = wavefunction.get_active_num_electrons()
                two_projection = num_alpha - num_beta
            name, params = "MPSSpinAdapted", self._generate_spin_adapted_data(container, two_projection)
            num_state_qubits = 2 * params["numSites"]
        else:
            if container.orthogonality_center() != 0:
                raise ValueError(f"{_DESCRIPTION} requires a right-canonical MPS with center zero.")
            method, params = self._generate_preparation_data(container)
            name = _QSHARP_NAMES[method]
            num_state_qubits = params["numQubitsPerSite"] * params["numSites"]
        # The MPS circuits are Adaptive-only; the default shared context targets Adaptive_RIF.
        module = QSHARP_UTILS.MPSStatePrep
        qsharp_params = getattr(module, f"{name}Params")(**params)
        if self._settings.get("allocate_phase_gradient"):
            qsharp_op = getattr(module, f"Make{name}Op")(qsharp_params)
            num_gradient_ancillas = 0
        else:
            # The caller owns the trailing gradient register, must leave the gradient in it, and
            # must exclude it from any reflection about |0>.
            qsharp_op = getattr(module, f"Make{name}OpWithPhaseGradient")(qsharp_params)
            num_gradient_ancillas = params["rotationBitPrecision"]
        # An exported circuit has no caller to own the gradient, so it always allocates its own.
        program = getattr(module, f"Make{name}Circuit")
        return Circuit(
            qsharp_factory=QsharpFactoryData(program=program, parameter={"params": params}),
            qsharp_op=qsharp_op,
            encoding="jordan-wigner",
            num_qubits=num_state_qubits + num_gradient_ancillas,
            metadata=CircuitMetadata(num_phase_gradient_ancillas=num_gradient_ancillas),
        )

    def _generate_spin_adapted_data(self, mps: MPSContainer, two_projection: int | None) -> dict[str, Any]:
        """Return the fields of the Q# ``MPSSpinAdaptedParams`` struct of a spin-adapted MPS.

        Each reduced column is contiguized over its fusion channels in ascending multiplicity (App. A.2 of
        :cite:`Hennefarth2026`) and stored as the state-preparation data of the Householder mapper.

        Args:
            mps: Real, left-canonical MPS of reduced tensors with (particle number, total spin) labels.
            two_projection: Twice the spin projection M_S to prepare, or None for M_S = S.

        Returns:
            The fields of the Q# ``MPSSpinAdaptedParams`` struct.

        Raises:
            TypeError: If the ``unitary_synthesis`` setting does not refer to the Householder mapper.
            ValueError: If the MPS is complex, not spin-adapted or not left-canonical, or the spin
                projection is incompatible with the total spin.

        """
        description = f"Spin-adapted {_DESCRIPTION}"
        if mps.is_complex():
            raise ValueError(f"{description} supports only real-valued MPS tensors.")
        num_sites = mps.num_sites()
        if mps.orthogonality_center() != num_sites - 1:
            raise ValueError(f"{description} requires a left-canonical MPS with its center on the last site.")

        def quantum_numbers(label: SymmetryLabel) -> tuple[int, int]:
            if not (label.has(AxisName.ParticleNumber) and label.has(AxisName.TotalSpin)) or label.has(AxisName.Spin):
                raise ValueError(f"{description} requires (particle number, total spin) sector labels.")
            return label.get(AxisName.ParticleNumber).value(), label.get(AxisName.TotalSpin).value()

        sites, bond_numbers, max_rows = [], set(), 1
        for index, site in enumerate(mps.sites()):
            if site.physical_basis() != _SPIN_ADAPTED_BASIS:
                raise ValueError(f"{description} requires the reduced physical basis ('0', 'u', '2').")
            left_extents, _, right_extents = (
                {quantum_numbers(label): extent for label, extent in slot.items()} for slot in site.tensor().extents()
            )
            bond_numbers |= left_extents.keys() | right_extents.keys()
            blocks = {
                tuple(quantum_numbers(label) for label in labels): np.asarray(block)
                for labels, block in site.tensor().blocks()
            }
            sectors = []
            for right_label in site.right_sector_order():
                right = quantum_numbers(right_label)
                cases = []
                for channel, ((dn, ds), physical, code) in enumerate(_CHANNELS):
                    left = (right[0] + dn, right[1] + ds)
                    block = blocks.get((left, physical, right))
                    if block is not None:
                        cases.append((left_extents[left], channel, left, code, block))
                cases.sort(key=lambda case: case[:2])
                matrix = np.concatenate([case[4] for case in cases]) if cases else np.zeros((0, right_extents[right]))
                gram = matrix.T @ matrix
                if not np.allclose(gram, np.eye(len(gram)), atol=_ISOMETRY_TOLERANCE):
                    raise ValueError(
                        f"{description} requires left-isometric reduced blocks (Eq. 21); site {index} "
                        f"sector {right} deviates by {np.abs(gram - np.eye(len(gram))).max():.2e}."
                    )
                max_rows = max(max_rows, len(matrix))
                columns = [HouseholderUnitarySynthesisMapper.column_preparation(column) for column in matrix.T]
                sectors.append(
                    {
                        "numParticles": right[0],
                        "twoSpin": right[1],
                        "caseNumParticles": [case[2][0] for case in cases],
                        "caseTwoSpin": [case[2][1] for case in cases],
                        "caseCodes": [case[3] for case in cases],
                        "caseMultiplicities": [case[0] for case in cases],
                        "columnAngles": [angles for angles, _ in columns],
                        "columnSigns": [signs for _, signs in columns],
                    }
                )
            sites.append({"sectors": sectors})

        left_boundary = list(mps.sites()[0].tensor().extents()[0])
        right_boundary = list(mps.sites()[-1].tensor().extents()[2])
        if len(left_boundary) != 1 or quantum_numbers(left_boundary[0]) != (0, 0):
            raise ValueError(f"{description} requires the vacuum (0, 0) as the left boundary bond.")
        if len(right_boundary) != 1:
            raise ValueError(f"{description} requires a single (N, 2S) sector as the right boundary bond.")
        num_particles, two_spin = quantum_numbers(right_boundary[0])
        if two_projection is None:
            two_projection = two_spin
        if abs(two_projection) > two_spin or (two_spin - two_projection) % 2:
            raise ValueError(f"{description} cannot prepare 2M_S = {two_projection} of a state with 2S = {two_spin}.")

        max_two_spin = max(two_s for _, two_s in bond_numbers)
        register_widths = [
            max(1, (max_rows - 1).bit_length()),
            max(1, max(n for n, _ in bond_numbers).bit_length()),
            max(1, max_two_spin.bit_length()),
            # Two's complement of 2m with |2m| <= 2S_max, so that a shift by 2S_R never wraps.
            max_two_spin.bit_length() + 1,
        ]
        mapper, method = self._create_site_mapper(register_widths[0])
        if method != "householder":
            raise TypeError(
                f"{description} requires the unitary_synthesis_householder circuit mapper, got '{mapper.name()}'."
            )
        return {
            "numSites": num_sites,
            "siteToOrbitalOrder": list(mps.site_to_orbital_order()),
            "rotationBitPrecision": self._settings.get("rotation_bit_precision"),
            "maxSwapBits": mapper.settings().get("max_swap_address_bits"),
            "registerWidths": register_widths,
            "target": [num_particles, two_spin, two_projection],
            "sites": sites,
        }

    def _create_site_mapper(self, ancilla_bits: int) -> tuple[UnitarySynthesisMapper, str]:
        """Return the nested site mapper, configured for the bond register, and its method.

        Raises:
            TypeError: If the ``unitary_synthesis`` setting does not refer to a unitary synthesis mapper.

        """
        mapper = self._create_nested("unitary_synthesis")
        if isinstance(mapper, DenseUnitarySynthesisMapper):
            method = "dense"
            mapper.settings().update("omit_right_factor", True)
        elif isinstance(mapper, BlockSparseUnitarySynthesisMapper):
            method = "block_sparse"
        elif isinstance(mapper, HouseholderUnitarySynthesisMapper):
            method = "householder"
        else:
            raise TypeError(
                f"{_DESCRIPTION} requires the unitary_synthesis_dense, unitary_synthesis_block_sparse or "
                f"unitary_synthesis_householder circuit mapper, got {mapper.type_name()} '{mapper.name()}'."
            )
        # The bond register is the input register; the site qubits are the added qubits.
        mapper.settings().update("num_input_qubits", ancilla_bits)
        mapper.settings().update("rotation_bit_precision", self._settings.get("rotation_bit_precision"))
        return mapper, method

    @staticmethod
    def _site_isometry(site: MPSSite, ancilla_dim: int) -> np.ndarray:
        """Return ``isometry[p, b, a]`` = A^p_{ab}, zero-padded to ``ancilla_dim`` right bond states.

        Reshaped to ``(physical * ancilla_dim, left)``, row ``p * ancilla_dim + b`` is site
        qubit state ``p`` and bond register state ``b``, and column ``a`` is the left bond state.
        """
        left, physical, right = site.shape()
        tensor = np.asarray(site.to_dense()).reshape(left, physical, right)
        isometry = np.zeros((physical, ancilla_dim, left))
        isometry[:, :right] = tensor.transpose(1, 2, 0)
        return isometry

    @staticmethod
    def _sparse_site_isometry(site: MPSSite, ancilla_dim: int) -> sparse.csc_array:
        """Return :meth:`_site_isometry` reshaped to ``(physical * ancilla_dim, left)`` as a sparse matrix.

        The symmetry blocks of the site are scattered straight into the matrix, so neither the dense
        site nor the padded isometry is formed.
        """
        tensor = site.tensor()
        extents = tensor.extents()
        offsets = []
        for order, slot_extents in zip(site.sector_orders(), extents, strict=True):
            starts = np.cumsum([0] + [slot_extents[label] for label in order[:-1]])
            offsets.append(dict(zip(order, starts.tolist(), strict=True)))
        rows, columns, values = [np.empty(0, dtype=int)], [np.empty(0, dtype=int)], [np.empty(0)]
        for labels, block in tensor.blocks():
            shape = tuple(extents[slot][label] for slot, label in enumerate(labels))
            values_3d = np.asarray(block).reshape(shape)
            left, physical, right = np.nonzero(values_3d)
            left_label, physical_label, right_label = labels
            rows.append((offsets[1][physical_label] + physical) * ancilla_dim + offsets[2][right_label] + right)
            columns.append(offsets[0][left_label] + left)
            values.append(values_3d[left, physical, right])
        return sparse.csc_array(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(site.physical_dimension() * ancilla_dim, site.left_bond_dimension()),
        )
