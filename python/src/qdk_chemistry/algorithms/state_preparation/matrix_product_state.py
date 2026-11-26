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
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation

__all__: list[str] = ["MatrixProductStatePreparation"]

_DESCRIPTION = "MPS state preparation"

# Supported physical bases by the number of physical states per site. A site with two states
# is one spinless mode on one qubit; a site with four states is one spatial orbital on two
# qubits, its alpha and beta modes. Each state is compared by its encoding and first mode,
# since MPS sites may pad their states with empty modes.
_PHYSICAL_BASIS_LABELS = {2: ("0", "1"), 4: ("0", "u", "d", "2")}
_PHYSICAL_BASES = {
    2: [(1, Configuration.from_bitstring(state).get_mode_state(0)) for state in _PHYSICAL_BASIS_LABELS[2]],
    4: [(2, Configuration.from_spin_half_string(state).get_mode_state(0)) for state in _PHYSICAL_BASIS_LABELS[4]],
}


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
            "unitary_synthesis_dense or unitary_synthesis_block_sparse circuit mapper that synthesizes "
            "every site isometry. Its num_input_qubits is set to the bond register width, its "
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
            The site synthesis method, ``"dense"`` or ``"block_sparse"``, and the fields of the Q#
            ``MPSSequentialParams`` or ``MPSSparseParams`` struct.

        Raises:
            TypeError: If the ``unitary_synthesis`` setting does not refer to a dense or block-sparse
                unitary synthesis mapper.
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
        if any(
            [(state.bits_per_mode(), state.get_mode_state(0)) for state in site.physical_basis()]
            != _PHYSICAL_BASES[dimension]
            for site in mps_sites
        ):
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
            synthesis = mapper.synthesize(UnitaryRepresentation(container))
            right_factor = synthesis.pop("rightFactor", None)
            syntheses.append(synthesis)
        syntheses.reverse()

        # With a left bond of dimension one the packed matrix is (physical, right). Dividing by the largest
        # amplitude before absorbing V keeps the check, the product and the norm independent of the overall scale.
        first_site = np.asarray(mps_sites[0].to_dense())
        scale = np.max(np.abs(first_site))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("MPS initial state must contain finite amplitudes with nonzero norm.")
        first_site = first_site / scale
        if right_factor is not None:
            first_site = first_site @ right_factor.T
        # Site zero's right bond only needs the low qubits of the bond register, so the initial state
        # is indexed by physical * initial_bond_dim + bond.
        initial_bond_dim = 1 << (first_site.shape[1] - 1).bit_length()
        padded = np.zeros((first_site.shape[0], initial_bond_dim))
        padded[:, : first_site.shape[1]] = first_site
        vector = padded.reshape(-1)

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
                setting does not refer to a dense or block-sparse unitary synthesis mapper.
            ValueError: If the MPS is complex, not right-canonical with orthogonality
                center zero, or does not use one of the ``('0', '1')`` and ``('0', 'u', 'd', '2')``
                physical bases on every site.

        """
        container = wavefunction.get_container()
        if not isinstance(container, MPSContainer):
            raise TypeError(f"MatrixProductStatePreparation requires an MPSContainer, got {type(container)}.")
        if container.orthogonality_center() != 0:
            raise ValueError(f"{_DESCRIPTION} requires a right-canonical MPS with center zero.")

        method, params = self._generate_preparation_data(container)
        # The MPS circuits are Adaptive-only; the default shared context targets Adaptive_RIF.
        name = "MPSSparse" if method == "block_sparse" else "MPSSequential"
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
            num_qubits=params["numQubitsPerSite"] * params["numSites"] + num_gradient_ancillas,
            metadata=CircuitMetadata(num_phase_gradient_ancillas=num_gradient_ancillas),
        )

    def _create_site_mapper(self, ancilla_bits: int) -> tuple[UnitarySynthesisMapper, str]:
        """Return the nested site mapper, configured for the bond register, and its method.

        Raises:
            TypeError: If the ``unitary_synthesis`` setting does not refer to a dense or block-sparse
                unitary synthesis mapper.

        """
        mapper = self._create_nested("unitary_synthesis")
        if isinstance(mapper, DenseUnitarySynthesisMapper):
            method = "dense"
            mapper.settings().update("omit_right_factor", True)
        elif isinstance(mapper, BlockSparseUnitarySynthesisMapper):
            method = "block_sparse"
        else:
            raise TypeError(
                f"{_DESCRIPTION} requires the unitary_synthesis_dense or unitary_synthesis_block_sparse "
                f"circuit mapper, got {mapper.type_name()} '{mapper.name()}'."
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
