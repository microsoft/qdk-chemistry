"""OpenFermion-based qubit mappers to map electronic structure Hamiltonians to qubit Hamiltonians.

This module provides an OpenFermionQubitMapper class to convert Hamiltonians to QubitHamiltonians
using different mapping strategies ("jordan-wigner", "bravyi-kitaev",
"symmetry-conserving-bravyi-kitaev", "bravyi-kitaev-fast", and "bravyi-kitaev-tree").
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import openfermion as of

from qdk_chemistry.algorithms.qubit_mapper import QubitMapper, QubitMapperSettings
from qdk_chemistry.data import Hamiltonian, QubitHamiltonian
from qdk_chemistry.plugins.openfermion.conversion import (
    hamiltonian_to_fermion_operator,
    hamiltonian_to_interaction_operator,
    qubit_operator_to_qubit_hamiltonian,
)
from qdk_chemistry.utils import Logger

__all__ = ["OpenFermionQubitMapper", "OpenFermionQubitMapperSettings"]

_VALID_ENCODINGS = [
    "jordan-wigner",
    "bravyi-kitaev",
    "symmetry-conserving-bravyi-kitaev",
    "bravyi-kitaev-fast",
    "bravyi-kitaev-tree",
]


class OpenFermionQubitMapperSettings(QubitMapperSettings):
    """Settings configuration for an OpenFermionQubitMapper.

    Inherits ``encoding`` from :class:`~qdk_chemistry.algorithms.qubit_mapper.QubitMapperSettings`.

    Additional settings:
        n_active_electrons (integer, default=0): Required for ``symmetry-conserving-bravyi-kitaev`` (0 = auto-detect).

    """

    def __init__(self):
        """Initialize OpenFermionQubitMapperSettings."""
        Logger.trace_entering()
        super().__init__(valid_encodings=_VALID_ENCODINGS)
        self._set_default(
            "n_active_electrons",
            "int",
            0,
            "Number of active electrons (required for symmetry-conserving-bravyi-kitaev, 0 = auto-detect)",
        )


class OpenFermionQubitMapper(QubitMapper):
    """Class to map an electronic structure Hamiltonian to a QubitHamiltonian using an OpenFermion mapper."""

    def __init__(self, encoding: str = "jordan-wigner"):
        """Initialize OpenFermionQubitMapper with a specific mapping strategy.

        Args:
            encoding (str): Qubit mapping strategy to use ("jordan-wigner", "bravyi-kitaev",
                "symmetry-conserving-bravyi-kitaev", "bravyi-kitaev-fast", or
                "bravyi-kitaev-tree"). Default: "jordan-wigner".

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = OpenFermionQubitMapperSettings()
        self._settings.set("encoding", encoding)

    def _run_impl(self, hamiltonian: Hamiltonian) -> QubitHamiltonian:
        """Construct a QubitHamiltonian from a Hamiltonian using the selected mapping strategy.

        Args:
            hamiltonian (Hamiltonian): The fermionic Hamiltonian.

        Returns:
            QubitHamiltonian: An instance of the QubitHamiltonian.

        """
        Logger.trace_entering()
        encoding = self._settings.get("encoding")
        if encoding not in _VALID_ENCODINGS:
            raise ValueError(
                f"Encoding '{encoding}' is unknown for OpenFermionQubitMapper.\nPlease use one of: {_VALID_ENCODINGS}"
            )

        Logger.debug(f"Mapping Hamiltonian with OpenFermion encoding: {encoding}")

        if encoding == "bravyi-kitaev-fast":
            qubit_op = self._map_bksf(hamiltonian)
        elif encoding == "symmetry-conserving-bravyi-kitaev":
            qubit_op = self._map_scbk(hamiltonian)
        else:
            qubit_op = self._map_standard(hamiltonian, encoding)

        qubit_op.compress()

        # OpenFermion folds core_energy into the identity Pauli term.
        # QDK convention stores core energy separately, so subtract it.
        core_energy = hamiltonian.get_core_energy()
        if abs(core_energy) > np.finfo(np.float64).eps:
            qubit_op -= core_energy * of.QubitOperator(())
            qubit_op.compress()

        return qubit_operator_to_qubit_hamiltonian(qubit_op, encoding=encoding)

    def _map_standard(self, hamiltonian: Hamiltonian, encoding: str) -> "of.QubitOperator":
        """Apply a standard fermion-to-qubit transform (JW, BK, or BK-tree).

        Uses blocked spin-orbital ordering (α₀, α₁, …, β₀, β₁, …) so that
        the resulting qubit Hamiltonian matches the QDK native mapper output.

        Args:
            hamiltonian: The fermionic Hamiltonian.
            encoding: One of ``"jordan-wigner"``, ``"bravyi-kitaev"``,
                or ``"bravyi-kitaev-tree"``.

        Returns:
            openfermion.QubitOperator: The mapped qubit operator.

        """
        fermion_op = _build_blocked_fermion_operator(hamiltonian)

        transform_map = {
            "jordan-wigner": of.transforms.jordan_wigner,
            "bravyi-kitaev": of.transforms.bravyi_kitaev,
            "bravyi-kitaev-tree": of.transforms.bravyi_kitaev_tree,
        }
        transform = transform_map[encoding]
        return transform(fermion_op)

    def _map_scbk(self, hamiltonian: Hamiltonian) -> "of.QubitOperator":
        """Apply symmetry-conserving Bravyi-Kitaev transformation.

        This transform reduces the qubit count by 2 by exploiting particle number
        and spin symmetry. It requires the number of active electrons.

        Args:
            hamiltonian: The fermionic Hamiltonian.

        Returns:
            openfermion.QubitOperator: The mapped qubit operator.

        Raises:
            ValueError: If the number of active electrons cannot be determined.

        """
        fermion_op = hamiltonian_to_fermion_operator(hamiltonian)

        n_active_electrons = self._settings.get("n_active_electrons")
        if n_active_electrons <= 0:
            # Try to infer from orbital data
            try:
                orbitals = hamiltonian.get_orbitals()
                alpha_inactive, _ = orbitals.get_inactive_space_indices()
                alpha_active, _ = orbitals.get_active_space_indices()
                # Number of active electrons = total electrons - inactive electrons * 2
                # For restricted: each inactive orbital contributes 2 electrons
                n_active_electrons = 2 * len(alpha_inactive)
                Logger.debug(
                    f"Auto-detected n_active_electrons={n_active_electrons} from orbital data "
                    f"(inactive orbitals: {len(alpha_inactive)})"
                )
            except (AttributeError, RuntimeError, TypeError):
                raise ValueError(
                    "Cannot determine the number of active electrons for symmetry-conserving "
                    "Bravyi-Kitaev. Please set the 'n_active_electrons' setting explicitly."
                ) from None

        # Number of spin-orbitals
        h1_alpha, _ = hamiltonian.get_one_body_integrals()
        n_spinorbitals = 2 * h1_alpha.shape[0]

        Logger.debug(f"SCBK: n_spinorbitals={n_spinorbitals}, n_active_electrons={n_active_electrons}")

        return of.transforms.symmetry_conserving_bravyi_kitaev(
            fermion_op,
            n_spinorbitals,
            n_active_electrons,
        )

    def _map_bksf(self, hamiltonian: Hamiltonian) -> "of.QubitOperator":
        """Apply Bravyi-Kitaev superfast (BKSF) transformation.

        The BKSF transform operates directly on the ``InteractionOperator`` rather
        than on a ``FermionOperator``, and maps edges of an interaction graph to qubits.

        Args:
            hamiltonian: The fermionic Hamiltonian.

        Returns:
            openfermion.QubitOperator: The mapped qubit operator.

        """
        iop = hamiltonian_to_interaction_operator(hamiltonian)
        return of.transforms.bravyi_kitaev_fast(iop)

    def name(self) -> str:
        """Return the algorithm name ``openfermion``."""
        Logger.trace_entering()
        return "openfermion"


def _build_blocked_fermion_operator(hamiltonian: Hamiltonian) -> "of.FermionOperator":
    """Build a FermionOperator using blocked spin-orbital ordering.

    Blocked ordering: [α₀, α₁, …, αₙ₋₁, β₀, β₁, …, βₙ₋₁]
    (QDK native convention)

    This differs from OpenFermion's native interleaved convention
    [α₀, β₀, α₁, β₁, …].  Building the operator in blocked order ensures
    that subsequent fermion-to-qubit transforms (JW, BK, BK-tree) produce
    qubit Hamiltonians directly compatible with the QDK native mapper.

    Args:
        hamiltonian: The QDK/Chemistry Hamiltonian.

    Returns:
        openfermion.FermionOperator in blocked spin-orbital ordering.

    """
    # Start from the correct interleaved InteractionOperator
    iop = hamiltonian_to_interaction_operator(hamiltonian)

    n_so = iop.n_qubits
    n_spatial = n_so // 2

    # Permutation: blocked index j → interleaved index
    #   j < n_spatial  (alpha): 2*j
    #   j >= n_spatial (beta):  2*(j - n_spatial) + 1
    idx = np.array([2 * j if j < n_spatial else 2 * (j - n_spatial) + 1 for j in range(n_so)])

    # Re-index the one- and two-body tensors from interleaved to blocked
    h1_blocked = iop.one_body_tensor[np.ix_(idx, idx)]
    h2_blocked = iop.two_body_tensor[np.ix_(idx, idx, idx, idx)]

    iop_blocked = of.InteractionOperator(iop.constant, h1_blocked, h2_blocked)
    return of.transforms.get_fermion_operator(iop_blocked)
