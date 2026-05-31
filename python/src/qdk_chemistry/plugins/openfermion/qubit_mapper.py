"""OpenFermion-based qubit mappers to map electronic structure Hamiltonians to qubit Hamiltonians.

This module provides an OpenFermionQubitMapper class to convert Hamiltonians to QubitHamiltonians
using different mapping strategies. The encoding is determined by the MajoranaMapping passed
to ``run()``.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import openfermion as of

from qdk_chemistry.algorithms.qubit_mapper import QubitMapper, QubitMapperSettings
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.plugins.openfermion.conversion import (
    hamiltonian_to_interaction_operator,
    qubit_operator_to_qubit_hamiltonian,
)
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from qdk_chemistry.data import Hamiltonian, QubitHamiltonian
    from qdk_chemistry.data.majorana_mapping import MajoranaMapping

__all__ = ["OpenFermionQubitMapper", "OpenFermionQubitMapperSettings"]

_STANDARD_TRANSFORMS: dict[str, Callable[..., of.QubitOperator]] = {
    "jordan-wigner": of.transforms.jordan_wigner,
    "bravyi-kitaev": of.transforms.bravyi_kitaev,
    "bravyi-kitaev-tree": of.transforms.bravyi_kitaev_tree,
}


class OpenFermionQubitMapperSettings(QubitMapperSettings):
    """Settings configuration for an OpenFermionQubitMapper."""

    def __init__(self):
        """Initialize OpenFermionQubitMapperSettings."""
        Logger.trace_entering()
        super().__init__()


class OpenFermionQubitMapper(QubitMapper):
    """Map an electronic structure Hamiltonian to a QubitHamiltonian using OpenFermion.

    This is a **name-dispatched** backend: it reads
    ``mapping.base_encoding`` to select the corresponding OpenFermion
    transform function and **ignores** ``mapping.table`` entirely.  The
    qubit operator is built from scratch using OpenFermion's own
    fermion-to-qubit pipeline.

    .. warning::

       Because this backend dispatches on the encoding *name* rather than
       the Pauli table, it relies on the assumption that the mapping's
       ``base_encoding`` string is consistent with its table contents.
       This invariant is guaranteed for factory-produced mappings
       (``MajoranaMapping.jordan_wigner()``, ``.bravyi_kitaev()``, etc.)
       and is verified by cross-backend eigenvalue tests in the test
       suite.  Manually constructed mappings with mismatched names will
       produce silently incorrect results.

    Tapering-based encodings (e.g. symmetry-conserving Bravyi-Kitaev) are
    supported — each backend handles tapering in its own ``_run_impl()``
    via the ``QubitMapper._taper_result()``
    helper.

    Both restricted (RHF) and unrestricted (UHF) Hamiltonians are supported.
    For unrestricted systems, separate alpha/beta spin channels are handled
    via ``hamiltonian_to_interaction_operator``.

    Supported base encodings:
        - ``"jordan-wigner"`` → ``openfermion.transforms.jordan_wigner``
        - ``"bravyi-kitaev"`` → ``openfermion.transforms.bravyi_kitaev``
        - ``"bravyi-kitaev-tree"`` → ``openfermion.transforms.bravyi_kitaev_tree``

    Examples:
        >>> from qdk_chemistry.algorithms import create
        >>> from qdk_chemistry.data import MajoranaMapping
        >>> mapper = create("qubit_mapper", "openfermion")
        >>> mapping = MajoranaMapping.jordan_wigner(num_modes=n_spin_orbitals)
        >>> qh = mapper.run(hamiltonian, mapping)

    """

    def __init__(self):
        """Initialize OpenFermionQubitMapper."""
        Logger.trace_entering()
        super().__init__()
        self._settings = OpenFermionQubitMapperSettings()

    def _run_impl(
        self,
        hamiltonian: Hamiltonian,
        mapping: MajoranaMapping,
    ) -> QubitHamiltonian:
        """Build a qubit Hamiltonian via OpenFermion (name-dispatched).

        Reads ``mapping.base_encoding`` to select an OpenFermion transform
        function.  ``mapping.table`` is **not used** — the qubit operator
        is rebuilt entirely by OpenFermion's own pipeline.

        If *mapping* carries tapering metadata, the base encoding is
        extracted first, mapped, and tapering is applied to the result
        via :meth:`~QubitMapper._taper_result`.

        Args:
            hamiltonian: The fermionic Hamiltonian (restricted or unrestricted).
            mapping: Encoding selector — only ``base_encoding`` is read.

        Returns:
            QubitHamiltonian: An instance of the QubitHamiltonian.

        Raises:
            NotImplementedError: If ``mapping.base_encoding`` is not a supported OpenFermion encoding.

        """
        Logger.trace_entering()

        # --- Name dispatch (see QubitMapper class docstring) ---
        encoding_name = mapping.base_encoding

        if encoding_name not in _STANDARD_TRANSFORMS:
            raise NotImplementedError(
                f"OpenFermion plugin does not support base encoding {encoding_name!r}. "
                f"Supported encodings: {sorted(_STANDARD_TRANSFORMS.keys())}."
            )

        qubit_op = self._map_standard(hamiltonian, encoding_name)
        fermion_mode_order = FermionModeOrder.BLOCKED

        Logger.debug(f"Mapping Hamiltonian with OpenFermion encoding: {encoding_name}")

        qubit_op.compress()

        # OpenFermion folds core_energy into the identity Pauli term.
        # QDK convention stores core energy separately, so subtract it.
        core_energy = hamiltonian.get_core_energy()
        if abs(core_energy) > np.finfo(np.float64).eps:
            qubit_op -= core_energy * of.QubitOperator(())
            qubit_op.compress()

        qh = qubit_operator_to_qubit_hamiltonian(
            qubit_op,
            encoding=encoding_name,
            fermion_mode_order=fermion_mode_order,
        )
        return self._taper_result(qh, mapping)

    def _map_standard(self, hamiltonian: Hamiltonian, encoding: str) -> of.QubitOperator:
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
        transform = _STANDARD_TRANSFORMS[encoding]
        return transform(fermion_op)

    def name(self) -> str:
        """Return the algorithm name ``openfermion``."""
        Logger.trace_entering()
        return "openfermion"


def _build_blocked_fermion_operator(hamiltonian: Hamiltonian) -> of.FermionOperator:
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
