"""QDK native qubit mapper using Majorana-level C++ building blocks.

This module provides the QdkQubitMapper class for transforming electronic structure
Hamiltonians to qubit Hamiltonians. The encoding is specified by a
:class:`~qdk_chemistry.data.MajoranaMapping` passed to ``run()``, making the
mapper encoding-agnostic.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry._core.data import (
    majorana_map_hamiltonian,
    sparse_pauli_word_to_label,
)
from qdk_chemistry.algorithms.qubit_mapper.qubit_mapper import QubitMapper, QubitMapperSettings
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import Hamiltonian, MajoranaMapping

__all__ = ["QdkQubitMapper", "QdkQubitMapperSettings"]


class QdkQubitMapperSettings(QubitMapperSettings):
    """Settings configuration for a QdkQubitMapper.

    Settings:
        threshold (double, default=1e-12): Threshold for pruning small Pauli coefficients.
        integral_threshold (double, default=1e-12): Threshold for filtering small integrals.

    """

    def __init__(self) -> None:
        """Initialize QdkQubitMapperSettings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default(
            "threshold",
            "double",
            1e-12,
            "Threshold for pruning small Pauli coefficients",
        )
        self._set_default(
            "integral_threshold",
            "double",
            1e-12,
            "Threshold for filtering small integrals (improves performance)",
        )


class QdkQubitMapper(QubitMapper):
    """QDK native qubit mapper using Majorana-level C++ engine.

    This is a **table-driven** backend: it reads the Pauli-string table
    from the :class:`~qdk_chemistry.data.MajoranaMapping` and passes it
    directly to the C++ ``majorana_map_hamiltonian`` engine.  Any valid
    ``MajoranaMapping`` works — built-in (Jordan-Wigner, Bravyi-Kitaev,
    parity, BK-tree, SCBK) or custom user-defined tables.  The mapping's
    ``name`` and ``base_encoding`` are used only for metadata on the
    output :class:`~qdk_chemistry.data.QubitHamiltonian`, not for dispatch.

    Both restricted (RHF) and unrestricted (UHF) Hamiltonians are supported.
    For unrestricted systems, the engine handles all four spin-channel ERI
    blocks (aa, ab, ba, bb) independently.

    The two-body integrals are consumed in the native storage format of the
    Hamiltonian's container (the C++ engine dispatches on the container type):

    * :class:`~qdk_chemistry.data.SparseHamiltonianContainer`: the engine
      iterates only the stored non-zero ``(p, q, r, s)`` entries, so both
      memory and runtime improve for the mostly-zero integrals of lattice /
      model Hamiltonians.
    * :class:`~qdk_chemistry.data.CholeskyHamiltonianContainer`: the
      three-center factors are kept in their ``O(N**2 * naux)`` form and the
      auxiliary index is contracted one ``(pq|.)`` row at a time, so the dense
      ``N**4`` tensor is never built and peak additional memory is a single
      ``N**2`` row.
    * All other containers use the dense engine path.

    In all cases the result is numerically equivalent (term-by-term, to within
    ``1e-12``) to the dense path.

    The mapper uses canonical blocked spin-orbital ordering internally:
    qubits 0..N-1 for alpha spin, qubits N..2N-1 for beta spin (where N is the
    number of spatial orbitals). Use ``QubitHamiltonian.to_interleaved()``
    for alternative qubit orderings.

    Examples:
        >>> from qdk_chemistry.algorithms import create
        >>> from qdk_chemistry.data import MajoranaMapping
        >>> mapper = create("qubit_mapper")
        >>> mapping = MajoranaMapping.jordan_wigner(num_modes=n_spin_orbitals)
        >>> qh = mapper.run(hamiltonian, mapping)

    """

    def __init__(
        self,
        threshold: float = 1e-12,
        integral_threshold: float = 1e-12,
    ) -> None:
        """Initialize the QdkQubitMapper with default settings.

        Args:
            threshold: Threshold for pruning small Pauli coefficients. Default: 1e-12.
            integral_threshold: Threshold for filtering small integrals. Default: 1e-12.

        """
        super().__init__()
        self._settings = QdkQubitMapperSettings()
        self._settings.set("threshold", threshold)
        self._settings.set("integral_threshold", integral_threshold)

    def name(self) -> str:
        """Return the algorithm name."""
        return "qdk"

    def _run_impl(
        self,
        hamiltonian: Hamiltonian,
        mapping: MajoranaMapping,
    ) -> QubitHamiltonian:
        """Transform a fermionic Hamiltonian to a qubit Hamiltonian (table-driven).

        This backend passes the C++ ``MajoranaMapping`` directly to the
        native mapper.  The ``base_encoding`` name
        is used only for metadata on the output, not for dispatch.

        If *mapping* carries tapering metadata, the base encoding is
        extracted first, mapped, and tapering is applied to the result
        via :meth:`~QubitMapper._taper_result`.

        Args:
            hamiltonian: The fermionic Hamiltonian with one-body and two-body integrals.
            mapping: The Majorana-to-Pauli encoding (table is consumed directly by the C++ engine).

        Returns:
            QubitHamiltonian: The qubit Hamiltonian with Pauli strings and coefficients.

        """
        Logger.trace_entering()

        # Strip tapering — the C++ engine maps the base encoding only
        base_mapping = mapping.without_tapering() if mapping.tapering else mapping

        threshold = float(self.settings().get("threshold"))
        integral_threshold = float(self.settings().get("integral_threshold"))

        n_spatial = hamiltonian.get_one_body_integrals()[0].shape[0]
        n_spin_orbitals = 2 * n_spatial
        spin_symmetric = hamiltonian.get_orbitals().is_restricted()

        if base_mapping.num_modes not in (n_spatial, n_spin_orbitals):
            raise ValueError(
                f"MajoranaMapping has {base_mapping.num_modes} modes but the Hamiltonian has "
                f"{n_spatial} spatial orbitals (which requires either {n_spatial} or {n_spin_orbitals} modes). "
                f"Use MajoranaMapping.jordan_wigner(num_modes={n_spin_orbitals}) or equivalent."
            )

        # The C++ engine dispatches on the container type: sparse and Cholesky
        # containers feed their native sparse / low-rank two-body integrals
        # straight into the engine (the dense N^4 tensor is never
        # materialized); all other containers use the dense engine path.
        words, coefficients = majorana_map_hamiltonian(
            base_mapping,
            hamiltonian,
            spin_symmetric,
            threshold,
            integral_threshold,
        )

        n_qubits = base_mapping.num_qubits
        pauli_strings = [sparse_pauli_word_to_label(word, n_qubits) for word in words]

        Logger.debug(f"Generated {len(pauli_strings)} Pauli terms for {n_qubits} qubits")

        qh = QubitHamiltonian(
            pauli_strings=pauli_strings,
            coefficients=np.array(coefficients, dtype=complex),
            encoding=base_mapping.base_encoding,
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )
        return self._taper_result(qh, mapping)
