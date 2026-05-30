"""QDK native qubit mapper using Majorana-level C++ building blocks.

This module provides the QdkQubitMapper class for transforming electronic structure
Hamiltonians to qubit Hamiltonians. The encoding is specified by a
:class:`~qdk_chemistry.data.MajoranaMapping` passed to :meth:`run`, making the
mapper encoding-agnostic.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry._core.data import majorana_map_hamiltonian
from qdk_chemistry.algorithms.qubit_mapper.qubit_mapper import QubitMapper, QubitMapperSettings
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.data.majorana_mapping import _sparse_to_dense_le
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import Hamiltonian
    from qdk_chemistry.data.majorana_mapping import MajoranaMapping

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

        This backend reads ``mapping.core`` (the C++ MajoranaMapping) and
        uses the Pauli-string table directly.  The ``base_encoding`` name
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

        h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
        h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()
        n_spatial = h1_alpha.shape[0]
        n_spin_orbitals = 2 * n_spatial
        # Restricted orbitals produce spin-symmetric integrals, enabling the
        # spin-summed fast path in the engine.
        spin_symmetric = hamiltonian.get_orbitals().is_restricted()

        if base_mapping.num_modes != n_spin_orbitals:
            raise ValueError(
                f"MajoranaMapping has {base_mapping.num_modes} modes but the Hamiltonian has "
                f"{n_spin_orbitals} spin-orbitals (2 x {n_spatial} spatial orbitals). "
                f"Use MajoranaMapping.jordan_wigner(num_modes={n_spin_orbitals}) or equivalent."
            )

        # Use ravel() instead of flatten() to avoid copying contiguous arrays.
        # For spin-symmetric integrals (restricted orbitals) the containers
        # share the same arrays across spin channels.
        h1_a_flat = np.ascontiguousarray(h1_alpha).ravel()
        h1_b_flat = h1_a_flat if spin_symmetric else np.ascontiguousarray(h1_beta).ravel()
        h2_aaaa_flat = np.ascontiguousarray(h2_aaaa).ravel()
        h2_aabb_flat = h2_aaaa_flat if spin_symmetric else np.ascontiguousarray(h2_aabb).ravel()
        h2_bbbb_flat = h2_aaaa_flat if spin_symmetric else np.ascontiguousarray(h2_bbbb).ravel()

        # Single C++ call: Majorana-loop engine builds all Pauli terms as sparse words
        words, coefficients = majorana_map_hamiltonian(
            base_mapping.core,
            0.0,  # core energy not included (QDK convention)
            h1_a_flat,
            h1_b_flat,
            h2_aaaa_flat,
            h2_aabb_flat,
            h2_bbbb_flat,
            n_spatial,
            spin_symmetric,
            threshold,
            integral_threshold,
        )

        # Render sparse words into dense little-endian strings (Python owns string form)
        n_qubits = base_mapping.num_qubits
        pauli_strings = [_sparse_to_dense_le(word, n_qubits) for word in words]

        Logger.debug(f"Generated {len(pauli_strings)} Pauli terms for {2 * n_spatial} qubits")

        qh = QubitHamiltonian(
            pauli_strings=pauli_strings,
            coefficients=np.array(coefficients, dtype=complex),
            encoding=base_mapping.base_encoding,
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )
        return self._taper_result(qh, mapping)
