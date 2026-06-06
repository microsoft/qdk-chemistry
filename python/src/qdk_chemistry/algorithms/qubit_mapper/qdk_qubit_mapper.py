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
    majorana_map_hamiltonian_factorized,
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

    Sparse and factorized fast paths: when the input Hamiltonian is backed by a
    :class:`~qdk_chemistry.data.CholeskyHamiltonianContainer` (three-center /
    density-fitted factors) or a
    :class:`~qdk_chemistry.data.SparseHamiltonianContainer` (sparse two-body
    map), the two-body integrals are consumed directly in their compressed form
    and the dense ``N**4`` tensor is never materialized. This reduces both memory
    and runtime while producing a result numerically equivalent (term-by-term,
    to within ``1e-12``) to the dense path. Selection is automatic based on the
    container type; all other containers use the dense path.

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

        h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
        n_spatial = h1_alpha.shape[0]
        n_spin_orbitals = 2 * n_spatial
        spin_symmetric = hamiltonian.get_orbitals().is_restricted()

        if base_mapping.num_modes != n_spin_orbitals:
            raise ValueError(
                f"MajoranaMapping has {base_mapping.num_modes} modes but the Hamiltonian has "
                f"{n_spin_orbitals} spin-orbitals (2 x {n_spatial} spatial orbitals). "
                f"Use MajoranaMapping.jordan_wigner(num_modes={n_spin_orbitals}) or equivalent."
            )

        # Fast paths: factorized (Cholesky) and sparse containers feed their
        # native low-rank / sparse two-body integrals straight into the C++
        # engine, so the dense N^4 two-body tensor is never materialized. The
        # output is numerically equivalent to the dense path (see the C++
        # ``majorana_map_hamiltonian_factorized`` dispatcher).
        container_type = hamiltonian.get_container_type()
        if container_type in ("cholesky", "sparse"):
            words, coefficients = majorana_map_hamiltonian_factorized(
                base_mapping,
                hamiltonian,
                spin_symmetric,
                threshold,
                integral_threshold,
            )
        else:
            h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()

            h1_a_flat = np.ascontiguousarray(h1_alpha).ravel()
            h1_b_flat = h1_a_flat if spin_symmetric else np.ascontiguousarray(h1_beta).ravel()
            h2_aaaa_flat = np.ascontiguousarray(h2_aaaa).ravel()
            h2_aabb_flat = h2_aaaa_flat if spin_symmetric else np.ascontiguousarray(h2_aabb).ravel()
            h2_bbbb_flat = h2_aaaa_flat if spin_symmetric else np.ascontiguousarray(h2_bbbb).ravel()

            words, coefficients = majorana_map_hamiltonian(
                base_mapping,
                0.0,
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
