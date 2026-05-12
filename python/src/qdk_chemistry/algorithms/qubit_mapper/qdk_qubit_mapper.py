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

    This mapper transforms a fermionic Hamiltonian to a qubit Hamiltonian.
    The encoding is determined by the :class:`~qdk_chemistry.data.MajoranaMapping`
    passed to :meth:`run`. Any valid MajoranaMapping works -- built-in
    (Jordan-Wigner, Bravyi-Kitaev, parity) or custom.

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
        """Transform a fermionic Hamiltonian to a qubit Hamiltonian.

        Args:
            hamiltonian: The fermionic Hamiltonian with one-body and two-body integrals.
            mapping: The Majorana-to-Pauli encoding.

        Returns:
            QubitHamiltonian: The qubit Hamiltonian with Pauli strings and coefficients.

        """
        Logger.trace_entering()

        threshold = float(self.settings().get("threshold"))
        integral_threshold = float(self.settings().get("integral_threshold"))

        h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
        h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()
        n_spatial = h1_alpha.shape[0]
        n_spin_orbitals = 2 * n_spatial
        is_restricted = hamiltonian.get_orbitals().is_restricted()

        if mapping.num_modes != n_spin_orbitals:
            raise ValueError(
                f"MajoranaMapping has {mapping.num_modes} modes but the Hamiltonian has "
                f"{n_spin_orbitals} spin-orbitals (2 x {n_spatial} spatial orbitals). "
                f"Use MajoranaMapping.jordan_wigner(num_modes={n_spin_orbitals}) or equivalent."
            )

        # Use ravel() instead of flatten() to avoid copying contiguous arrays.
        # For restricted Hamiltonians the containers share the same two-body
        # vector across aaaa/aabb/bbbb, so pass the same array to avoid
        # materializing unused copies.
        h1_a_flat = np.ascontiguousarray(h1_alpha).ravel()
        h1_b_flat = h1_a_flat if is_restricted else np.ascontiguousarray(h1_beta).ravel()
        h2_aaaa_flat = np.ascontiguousarray(h2_aaaa).ravel()
        h2_aabb_flat = h2_aaaa_flat if is_restricted else np.ascontiguousarray(h2_aabb).ravel()
        h2_bbbb_flat = h2_aaaa_flat if is_restricted else np.ascontiguousarray(h2_bbbb).ravel()

        # Single C++ call: Majorana-loop engine builds all Pauli terms
        pauli_strings, coefficients = majorana_map_hamiltonian(
            mapping.core,
            0.0,  # core energy not included (QDK convention)
            h1_a_flat,
            h1_b_flat,
            h2_aaaa_flat,
            h2_aabb_flat,
            h2_bbbb_flat,
            n_spatial,
            is_restricted,
            threshold,
            integral_threshold,
        )

        Logger.debug(f"Generated {len(pauli_strings)} Pauli terms for {2 * n_spatial} qubits")

        qh = QubitHamiltonian(
            pauli_strings=list(pauli_strings),
            coefficients=np.array(coefficients, dtype=complex),
            encoding=mapping.base_encoding,
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )

        # Apply post-mapping tapering if specified (e.g. SCBK)
        if mapping.tapering is not None:
            from qdk_chemistry.utils.tapering import taper_qubits  # noqa: PLC0415

            qh = taper_qubits(qh, mapping.tapering.qubit_indices, mapping.tapering.eigenvalues)
            qh = QubitHamiltonian(
                pauli_strings=qh.pauli_strings,
                coefficients=qh.coefficients,
                encoding=mapping.name,
                fermion_mode_order=qh.fermion_mode_order,
                tapering=mapping.tapering,
            )
            Logger.debug(f"Tapered {mapping.tapering.num_tapered} qubits → {qh.num_qubits} qubits")

        return qh
