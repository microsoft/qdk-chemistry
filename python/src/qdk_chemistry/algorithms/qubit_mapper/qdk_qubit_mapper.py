"""QDK native qubit mapper using PauliOperator expression layer.

This module provides the QdkQubitMapper class for transforming electronic structure
Hamiltonians to qubit Hamiltonians using various fermion-to-qubit encodings.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.algorithms.qubit_mapper.qubit_mapper import QubitMapper
from qdk_chemistry.data import PauliOperator, Settings
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import Hamiltonian

__all__ = ["QdkQubitMapper", "QdkQubitMapperSettings"]

# Valid mapping types (extensible for future encodings)
_VALID_MAPPING_TYPES = frozenset({"jordan_wigner"})


class QdkQubitMapperSettings(Settings):
    """Settings configuration for a QdkQubitMapper.

    QdkQubitMapper-specific settings:
        mapping_type (string, default="jordan_wigner"): Fermion-to-qubit encoding type.
            Valid options: "jordan_wigner"

        threshold (double, default=1e-12): Threshold for pruning small coefficients.

    """

    def __init__(self) -> None:
        """Initialize QdkQubitMapperSettings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default(
            "mapping_type",
            "string",
            "jordan_wigner",
            "Fermion-to-qubit encoding type",
            ["jordan_wigner"],
        )
        self._set_default(
            "threshold",
            "double",
            1e-12,
            "Threshold for pruning small coefficients",
        )


class QdkQubitMapper(QubitMapper):
    """QDK native qubit mapper using the PauliOperator expression layer.

    This mapper transforms a fermionic Hamiltonian to a qubit Hamiltonian using
    configurable fermion-to-qubit encodings. Currently supports Jordan-Wigner encoding.

    The mapper uses canonical blocked spin-orbital ordering internally:
    qubits 0..N-1 for alpha spin, qubits N..2N-1 for beta spin (where N is the
    number of spatial orbitals). Use ``QubitHamiltonian.reorder_qubits()`` or
    ``QubitHamiltonian.to_interleaved()`` for alternative qubit orderings.

    Attributes:
        mapping_type (str): The fermion-to-qubit encoding type. Default: "jordan_wigner".
        threshold (float): Threshold for pruning small coefficients. Default: 1e-12.

    Examples:
        >>> from qdk_chemistry.algorithms import QdkQubitMapper
        >>> mapper = QdkQubitMapper()
        >>> mapper.settings().set("mapping_type", "jordan_wigner")
        >>> mapper.settings().set("threshold", 1e-10)
        >>> qubit_hamiltonian = mapper.run(hamiltonian)

    """

    def __init__(self, mapping_type: str = "jordan_wigner", threshold: float = 1e-12) -> None:
        """Initialize the QdkQubitMapper with default settings.

        Args:
            mapping_type: Fermion-to-qubit encoding type. Default: "jordan_wigner".
            threshold: Threshold for pruning small coefficients. Default: 1e-12.

        """
        super().__init__()
        self._settings = QdkQubitMapperSettings()
        self._settings.set("mapping_type", mapping_type)
        self._settings.set("threshold", threshold)

    def name(self) -> str:
        """Return the algorithm name."""
        return "qdk"

    def _run_impl(self, hamiltonian: Hamiltonian) -> QubitHamiltonian:
        """Transform a fermionic Hamiltonian to a qubit Hamiltonian.

        Args:
            hamiltonian: The fermionic Hamiltonian with one-body and two-body integrals.

        Returns:
            QubitHamiltonian: The qubit Hamiltonian with Pauli strings and coefficients.

        Raises:
            ValueError: If the mapping type is not supported.
            RuntimeError: If the Hamiltonian does not have required integrals.

        """
        Logger.trace_entering()

        mapping_type = str(self.settings().get("mapping_type"))
        threshold = float(self.settings().get("threshold"))

        if mapping_type == "jordan_wigner":
            return self._jordan_wigner_transform(hamiltonian, threshold)

        raise ValueError(f"Unsupported mapping type: '{mapping_type}'.")

    def _jordan_wigner_transform(self, hamiltonian: Hamiltonian, threshold: float) -> QubitHamiltonian:
        """Perform Jordan-Wigner transformation.

        Uses blocked spin-orbital ordering: alpha orbitals first, then beta orbitals.
        Spin-orbital index p = spatial_orbital for alpha, p = spatial_orbital + n_spatial for beta.

        The second-quantized Hamiltonian in chemist notation is:
            H = sum_{pq,sigma} h_pq a†_{p,sigma} a_{q,sigma}
              + 1/2 sum_{pqrs,sigma,tau} (pq|rs) a†_{p,sigma} a†_{r,tau} a_{s,tau} a_{q,sigma}

        Args:
            hamiltonian: The fermionic Hamiltonian.
            threshold: Threshold for pruning small coefficients.

        Returns:
            QubitHamiltonian: The transformed qubit Hamiltonian.

        """
        Logger.trace_entering()

        # Get integrals - using chemist notation (pq|rs)
        h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
        h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()
        core_energy = hamiltonian.get_core_energy()

        # Infer n_spatial from integral shape (not from orbitals which may include
        # inactive/virtual orbitals outside the active space)
        n_spatial = h1_alpha.shape[0]
        n_spin_orbitals = 2 * n_spatial

        # Build the qubit Hamiltonian expression
        # Start with core energy as identity term
        qubit_expr = core_energy * PauliOperator.I(0)

        # Spin-orbital index functions (blocked ordering: alpha then beta)
        def alpha_idx(p: int) -> int:
            return p

        def beta_idx(p: int) -> int:
            return p + n_spatial

        # Get chemist integral (pq|rs)
        def get_eri(p: int, q: int, r: int, s: int, channel: str) -> float:
            """Get (pq|rs) integral in chemist notation."""
            idx = p + q * n_spatial + r * n_spatial**2 + s * n_spatial**3
            if channel == "aaaa":
                return float(h2_aaaa[idx])
            if channel == "aabb":
                return float(h2_aabb[idx])
            if channel == "bbbb":
                return float(h2_bbbb[idx])
            return 0.0

        # Build ladder operators using Jordan-Wigner transformation
        def creation_operator(p: int):
            """Build a_p^dagger using Jordan-Wigner."""
            x_term = PauliOperator.X(p)
            y_term = PauliOperator.Y(p)
            result = 0.5 * (x_term - 1j * y_term)
            for j in range(p - 1, -1, -1):
                result = result * PauliOperator.Z(j)
            return result

        def annihilation_operator(p: int):
            """Build a_p using Jordan-Wigner."""
            x_term = PauliOperator.X(p)
            y_term = PauliOperator.Y(p)
            result = 0.5 * (x_term + 1j * y_term)
            for j in range(p - 1, -1, -1):
                result = result * PauliOperator.Z(j)
            return result

        # One-body terms: sum_{pq,sigma} h_pq * a†_{p,sigma} * a_{q,sigma}
        Logger.debug("Building one-body terms...")
        for p in range(n_spatial):
            for q in range(n_spatial):
                h_pq_alpha = float(h1_alpha[p, q])
                h_pq_beta = float(h1_beta[p, q])

                if h_pq_alpha != 0.0:
                    term = h_pq_alpha * creation_operator(alpha_idx(p)) * annihilation_operator(alpha_idx(q))
                    qubit_expr = qubit_expr + term

                if h_pq_beta != 0.0:
                    term = h_pq_beta * creation_operator(beta_idx(p)) * annihilation_operator(beta_idx(q))
                    qubit_expr = qubit_expr + term

        # Simplify one-body terms to keep expression tree flat
        qubit_expr = qubit_expr.simplify()

        # Two-body terms using chemist notation:
        # H_2 = 1/2 sum_{pqrs,sigma,tau} (pq|rs) a†_{p,sigma} a†_{r,tau} a_{s,tau} a_{q,sigma}
        #
        # For same-spin (sigma = tau):
        #   1/2 sum_{pqrs} (pq|rs) [a†_{pa} a†_{ra} a_{sa} a_{qa} + a†_{pb} a†_{rb} a_{sb} a_{qb}]
        # For opposite-spin (sigma != tau):
        #   1/2 sum_{pqrs} (pq|rs) [a†_{pa} a†_{rb} a_{sb} a_{qa} + a†_{pb} a†_{ra} a_{sa} a_{qb}]
        Logger.debug("Building two-body terms...")

        # Process each spin channel separately and simplify frequently to prevent
        # exponential growth of the expression tree during distribute()
        # Simplify after each p-index batch (n³ terms) to keep expression tree small
        for channel_name, get_indices, channel_key in [
            ("aaaa", lambda p, q, r, s: (alpha_idx(p), alpha_idx(q), alpha_idx(r), alpha_idx(s)), "aaaa"),
            ("bbbb", lambda p, q, r, s: (beta_idx(p), beta_idx(q), beta_idx(r), beta_idx(s)), "bbbb"),
            ("aabb", lambda p, q, r, s: (alpha_idx(p), alpha_idx(q), beta_idx(r), beta_idx(s)), "aabb"),
            ("bbaa", lambda p, q, r, s: (beta_idx(p), beta_idx(q), alpha_idx(r), alpha_idx(s)), "aabb"),
        ]:
            Logger.debug(f"Processing {channel_name} channel...")
            channel_expr = 0.0 * PauliOperator.I(0)  # Start with zero expression

            for p in range(n_spatial):
                # Build terms for this p value
                p_batch_expr = 0.0 * PauliOperator.I(0)
                for q in range(n_spatial):
                    for r in range(n_spatial):
                        for s in range(n_spatial):
                            eri = get_eri(p, q, r, s, channel_key)
                            if eri != 0.0:
                                pi, qi, ri, si = get_indices(p, q, r, s)
                                # a†_pi a†_ri a_si a_qi
                                term = (
                                    0.5
                                    * eri
                                    * creation_operator(pi)
                                    * creation_operator(ri)
                                    * annihilation_operator(si)
                                    * annihilation_operator(qi)
                                )
                                p_batch_expr = p_batch_expr + term

                # Simplify this p-batch and accumulate
                p_batch_simplified = p_batch_expr.simplify()
                channel_expr = channel_expr + p_batch_simplified

            # Add this channel to main expression
            qubit_expr = qubit_expr + channel_expr

        # Simplify and prune the expression
        # Threshold is applied to final Pauli coefficients, not input fermionic coefficients
        Logger.debug("Simplifying expression...")
        simplified = qubit_expr.simplify()
        pruned = simplified.prune_threshold(threshold)

        # Extract canonical terms
        canonical_terms = pruned.to_canonical_terms(n_spin_orbitals)

        # Build output arrays
        # Note: PauliOperator uses big-endian ordering (qubit 0 at leftmost),
        # but Qiskit convention is little-endian (qubit 0 at rightmost).
        # We reverse the strings to match Qiskit convention.
        pauli_strings = []
        coefficients = []

        for coeff, pauli_str in canonical_terms:
            if abs(coeff) > threshold:
                # Reverse string to convert from big-endian to little-endian
                pauli_strings.append(pauli_str[::-1])
                coefficients.append(coeff)

        Logger.debug(f"Generated {len(pauli_strings)} Pauli terms for {n_spin_orbitals} qubits")

        return QubitHamiltonian(
            pauli_strings=pauli_strings,
            coefficients=np.array(coefficients, dtype=complex),
        )
