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
        # Pre-build all operators upfront to maximize cache benefit
        _creation_cache = {}
        _annihilation_cache = {}

        def creation_operator(p: int):
            """Build a_p^dagger using Jordan-Wigner."""
            if p in _creation_cache:
                return _creation_cache[p]
            x_term = PauliOperator.X(p)
            y_term = PauliOperator.Y(p)
            result = 0.5 * (x_term - 1j * y_term)
            for j in range(p - 1, -1, -1):
                result = result * PauliOperator.Z(j)
            _creation_cache[p] = result
            return result

        def annihilation_operator(p: int):
            """Build a_p using Jordan-Wigner."""
            if p in _annihilation_cache:
                return _annihilation_cache[p]
            x_term = PauliOperator.X(p)
            y_term = PauliOperator.Y(p)
            result = 0.5 * (x_term + 1j * y_term)
            for j in range(p - 1, -1, -1):
                result = result * PauliOperator.Z(j)
            _annihilation_cache[p] = result
            return result

        # Pre-populate caches for all spin-orbitals
        for i in range(n_spin_orbitals):
            creation_operator(i)
            annihilation_operator(i)

        # Cache for one-body excitation operators E_pq = a†_p a_q
        _excitation_cache = {}

        def excitation_operator(p: int, q: int):
            """Build E_pq = a†_p a_q with caching and diagonal optimization."""
            key = (p, q)
            if key in _excitation_cache:
                return _excitation_cache[key]

            if p == q:
                # Diagonal: n_p = a†_p a_p = (I - Z_p) / 2
                # Direct construction avoids expression tree multiplication
                result = 0.5 * PauliOperator.I(p) - 0.5 * PauliOperator.Z(p)
            else:
                # Off-diagonal: multiply cached ladder operators
                result = creation_operator(p) * annihilation_operator(q)

            _excitation_cache[key] = result
            return result

        # Pre-populate excitation cache for all spin-orbital pairs
        for i in range(n_spin_orbitals):
            for j in range(n_spin_orbitals):
                excitation_operator(i, j)

        # Cache for spin-summed excitation operators E_pq = a†_pα a_qα + a†_pβ a_qβ
        # Indexed by spatial orbital indices (p, q)
        _spin_summed_excitation_cache = {}

        def spin_summed_excitation(p: int, q: int):
            """Build spin-summed E_pq = a†_pα a_qα + a†_pβ a_qβ with caching.

            These are indexed by spatial orbitals and sum over both spin channels.
            The result is simplified and cached for reuse in two-body factorization.
            """
            key = (p, q)
            if key in _spin_summed_excitation_cache:
                return _spin_summed_excitation_cache[key]

            # E_pq = E_pq_alpha + E_pq_beta (spin-orbital excitations)
            E_alpha = excitation_operator(alpha_idx(p), alpha_idx(q))
            E_beta = excitation_operator(beta_idx(p), beta_idx(q))
            result = (E_alpha + E_beta).simplify()

            _spin_summed_excitation_cache[key] = result
            return result

        # Pre-populate spin-summed excitation cache for all spatial orbital pairs
        for i in range(n_spatial):
            for j in range(n_spatial):
                spin_summed_excitation(i, j)

        # One-body terms: sum_{pq} h_pq * E_pq (using spin-summed operators for spin-free case)
        # For spin-free Hamiltonians: h_pq_alpha == h_pq_beta, so we can use spin-summed E_pq
        Logger.debug("Building one-body terms...")
        is_spin_free = np.allclose(h1_alpha, h1_beta) and np.allclose(h2_aaaa, h2_bbbb)

        if is_spin_free:
            # Use spin-summed excitation operators for efficiency
            for p in range(n_spatial):
                for q in range(n_spatial):
                    h_pq = float(h1_alpha[p, q])
                    if h_pq != 0.0:
                        term = h_pq * spin_summed_excitation(p, q)
                        qubit_expr = qubit_expr + term
        else:
            # General case: handle alpha and beta separately
            for p in range(n_spatial):
                for q in range(n_spatial):
                    h_pq_alpha = float(h1_alpha[p, q])
                    h_pq_beta = float(h1_beta[p, q])

                    if h_pq_alpha != 0.0:
                        term = h_pq_alpha * excitation_operator(alpha_idx(p), alpha_idx(q))
                        qubit_expr = qubit_expr + term

                    if h_pq_beta != 0.0:
                        term = h_pq_beta * excitation_operator(beta_idx(p), beta_idx(q))
                        qubit_expr = qubit_expr + term

        # Simplify one-body terms to keep expression tree flat
        qubit_expr = qubit_expr.simplify()

        # Two-body terms using chemist notation:
        # H_2 = 1/2 sum_{pqrs,sigma,tau} (pq|rs) a†_{p,sigma} a†_{r,tau} a_{s,tau} a_{q,sigma}
        #
        # For spin-free Hamiltonians, use spin-summed factorization:
        #   e_pqrs = E_pq * E_rs - δ_qr * E_ps
        # where E_pq = a†_pα a_qα + a†_pβ a_qβ (spin-summed, already simplified and cached)
        #
        # This reduces the loop from 4*n^4 to n^4 iterations and reuses cached operators.
        Logger.debug("Building two-body terms...")

        if is_spin_free:
            # Spin-free case: use spin-summed factorization
            # H_2 = (1/2) Σ_{pqrs} g_pqrs [E_pq * E_rs - δ_qr * E_ps]
            # where E_pq are spin-summed and already simplified/cached
            for p in range(n_spatial):
                # Build terms for this p value (n³ terms)
                p_batch_expr = 0.0 * PauliOperator.I(0)
                for q in range(n_spatial):
                    for r in range(n_spatial):
                        for s in range(n_spatial):
                            eri = get_eri(p, q, r, s, "aaaa")  # All channels are equal
                            if eri != 0.0:
                                # Use spin-summed factorization: E_pq * E_rs - δ_qr * E_ps
                                E_pq = spin_summed_excitation(p, q)
                                E_rs = spin_summed_excitation(r, s)
                                product_term = E_pq * E_rs

                                if q == r:
                                    # Kronecker delta correction
                                    E_ps = spin_summed_excitation(p, s)
                                    term = 0.5 * eri * (product_term - E_ps)
                                else:
                                    term = 0.5 * eri * product_term

                                p_batch_expr = p_batch_expr + term

                # Simplify this p-batch and accumulate
                p_batch_simplified = p_batch_expr.simplify()
                qubit_expr = qubit_expr + p_batch_simplified

        else:
            # General case: process each spin channel separately
            for channel_name, spin1_idx, spin2_idx, channel_key, is_same_spin in [
                ("aaaa", alpha_idx, alpha_idx, "aaaa", True),
                ("bbbb", beta_idx, beta_idx, "bbbb", True),
                ("aabb", alpha_idx, beta_idx, "aabb", False),
                ("bbaa", beta_idx, alpha_idx, "aabb", False),
            ]:
                Logger.debug(f"Processing {channel_name} channel...")
                channel_expr = 0.0 * PauliOperator.I(0)

                for p in range(n_spatial):
                    p_batch_expr = 0.0 * PauliOperator.I(0)
                    for q in range(n_spatial):
                        for r in range(n_spatial):
                            if is_same_spin and p == r:
                                continue
                            for s in range(n_spatial):
                                if is_same_spin and q == s:
                                    continue
                                eri = get_eri(p, q, r, s, channel_key)
                                if eri != 0.0:
                                    E_pq = excitation_operator(spin1_idx(p), spin1_idx(q))
                                    E_rs = excitation_operator(spin2_idx(r), spin2_idx(s))
                                    product_term = E_pq * E_rs

                                    if is_same_spin and q == r:
                                        E_ps = excitation_operator(spin1_idx(p), spin1_idx(s))
                                        term = 0.5 * eri * (product_term - E_ps)
                                    else:
                                        term = 0.5 * eri * product_term

                                    p_batch_expr = p_batch_expr + term

                    p_batch_simplified = p_batch_expr.simplify()
                    channel_expr = channel_expr + p_batch_simplified

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
