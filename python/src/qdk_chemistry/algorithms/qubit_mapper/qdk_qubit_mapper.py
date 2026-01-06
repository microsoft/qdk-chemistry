"""QDK native qubit mapper using an optimized expression layer.

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
from qdk_chemistry.data import PauliTermAccumulator, Settings
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import Hamiltonian

# Type alias for sparse Pauli word: list of (qubit_index, op_type)
# op_type: 1=X, 2=Y, 3=Z (identity is implicit/omitted)
SparsePauliWord = list[tuple[int, int]]

# Pauli operator type constants
_X = 1
_Y = 2
_Z = 3

__all__ = ["QdkQubitMapper", "QdkQubitMapperSettings"]

# =============================================================================
# Bravyi-Kitaev Binary Tree Index Sets
# =============================================================================
# The following functions compute the qubit index sets for Bravyi-Kitaev encoding
# as defined in the original paper:
#
#   Seeley, Richard, and Love. "The Bravyi-Kitaev transformation for quantum
#   computation of electronic structure." J. Chem. Phys. 137, 224109 (2012).
#   https://doi.org/10.1063/1.4768229
#
# The BK encoding maps fermionic operators to qubit operators using a binary
# tree structure where:
#   - Even-indexed qubits store occupation information
#   - Odd-indexed qubits store partial parity sums
#
# Each ladder operator requires three index sets derived from the tree structure:
#   - P(j): "parity set" - qubits encoding parity of orbitals < j
#   - U(j): "update set" - ancestor qubits whose parity includes orbital j
#   - F(j): "flip set" - subset of P(j) for the imaginary component
#   - R(j): "remainder set" = P(j) \ F(j) - used for Z operators in Y component
# =============================================================================


def _bk_compute_parity_indices(j: int, n: int) -> frozenset[int]:
    """Compute qubit indices encoding cumulative parity for orbital j.

    In the Bravyi-Kitaev binary tree, the parity set P(j) contains indices
    of qubits that together encode the parity of all orbitals with index < j.
    This is used to construct the Z-string in the real component of ladder operators.

    Reference: Seeley et al., J. Chem. Phys. 137, 224109 (2012), Eq. (17).

    Args:
        j: Spin-orbital index.
        n: Size of the binary superset (must be a power of 2).

    Returns:
        Frozenset of qubit indices in the parity set.

    """
    if n % 2 != 0:
        return frozenset()
    half = n // 2
    if j < half:
        return _bk_compute_parity_indices(j, half)
    # Right half: recurse with offset, then add n/2-1
    return frozenset(i + half for i in _bk_compute_parity_indices(j - half, half)) | frozenset({half - 1})


def _bk_compute_ancestor_indices(j: int, n: int) -> frozenset[int]:
    """Compute qubit indices that are ancestors of orbital j in the binary tree.

    The update set U(j) contains indices of qubits whose stored parity value
    must be flipped when orbital j is occupied. These correspond to ancestor
    nodes in the binary tree representation.

    Reference: Seeley et al., J. Chem. Phys. 137, 224109 (2012), Eq. (18).

    Args:
        j: Spin-orbital index.
        n: Size of the binary superset (must be a power of 2).

    Returns:
        Frozenset of qubit indices in the update (ancestor) set.

    """
    if n % 2 != 0:
        return frozenset()
    half = n // 2
    if j < half:
        # Left half: include n-1 and recurse
        return frozenset({n - 1}) | _bk_compute_ancestor_indices(j, half)
    # Right half: recurse with offset
    return frozenset(i + half for i in _bk_compute_ancestor_indices(j - half, half))


def _bk_compute_children_indices(j: int, n: int) -> frozenset[int]:
    """Compute qubit indices for the imaginary component parity subset.

    The flip set F(j) is used to partition the parity set when constructing
    the Y-component of BK ladder operators. It identifies which parity qubits
    contribute to the imaginary vs real components.

    Reference: Seeley et al., J. Chem. Phys. 137, 224109 (2012), Eq. (19).

    Args:
        j: Spin-orbital index.
        n: Size of the binary superset (must be a power of 2).

    Returns:
        Frozenset of qubit indices in the flip (children) set.

    """
    if n % 2 != 0:
        return frozenset()
    half = n // 2
    if j < half:
        # Left half: recurse
        return _bk_compute_children_indices(j, half)
    if j < n - 1:
        # Right half but not last: recurse with offset
        return frozenset(i + half for i in _bk_compute_children_indices(j - half, half))
    # Last element (j == n-1): recurse with offset and add n/2-1
    return frozenset(i + half for i in _bk_compute_children_indices(j - half, half)) | frozenset({half - 1})


def _bk_compute_z_indices_for_y_component(j: int, n: int) -> frozenset[int]:
    r"""Compute qubit indices for Z operators in the Y-component of ladder operators.

    The remainder set R(j) = P(j) \\ F(j) determines which qubits receive Z gates
    in the imaginary (Y) component of BK ladder operators. This set difference
    partitions the parity information between real and imaginary components.

    Reference: Seeley et al., J. Chem. Phys. 137, 224109 (2012), Section II.B.

    Args:
        j: Spin-orbital index.
        n: Size of the binary superset (must be a power of 2).

    Returns:
        Frozenset of qubit indices for Z operators in Y-component.

    """
    parity = _bk_compute_parity_indices(j, n)
    flip = _bk_compute_children_indices(j, n)
    return parity - flip  # Set difference, not symmetric difference


class QdkQubitMapperSettings(Settings):
    """Settings configuration for a QdkQubitMapper.

    QdkQubitMapper-specific settings:
        mapping_type (string, default="jordan_wigner"): Fermion-to-qubit encoding type.
            Valid options: "jordan_wigner", "bravyi_kitaev"

        threshold (double, default=1e-12): Threshold for pruning small Pauli coefficients.

        integral_threshold (double, default=1e-12): Threshold for filtering small integrals.
            Integrals with absolute value below this threshold are treated as zero.
            This significantly improves performance when integrals contain floating-point noise.

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
            ["jordan_wigner", "bravyi_kitaev"],
        )
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
    """QDK native qubit mapper using PauliTermAccumulator.

    This mapper transforms a fermionic Hamiltonian to a qubit Hamiltonian using
    configurable fermion-to-qubit encodings. Supports Jordan-Wigner and Bravyi-Kitaev
    encodings.

    The mapper uses canonical blocked spin-orbital ordering internally:
    qubits 0..N-1 for alpha spin, qubits N..2N-1 for beta spin (where N is the
    number of spatial orbitals). Use ``QubitHamiltonian.reorder_qubits()`` or
    ``QubitHamiltonian.to_interleaved()`` for alternative qubit orderings.

    Attributes:
        mapping_type (str): The fermion-to-qubit encoding type. Default: "jordan_wigner".
        threshold (float): Threshold for pruning small Pauli coefficients. Default: 1e-12.
        integral_threshold (float): Threshold for filtering small integrals. Default: 1e-12.

    Examples:
        >>> from qdk_chemistry.algorithms import QdkQubitMapper
        >>> mapper = QdkQubitMapper()
        >>> mapper.settings().set("mapping_type", "jordan_wigner")
        >>> mapper.settings().set("threshold", 1e-10)
        >>> qubit_hamiltonian = mapper.run(hamiltonian)

    """

    def __init__(
        self,
        mapping_type: str = "jordan_wigner",
        threshold: float = 1e-12,
        integral_threshold: float = 1e-12,
    ) -> None:
        """Initialize the QdkQubitMapper with default settings.

        Args:
            mapping_type: Fermion-to-qubit encoding type. Default: "jordan_wigner".
            threshold: Threshold for pruning small Pauli coefficients. Default: 1e-12.
            integral_threshold: Threshold for filtering small integrals. Default: 1e-12.

        """
        super().__init__()
        self._settings = QdkQubitMapperSettings()
        self._settings.set("mapping_type", mapping_type)
        self._settings.set("threshold", threshold)
        self._settings.set("integral_threshold", integral_threshold)

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
        integral_threshold = float(self.settings().get("integral_threshold"))

        if mapping_type == "jordan_wigner":
            return self._jordan_wigner_transform(hamiltonian, threshold, integral_threshold)
        if mapping_type == "bravyi_kitaev":
            return self._bravyi_kitaev_transform(hamiltonian, threshold, integral_threshold)

        raise ValueError(f"Unsupported mapping type: '{mapping_type}'.")

    def _jordan_wigner_transform(
        self, hamiltonian: Hamiltonian, threshold: float, integral_threshold: float
    ) -> QubitHamiltonian:
        """Perform Jordan-Wigner transformation.

        Uses blocked spin-orbital ordering: alpha orbitals first, then beta orbitals.
        Spin-orbital index p = spatial_orbital for alpha, p = spatial_orbital + n_spatial for beta.

        Args:
            hamiltonian: The fermionic Hamiltonian.
            threshold: Threshold for pruning small coefficients.
            integral_threshold: Threshold for discarding small integrals.

        Returns:
            QubitHamiltonian: The transformed qubit Hamiltonian.

        """
        Logger.trace_entering()

        h1_alpha, _ = hamiltonian.get_one_body_integrals()
        n_spin_orbitals = 2 * h1_alpha.shape[0]

        # Use C++ to compute all N² excitation terms in one call
        Logger.debug("Computing all JW excitation terms in C++...")
        all_excitation_terms = PauliTermAccumulator.compute_all_jw_excitation_terms(n_spin_orbitals)

        return self._transform_with_excitation_terms_dict(
            hamiltonian, threshold, integral_threshold, n_spin_orbitals, all_excitation_terms
        )

    def _bravyi_kitaev_transform(
        self, hamiltonian: Hamiltonian, threshold: float, integral_threshold: float
    ) -> QubitHamiltonian:
        r"""Perform Bravyi-Kitaev transformation.

        Implements the fermion-to-qubit encoding from Seeley, Richard, and Love,
        "The Bravyi-Kitaev transformation for quantum computation of electronic
        structure," J. Chem. Phys. 137, 224109 (2012).

        Uses blocked spin-orbital ordering: alpha orbitals first, then beta orbitals.
        The Bravyi-Kitaev encoding uses a binary tree structure where even-indexed
        qubits store occupation and odd-indexed qubits store partial parity sums,
        achieving O(log n) operator weight compared to O(n) for Jordan-Wigner.

        Args:
            hamiltonian: The fermionic Hamiltonian.
            threshold: Threshold for pruning small coefficients.
            integral_threshold: Threshold for discarding small integrals.

        Returns:
            QubitHamiltonian: The transformed qubit Hamiltonian.

        """
        Logger.trace_entering()

        h1_alpha, _ = hamiltonian.get_one_body_integrals()
        n_spin_orbitals = 2 * h1_alpha.shape[0]

        # Binary superset size (next power of 2)
        bin_sup = 1
        while n_spin_orbitals > 2**bin_sup:
            bin_sup += 1
        n_binary = 2**bin_sup

        # Precompute BK index sets for all orbitals (as dict[int, list[int]] for C++)
        update_sets: dict[int, list[int]] = {}
        parity_sets: dict[int, list[int]] = {}
        remainder_sets: dict[int, list[int]] = {}

        for j in range(n_spin_orbitals):
            update_sets[j] = sorted(i for i in _bk_compute_ancestor_indices(j, n_binary) if i < n_spin_orbitals)
            parity_sets[j] = sorted(i for i in _bk_compute_parity_indices(j, n_binary) if i < n_spin_orbitals)
            remainder_sets[j] = sorted(
                i for i in _bk_compute_z_indices_for_y_component(j, n_binary) if i < n_spin_orbitals
            )

        # Use C++ to compute all N² excitation terms in one call
        Logger.debug("Computing all BK excitation terms in C++...")
        all_excitation_terms = PauliTermAccumulator.compute_all_bk_excitation_terms(
            n_spin_orbitals, parity_sets, update_sets, remainder_sets
        )

        return self._transform_with_excitation_terms_dict(
            hamiltonian, threshold, integral_threshold, n_spin_orbitals, all_excitation_terms
        )

    def _transform_with_excitation_terms_dict(
        self,
        hamiltonian: Hamiltonian,
        threshold: float,
        integral_threshold: float,
        n_spin_orbitals: int,
        excitation_terms_dict: dict[tuple[int, int], list[tuple[complex, SparsePauliWord]]],
    ) -> QubitHamiltonian:
        """Transform Hamiltonian to qubit representation using precomputed excitation terms.

        This is the shared infrastructure for all fermion-to-qubit encodings.
        It handles integral extraction, excitation operator construction,
        spin-summed operators, one-body and two-body terms, and output processing.

        The second-quantized Hamiltonian in chemist notation is:
            H = sum_{pq,sigma} h_pq a†_{p,sigma} a_{q,sigma}
              + 1/2 sum_{pqrs,sigma,tau} (pq|rs) a†_{p,sigma} a†_{r,tau} a_{s,tau} a_{q,sigma}

        Args:
            hamiltonian: The fermionic Hamiltonian.
            threshold: Threshold for pruning small coefficients.
            integral_threshold: Threshold for discarding small integrals.
            n_spin_orbitals: Total number of spin orbitals.
            excitation_terms_dict: Pre-computed dictionary mapping (p, q) to
                E_pq = a†_p a_q terms in sparse format.

        Returns:
            QubitHamiltonian: The transformed qubit Hamiltonian.

        """
        Logger.trace_entering()

        h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
        h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()
        core_energy = hamiltonian.get_core_energy()

        n_spatial = h1_alpha.shape[0]

        # Use C++ PauliTermAccumulator for efficient term accumulation
        accumulator = PauliTermAccumulator()

        # Add core energy as identity term (empty sparse word = identity)
        accumulator.accumulate([], complex(core_energy))

        # Spin-orbital index functions (blocked ordering: alpha then beta)
        def alpha_idx(p: int) -> int:
            return p

        def beta_idx(p: int) -> int:
            return p + n_spatial

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

        def get_excitation_terms(p: int, q: int) -> list[tuple[complex, SparsePauliWord]]:
            """Get excitation terms E_pq = a†_p a_q from precomputed dictionary."""
            return excitation_terms_dict[(p, q)]

        # Cache for spin-summed excitation operator terms
        # E_pq = E_pq_alpha + E_pq_beta (indexed by spatial orbitals)
        _spin_summed_terms_cache: dict[tuple[int, int], list[tuple[complex, SparsePauliWord]]] = {}

        def get_spin_summed_terms(p: int, q: int) -> list[tuple[complex, SparsePauliWord]]:
            """Get sparse terms for spin-summed E_pq with caching."""
            key = (p, q)
            if key in _spin_summed_terms_cache:
                return _spin_summed_terms_cache[key]
            # Combine alpha and beta excitation terms
            alpha_terms = get_excitation_terms(alpha_idx(p), alpha_idx(q))
            beta_terms = get_excitation_terms(beta_idx(p), beta_idx(q))
            # Merge terms with same sparse word
            combined: dict[tuple[tuple[int, int], ...], complex] = {}
            for coeff, word in alpha_terms:
                key_tuple = tuple(word)
                combined[key_tuple] = combined.get(key_tuple, 0) + coeff
            for coeff, word in beta_terms:
                key_tuple = tuple(word)
                combined[key_tuple] = combined.get(key_tuple, 0) + coeff
            # Filter using machine epsilon for numerical stability (not user threshold)
            terms = [
                (coeff, list(word_tuple))
                for word_tuple, coeff in combined.items()
                if abs(coeff) > np.finfo(np.float64).eps
            ]
            _spin_summed_terms_cache[key] = terms
            return terms

        # Pre-populate spin-summed terms cache
        Logger.debug("Pre-computing spin-summed excitation terms...")
        for i in range(n_spatial):
            for j in range(n_spatial):
                get_spin_summed_terms(i, j)

        def accumulate_terms(terms: list[tuple[complex, SparsePauliWord]], scale: complex) -> None:
            """Accumulate a list of (coeff, sparse_word) terms with a scale factor."""
            for coeff, word in terms:
                accumulator.accumulate(word, scale * coeff)

        def accumulate_product_terms(
            terms1: list[tuple[complex, SparsePauliWord]],
            terms2: list[tuple[complex, SparsePauliWord]],
            scale: complex,
        ) -> None:
            """Accumulate the product of two term lists with a scale factor."""
            for coeff1, word1 in terms1:
                for coeff2, word2 in terms2:
                    accumulator.accumulate_product(word1, word2, scale * coeff1 * coeff2)

        Logger.debug("Building one-body terms...")
        is_spin_free = np.allclose(h1_alpha, h1_beta) and np.allclose(h2_aaaa, h2_bbbb)

        if is_spin_free:
            for p in range(n_spatial):
                for q in range(n_spatial):
                    h_pq = float(h1_alpha[p, q])
                    if abs(h_pq) > integral_threshold:
                        terms = get_spin_summed_terms(p, q)
                        accumulate_terms(terms, complex(h_pq))
        else:
            # General case: handle alpha and beta separately
            for p in range(n_spatial):
                for q in range(n_spatial):
                    h_pq_alpha = float(h1_alpha[p, q])
                    h_pq_beta = float(h1_beta[p, q])

                    if abs(h_pq_alpha) > integral_threshold:
                        terms = get_excitation_terms(alpha_idx(p), alpha_idx(q))
                        accumulate_terms(terms, complex(h_pq_alpha))

                    if abs(h_pq_beta) > integral_threshold:
                        terms = get_excitation_terms(beta_idx(p), beta_idx(q))
                        accumulate_terms(terms, complex(h_pq_beta))

        Logger.debug("Building two-body terms...")

        if is_spin_free:
            # Spin-free case: use spin-summed factorization
            for p in range(n_spatial):
                for q in range(n_spatial):
                    for r in range(n_spatial):
                        for s in range(n_spatial):
                            eri = get_eri(p, q, r, s, "aaaa")
                            if abs(eri) > integral_threshold:
                                e_pq_terms = get_spin_summed_terms(p, q)
                                e_rs_terms = get_spin_summed_terms(r, s)
                                accumulate_product_terms(e_pq_terms, e_rs_terms, complex(0.5 * eri))

                                if q == r:
                                    e_ps_terms = get_spin_summed_terms(p, s)
                                    accumulate_terms(e_ps_terms, complex(-0.5 * eri))
        else:
            for channel_name, spin1_idx, spin2_idx, channel_key, is_same_spin in [
                ("aaaa", alpha_idx, alpha_idx, "aaaa", True),
                ("bbbb", beta_idx, beta_idx, "bbbb", True),
                ("aabb", alpha_idx, beta_idx, "aabb", False),
                ("bbaa", beta_idx, alpha_idx, "aabb", False),
            ]:
                Logger.debug(f"Processing {channel_name} channel...")

                for p in range(n_spatial):
                    for q in range(n_spatial):
                        for r in range(n_spatial):
                            if is_same_spin and p == r:
                                continue
                            for s in range(n_spatial):
                                if is_same_spin and q == s:
                                    continue
                                eri = get_eri(p, q, r, s, channel_key)
                                if abs(eri) > integral_threshold:
                                    e_pq_terms = get_excitation_terms(spin1_idx(p), spin1_idx(q))
                                    e_rs_terms = get_excitation_terms(spin2_idx(r), spin2_idx(s))
                                    accumulate_product_terms(e_pq_terms, e_rs_terms, complex(0.5 * eri))

                                    if is_same_spin and q == r:
                                        e_ps_terms = get_excitation_terms(spin1_idx(p), spin1_idx(s))
                                        accumulate_terms(e_ps_terms, complex(-0.5 * eri))

        Logger.debug("Finalizing Pauli terms...")

        # Get terms from C++ accumulator as canonical strings (only place we use strings)
        canonical_terms = accumulator.get_terms_as_strings(n_spin_orbitals, threshold)

        pauli_strings = []
        coefficients = []

        for coeff, pauli_str in canonical_terms:
            # Convert to Qiskit-style little-endian ordering
            pauli_strings.append(pauli_str[::-1])
            coefficients.append(coeff)

        Logger.debug(f"Generated {len(pauli_strings)} Pauli terms for {n_spin_orbitals} qubits")

        return QubitHamiltonian(
            pauli_strings=pauli_strings,
            coefficients=np.array(coefficients, dtype=complex),
        )
