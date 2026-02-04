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

    Raises:
        ValueError: If n is not a power of 2.

    """
    if n == 1:
        return frozenset()  # Base case: single orbital has no parity dependencies
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")
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

    Raises:
        ValueError: If n is not a power of 2.

    """
    if n == 1:
        return frozenset()  # Base case: single orbital has no ancestors
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")
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

    Raises:
        ValueError: If n is not a power of 2.

    """
    if n == 1:
        return frozenset()  # Base case: single orbital has no children
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")
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

    Raises:
        ValueError: If n is not a power of 2.

    """
    parity = _bk_compute_parity_indices(j, n)
    flip = _bk_compute_children_indices(j, n)
    return parity - flip  # Set difference, not symmetric difference


class QdkQubitMapperSettings(Settings):
    """Settings configuration for a QdkQubitMapper.

    QdkQubitMapper-specific settings:
        encoding (string, default="jordan-wigner"): Fermion-to-qubit encoding type.
            Valid options: "jordan-wigner", "bravyi-kitaev"

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
            "encoding",
            "string",
            "jordan-wigner",
            "Fermion-to-qubit encoding type",
            ["jordan-wigner", "bravyi-kitaev"],
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
        encoding (str): The fermion-to-qubit encoding type. Default: "jordan-wigner".
        threshold (float): Threshold for pruning small Pauli coefficients. Default: 1e-12.
        integral_threshold (float): Threshold for filtering small integrals. Default: 1e-12.

    Examples:
        >>> from qdk_chemistry.algorithms import QdkQubitMapper
        >>> mapper = QdkQubitMapper()
        >>> mapper.settings().set("encoding", "jordan-wigner")
        >>> mapper.settings().set("threshold", 1e-10)
        >>> qubit_hamiltonian = mapper.run(hamiltonian)

    """

    def __init__(
        self,
        encoding: str = "jordan-wigner",
        threshold: float = 1e-12,
        integral_threshold: float = 1e-12,
    ) -> None:
        """Initialize the QdkQubitMapper with default settings.

        Args:
            encoding: Fermion-to-qubit encoding type. Default: "jordan-wigner".
            threshold: Threshold for pruning small Pauli coefficients. Default: 1e-12.
            integral_threshold: Threshold for filtering small integrals. Default: 1e-12.

        """
        super().__init__()
        self._settings = QdkQubitMapperSettings()
        self._settings.set("encoding", encoding)
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

        encoding = str(self.settings().get("encoding"))
        threshold = float(self.settings().get("threshold"))
        integral_threshold = float(self.settings().get("integral_threshold"))

        if encoding == "jordan-wigner":
            return self._jordan_wigner_transform(hamiltonian, threshold, integral_threshold)
        if encoding == "bravyi-kitaev":
            return self._bravyi_kitaev_transform(hamiltonian, threshold, integral_threshold)

        raise ValueError(f"Unsupported encoding: '{encoding}'.")

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
            hamiltonian, threshold, integral_threshold, n_spin_orbitals, all_excitation_terms, "jordan-wigner"
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
            hamiltonian, threshold, integral_threshold, n_spin_orbitals, all_excitation_terms, "bravyi-kitaev"
        )

    def _transform_with_excitation_terms_dict(
        self,
        hamiltonian: Hamiltonian,
        threshold: float,
        integral_threshold: float,
        n_spin_orbitals: int,
        excitation_terms_dict: dict[tuple[int, int], list[tuple[complex, SparsePauliWord]]],
        encoding: str,
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
            encoding: The fermion-to-qubit encoding used (e.g., "jordan-wigner", "bravyi-kitaev").

        Returns:
            QubitHamiltonian: The transformed qubit Hamiltonian.

        """
        Logger.trace_entering()

        h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
        h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()

        n_spatial = h1_alpha.shape[0]

        # Reshape two-body integrals to 4D tensors for direct indexing (zero-copy view)
        eri_aaaa = h2_aaaa.reshape((n_spatial, n_spatial, n_spatial, n_spatial))
        eri_aabb = h2_aabb.reshape((n_spatial, n_spatial, n_spatial, n_spatial))
        eri_bbbb = h2_bbbb.reshape((n_spatial, n_spatial, n_spatial, n_spatial))

        # Use C++ PauliTermAccumulator for efficient term accumulation
        accumulator = PauliTermAccumulator()

        # Eagerly precompute spin-summed excitation terms: E_pq = E_pq_alpha + E_pq_beta
        # (indexed by spatial orbitals p, q)
        Logger.debug("Pre-computing spin-summed excitation terms...")
        spin_summed_terms: dict[tuple[int, int], list[tuple[complex, SparsePauliWord]]] = {}
        for p in range(n_spatial):
            for q in range(n_spatial):
                # Get alpha and beta excitation terms (inline index computation)
                alpha_terms = excitation_terms_dict[(p, q)]
                beta_terms = excitation_terms_dict[(p + n_spatial, q + n_spatial)]
                # Merge terms with same sparse word
                combined: dict[tuple[tuple[int, int], ...], complex] = {}
                for coeff, word in alpha_terms:
                    key_tuple = tuple(word)
                    combined[key_tuple] = combined.get(key_tuple, 0) + coeff
                for coeff, word in beta_terms:
                    key_tuple = tuple(word)
                    combined[key_tuple] = combined.get(key_tuple, 0) + coeff
                # Filter using machine epsilon for efficiency
                spin_summed_terms[(p, q)] = [
                    (coeff, list(word_tuple))
                    for word_tuple, coeff in combined.items()
                    if abs(coeff) > np.finfo(np.float64).eps
                ]

        Logger.debug("Building one-body terms...")
        is_spin_free = hamiltonian.get_orbitals().is_restricted()

        if is_spin_free:
            for p in range(n_spatial):
                for q in range(n_spatial):
                    h_pq = float(h1_alpha[p, q])
                    if abs(h_pq) > integral_threshold:
                        for coeff, word in spin_summed_terms[(p, q)]:
                            accumulator.accumulate(word, complex(h_pq) * coeff)
        else:
            # General case: handle alpha and beta separately
            for p in range(n_spatial):
                for q in range(n_spatial):
                    h_pq_alpha = float(h1_alpha[p, q])
                    h_pq_beta = float(h1_beta[p, q])

                    if abs(h_pq_alpha) > integral_threshold:
                        for coeff, word in excitation_terms_dict[(p, q)]:
                            accumulator.accumulate(word, complex(h_pq_alpha) * coeff)

                    if abs(h_pq_beta) > integral_threshold:
                        for coeff, word in excitation_terms_dict[(p + n_spatial, q + n_spatial)]:
                            accumulator.accumulate(word, complex(h_pq_beta) * coeff)

        Logger.debug("Building two-body terms...")

        if is_spin_free:
            # Spin-free case: use spin-summed factorization
            for p in range(n_spatial):
                for q in range(n_spatial):
                    for r in range(n_spatial):
                        for s in range(n_spatial):
                            eri = float(eri_aaaa[p, q, r, s])
                            if abs(eri) > integral_threshold:
                                e_pq_terms = spin_summed_terms[(p, q)]
                                e_rs_terms = spin_summed_terms[(r, s)]
                                scale = complex(0.5 * eri)
                                for c1, w1 in e_pq_terms:
                                    for c2, w2 in e_rs_terms:
                                        accumulator.accumulate_product(w1, w2, scale * c1 * c2)

                                if q == r:
                                    for coeff, word in spin_summed_terms[(p, s)]:
                                        accumulator.accumulate(word, complex(-0.5 * eri) * coeff)
        else:
            # Spin-polarized case: explicit channel blocks

            # aaaa channel (same-spin alpha-alpha)
            Logger.debug("Processing aaaa channel...")
            for p in range(n_spatial):
                for q in range(n_spatial):
                    for r in range(n_spatial):
                        if p == r:
                            continue
                        for s in range(n_spatial):
                            if q == s:
                                continue
                            eri = float(eri_aaaa[p, q, r, s])
                            if abs(eri) > integral_threshold:
                                e_pq_terms = excitation_terms_dict[(p, q)]
                                e_rs_terms = excitation_terms_dict[(r, s)]
                                scale = complex(0.5 * eri)
                                for c1, w1 in e_pq_terms:
                                    for c2, w2 in e_rs_terms:
                                        accumulator.accumulate_product(w1, w2, scale * c1 * c2)

                                if q == r:
                                    for coeff, word in excitation_terms_dict[(p, s)]:
                                        accumulator.accumulate(word, complex(-0.5 * eri) * coeff)

            # bbbb channel (same-spin beta-beta)
            Logger.debug("Processing bbbb channel...")
            for p in range(n_spatial):
                for q in range(n_spatial):
                    for r in range(n_spatial):
                        if p == r:
                            continue
                        for s in range(n_spatial):
                            if q == s:
                                continue
                            eri = float(eri_bbbb[p, q, r, s])
                            if abs(eri) > integral_threshold:
                                e_pq_terms = excitation_terms_dict[(p + n_spatial, q + n_spatial)]
                                e_rs_terms = excitation_terms_dict[(r + n_spatial, s + n_spatial)]
                                scale = complex(0.5 * eri)
                                for c1, w1 in e_pq_terms:
                                    for c2, w2 in e_rs_terms:
                                        accumulator.accumulate_product(w1, w2, scale * c1 * c2)

                                if q == r:
                                    for coeff, word in excitation_terms_dict[(p + n_spatial, s + n_spatial)]:
                                        accumulator.accumulate(word, complex(-0.5 * eri) * coeff)

            # aabb channel (mixed alpha-beta)
            Logger.debug("Processing aabb channel...")
            for p in range(n_spatial):
                for q in range(n_spatial):
                    for r in range(n_spatial):
                        for s in range(n_spatial):
                            eri = float(eri_aabb[p, q, r, s])
                            if abs(eri) > integral_threshold:
                                e_pq_terms = excitation_terms_dict[(p, q)]
                                e_rs_terms = excitation_terms_dict[(r + n_spatial, s + n_spatial)]
                                scale = complex(0.5 * eri)
                                for c1, w1 in e_pq_terms:
                                    for c2, w2 in e_rs_terms:
                                        accumulator.accumulate_product(w1, w2, scale * c1 * c2)

            # bbaa channel (mixed beta-alpha)
            Logger.debug("Processing bbaa channel...")
            for p in range(n_spatial):
                for q in range(n_spatial):
                    for r in range(n_spatial):
                        for s in range(n_spatial):
                            eri = float(eri_aabb[p, q, r, s])
                            if abs(eri) > integral_threshold:
                                e_pq_terms = excitation_terms_dict[(p + n_spatial, q + n_spatial)]
                                e_rs_terms = excitation_terms_dict[(r, s)]
                                scale = complex(0.5 * eri)
                                for c1, w1 in e_pq_terms:
                                    for c2, w2 in e_rs_terms:
                                        accumulator.accumulate_product(w1, w2, scale * c1 * c2)

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
            encoding=encoding,
        )
