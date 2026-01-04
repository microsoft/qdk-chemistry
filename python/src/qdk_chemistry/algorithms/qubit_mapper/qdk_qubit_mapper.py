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
    from collections.abc import Callable

    from qdk_chemistry.data import Hamiltonian

__all__ = ["QdkQubitMapper", "QdkQubitMapperSettings"]

# Valid mapping types (extensible for future encodings)
_VALID_MAPPING_TYPES = frozenset({"jordan_wigner", "bravyi_kitaev"})

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
    """QDK native qubit mapper using the PauliOperator expression layer.

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

        # Get number of spin-orbitals for cache pre-population
        h1_alpha, _ = hamiltonian.get_one_body_integrals()
        n_spin_orbitals = 2 * h1_alpha.shape[0]

        # Build ladder operators using Jordan-Wigner transformation
        # Pre-build all operators upfront to maximize cache benefit
        _creation_cache: dict[int, PauliOperator] = {}
        _annihilation_cache: dict[int, PauliOperator] = {}

        def creation_operator(p: int) -> PauliOperator:
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

        def annihilation_operator(p: int) -> PauliOperator:
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

        return self._transform_with_ladder_ops(
            hamiltonian, threshold, integral_threshold, creation_operator, annihilation_operator
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

        The ladder operators are decomposed into real and imaginary Pauli components:

            Creation:     a†_j = (1/2) * (X_component - i * Y_component)
            Annihilation: a_j  = (1/2) * (X_component + i * Y_component)

        Where the Pauli components are constructed from binary tree index sets:

            X_component = Z_{P(j)} · X_j · X_{U(j)}
            Y_component = Z_{R(j)} · Y_j · X_{U(j)}

        Here P(j) is the parity set, U(j) is the ancestor (update) set, and
        R(j) = P(j) \\ F(j) is the remainder set for the Y-component Z-string.

        Args:
            hamiltonian: The fermionic Hamiltonian.
            threshold: Threshold for pruning small coefficients.
            integral_threshold: Threshold for discarding small integrals.

        Returns:
            QubitHamiltonian: The transformed qubit Hamiltonian.

        """
        Logger.trace_entering()

        # Get number of spin-orbitals for cache pre-population
        h1_alpha, _ = hamiltonian.get_one_body_integrals()
        n_spin_orbitals = 2 * h1_alpha.shape[0]

        # Find binary superset size (next power of 2 >= n_spin_orbitals)
        bin_sup = 1
        while n_spin_orbitals > 2**bin_sup:
            bin_sup += 1
        n_binary = 2**bin_sup

        # Pre-compute BK sets for all spin-orbitals using binary superset
        # Then filter to indices < n_spin_orbitals
        update_sets: dict[int, frozenset[int]] = {}
        parity_sets: dict[int, frozenset[int]] = {}
        remainder_sets: dict[int, frozenset[int]] = {}

        for j in range(n_spin_orbitals):
            update_sets[j] = frozenset(i for i in _bk_compute_ancestor_indices(j, n_binary) if i < n_spin_orbitals)
            parity_sets[j] = frozenset(i for i in _bk_compute_parity_indices(j, n_binary) if i < n_spin_orbitals)
            remainder_sets[j] = frozenset(
                i for i in _bk_compute_z_indices_for_y_component(j, n_binary) if i < n_spin_orbitals
            )

        def build_pauli_chain(pauli_type: str, qubit_indices: frozenset[int]) -> PauliOperator:
            """Construct tensor product of identical Pauli operators on specified qubits.

            For a set of qubit indices {i, j, k, ...}, builds P_i ⊗ P_j ⊗ P_k ⊗ ...
            where P is X, Y, or Z. Returns identity if the set is empty.
            """
            if not qubit_indices:
                return PauliOperator.I(0)
            sorted_qubits = sorted(qubit_indices)
            if pauli_type == "X":
                factory = PauliOperator.X
            elif pauli_type == "Y":
                factory = PauliOperator.Y
            else:
                factory = PauliOperator.Z
            result = factory(sorted_qubits[0])
            for q in sorted_qubits[1:]:
                result = result * factory(q)
            return result

        # Construct BK ladder operators from binary tree index sets.
        # The decomposition into X and Y components follows from the requirement
        # that creation/annihilation operators satisfy fermionic anticommutation.
        # Reference: Seeley et al., J. Chem. Phys. 137, 224109 (2012), Eq. (20-21).

        _creation_cache: dict[int, PauliOperator] = {}
        _annihilation_cache: dict[int, PauliOperator] = {}

        def creation_operator(p: int) -> PauliOperator:
            """Build fermionic creation operator a†_p in Bravyi-Kitaev encoding."""
            if p in _creation_cache:
                return _creation_cache[p]

            # X-component: Z on parity qubits, X on orbital p, X on ancestor qubits
            x_component = PauliOperator.X(p)
            if parity_sets[p]:
                x_component = build_pauli_chain("Z", parity_sets[p]) * x_component
            if update_sets[p]:
                x_component = x_component * build_pauli_chain("X", update_sets[p])

            # Y-component: Z on remainder qubits, Y on orbital p, X on ancestor qubits
            y_component = PauliOperator.Y(p)
            if remainder_sets[p]:
                y_component = build_pauli_chain("Z", remainder_sets[p]) * y_component
            if update_sets[p]:
                y_component = y_component * build_pauli_chain("X", update_sets[p])

            # Creation operator: a†_p = (1/2)(X_component - i·Y_component)
            result = 0.5 * x_component - 0.5j * y_component

            _creation_cache[p] = result
            return result

        def annihilation_operator(p: int) -> PauliOperator:
            """Build fermionic annihilation operator a_p in Bravyi-Kitaev encoding."""
            if p in _annihilation_cache:
                return _annihilation_cache[p]

            # X-component: Z on parity qubits, X on orbital p, X on ancestor qubits
            x_component = PauliOperator.X(p)
            if parity_sets[p]:
                x_component = build_pauli_chain("Z", parity_sets[p]) * x_component
            if update_sets[p]:
                x_component = x_component * build_pauli_chain("X", update_sets[p])

            # Y-component: Z on remainder qubits, Y on orbital p, X on ancestor qubits
            y_component = PauliOperator.Y(p)
            if remainder_sets[p]:
                y_component = build_pauli_chain("Z", remainder_sets[p]) * y_component
            if update_sets[p]:
                y_component = y_component * build_pauli_chain("X", update_sets[p])

            # Annihilation operator: a_p = (1/2)(X_component + i·Y_component)
            result = 0.5 * x_component + 0.5j * y_component

            _annihilation_cache[p] = result
            return result

        # Pre-populate caches for all spin-orbitals
        for i in range(n_spin_orbitals):
            creation_operator(i)
            annihilation_operator(i)

        return self._transform_with_ladder_ops(
            hamiltonian, threshold, integral_threshold, creation_operator, annihilation_operator
        )

    def _transform_with_ladder_ops(
        self,
        hamiltonian: Hamiltonian,
        threshold: float,
        integral_threshold: float,
        creation_operator: Callable[[int], PauliOperator],
        annihilation_operator: Callable[[int], PauliOperator],
    ) -> QubitHamiltonian:
        """Transform Hamiltonian to qubit representation using provided ladder operators.

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
            creation_operator: Callable that returns the creation operator for index p.
            annihilation_operator: Callable that returns the annihilation operator for index p.

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

        # Cache for one-body excitation operators E_pq = a^dag_p a_q
        _excitation_cache: dict[tuple[int, int], PauliOperator] = {}

        def excitation_operator(p: int, q: int) -> PauliOperator:
            """Build E_pq = a†_p a_q with caching."""
            key = (p, q)
            if key in _excitation_cache:
                return _excitation_cache[key]

            # Always use ladder operator multiplication
            # The diagonal n_p = a†_p a_p has different forms for different encodings:
            # - Jordan-Wigner: n_p = (I - Z_p) / 2
            # - Bravyi-Kitaev: n_p = (I - Z_p * prod_{k in F(p)} Z_k) / 2
            # Using the general form ensures correctness for all encodings.
            result = creation_operator(p) * annihilation_operator(q)

            _excitation_cache[key] = result
            return result

        # Pre-populate excitation cache for all spin-orbital pairs
        for i in range(n_spin_orbitals):
            for j in range(n_spin_orbitals):
                excitation_operator(i, j)

        # Cache for spin-summed excitation operators E_pq = a_p_alpha^dag a_q_alpha + a_p_beta^dag a_q_beta
        # Indexed by spatial orbital indices (p, q)
        _spin_summed_excitation_cache: dict[tuple[int, int], PauliOperator] = {}

        def spin_summed_excitation(p: int, q: int) -> PauliOperator:
            """Build spin-summed E_pq = a_p_alpha^dag a_q_alpha + a_p_beta^dag a_q_beta with caching.

            These are indexed by spatial orbitals and sum over both spin channels.
            The result is simplified and cached for reuse in two-body factorization.
            """
            key = (p, q)
            if key in _spin_summed_excitation_cache:
                return _spin_summed_excitation_cache[key]

            # E_pq = E_pq_alpha + E_pq_beta (spin-orbital excitations)
            e_alpha = excitation_operator(alpha_idx(p), alpha_idx(q))
            e_beta = excitation_operator(beta_idx(p), beta_idx(q))
            result = (e_alpha + e_beta).simplify()

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
                    if abs(h_pq) > integral_threshold:
                        term = h_pq * spin_summed_excitation(p, q)
                        qubit_expr = qubit_expr + term
        else:
            # General case: handle alpha and beta separately
            for p in range(n_spatial):
                for q in range(n_spatial):
                    h_pq_alpha = float(h1_alpha[p, q])
                    h_pq_beta = float(h1_beta[p, q])

                    if abs(h_pq_alpha) > integral_threshold:
                        term = h_pq_alpha * excitation_operator(alpha_idx(p), alpha_idx(q))
                        qubit_expr = qubit_expr + term

                    if abs(h_pq_beta) > integral_threshold:
                        term = h_pq_beta * excitation_operator(beta_idx(p), beta_idx(q))
                        qubit_expr = qubit_expr + term

        # Simplify one-body terms to keep expression tree flat
        qubit_expr = qubit_expr.simplify()

        # Two-body terms using chemist notation:
        # The two-body Hamiltonian is (1/2) sum over pqrs and spins sigma,tau of
        # (pq|rs) times creation(p,sigma) creation(r,tau) annihilation(s,tau) annihilation(q,sigma)
        #
        # For spin-free Hamiltonians, use spin-summed factorization:
        # e_pqrs becomes E_pq times E_rs minus delta(q,r) times E_ps
        # where E_pq sums over alpha and beta spins (already simplified and cached)
        #
        # This reduces the loop from 4*n^4 to n^4 iterations and reuses cached operators.
        Logger.debug("Building two-body terms...")

        if is_spin_free:
            # Spin-free case: use spin-summed factorization
            # Contribution is (1/2) sum over pqrs of g_pqrs times (E_pq E_rs minus delta(q,r) E_ps)
            # where E_pq are spin-summed and already simplified/cached
            for p in range(n_spatial):
                # Build terms for this p value (n³ terms)
                p_batch_expr = 0.0 * PauliOperator.I(0)
                for q in range(n_spatial):
                    for r in range(n_spatial):
                        for s in range(n_spatial):
                            eri = get_eri(p, q, r, s, "aaaa")  # All channels are equal
                            if abs(eri) > integral_threshold:
                                # Use spin-summed factorization: E_pq * E_rs - delta_qr * E_ps
                                e_pq = spin_summed_excitation(p, q)
                                e_rs = spin_summed_excitation(r, s)
                                product_term = e_pq * e_rs

                                if q == r:
                                    # Kronecker delta correction
                                    e_ps = spin_summed_excitation(p, s)
                                    term = 0.5 * eri * (product_term - e_ps)
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
                                if abs(eri) > integral_threshold:
                                    e_pq = excitation_operator(spin1_idx(p), spin1_idx(q))
                                    e_rs = excitation_operator(spin2_idx(r), spin2_idx(s))
                                    product_term = e_pq * e_rs

                                    if is_same_spin and q == r:
                                        e_ps = excitation_operator(spin1_idx(p), spin1_idx(s))
                                        term = 0.5 * eri * (product_term - e_ps)
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
