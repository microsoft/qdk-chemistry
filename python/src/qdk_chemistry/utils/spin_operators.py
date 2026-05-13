"""Spin-S operator construction and qubit encoding for higher-spin lattice models.

This module provides utilities for constructing spin-S operators (Sˣ, Sʸ, Sᶻ)
decomposed into Pauli operator expressions, enabling the construction of
arbitrary-spin Heisenberg and Ising Hamiltonians on qubit hardware.

Each lattice site with spin quantum number S is encoded into ⌈log₂(2S+1)⌉ qubits.
When 2S+1 is not a power of 2, the embedding introduces unphysical computational
basis states that must be handled via energy penalties.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Sequence

import numpy as np

from qdk_chemistry.data import PauliOperator

__all__ = [
    "SpinEncoding",
    "SpinSOperators",
]

# Pauli matrices used in decomposition (column-major-friendly complex128).
_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

_PAULI_LABELS = ("I", "X", "Y", "Z")


def _validate_spin(spin: float) -> int:
    """Validate spin and return 2*spin as an integer.

    Args:
        spin: Spin quantum number (must be a positive integer or half-integer).

    Returns:
        Integer value of 2*spin.

    Raises:
        ValueError: If spin is not a positive integer or half-integer.

    """
    two_s = round(2 * spin)
    if two_s < 1 or not math.isclose(2 * spin, two_s):
        raise ValueError(f"spin must be a positive integer or half-integer, got {spin}")
    return two_s


def _spin_matrices(spin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (2S+1)-dimensional spin matrices Sˣ, Sʸ, Sᶻ.

    Uses standard angular momentum algebra:
    - Sᶻ is diagonal with eigenvalues m = S, S-1, ..., -S (descending)
    - S⁺|m⟩ = √(S(S+1) - m(m+1)) |m+1⟩
    - Sˣ = (S⁺ + S⁻)/2, Sʸ = (S⁺ - S⁻)/(2i)

    Args:
        spin: Spin quantum number S.

    Returns:
        Tuple (Sˣ, Sʸ, Sᶻ) as complex128 matrices of shape (2S+1, 2S+1).

    """
    two_s = _validate_spin(spin)
    dim = two_s + 1
    s = spin

    # Sᶻ: diagonal, eigenvalues descending from S to -S
    m_values = np.arange(s, -s - 0.5, -1.0)
    sz = np.diag(m_values).astype(np.complex128)

    # S⁺ (raising operator): superdiagonal
    s_plus = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim - 1):
        m = m_values[i + 1]  # m of the ket (lower state)
        s_plus[i, i + 1] = np.sqrt(s * (s + 1) - m * (m + 1))

    s_minus = s_plus.T.copy()

    sx = (s_plus + s_minus) / 2.0
    sy = (s_plus - s_minus) / 2.0j

    return sx, sy, sz


def _embed_matrix(mat: np.ndarray, num_qubits: int) -> np.ndarray:
    """Embed a (d x d) matrix into a (2ⁿ x 2ⁿ) matrix by zero-padding.

    The physical states occupy the first d rows/columns; unphysical states are zero.

    Args:
        mat: Matrix of shape (d, d) where d <= 2^num_qubits.
        num_qubits: Number of qubits for the embedding.

    Returns:
        Matrix of shape (2^num_qubits, 2^num_qubits).

    """
    dim_full = 1 << num_qubits
    d = mat.shape[0]
    if d == dim_full:
        return mat.copy()
    result = np.zeros((dim_full, dim_full), dtype=np.complex128)
    result[:d, :d] = mat
    return result


@lru_cache(maxsize=64)
def _pauli_decomposition_cached(
    spin: float,
) -> tuple[dict[str, complex], dict[str, complex], dict[str, complex]]:
    """Compute the Pauli decomposition of spin-S operators (cached by spin value).

    Decomposes Sˣ, Sʸ, Sᶻ into sums of Pauli strings:
      M = Σ_P  (Tr(P · M) / 2ⁿ) · P

    Args:
        spin: Spin quantum number.

    Returns:
        Tuple of dicts (sx_terms, sy_terms, sz_terms) mapping Pauli string labels to complex coefficients.

    """
    two_s = _validate_spin(spin)
    dim = two_s + 1
    num_qubits = math.ceil(math.log2(dim)) if dim > 1 else 1
    dim_full = 1 << num_qubits

    sx, sy, sz = _spin_matrices(spin)
    sx_emb = _embed_matrix(sx, num_qubits)
    sy_emb = _embed_matrix(sy, num_qubits)
    sz_emb = _embed_matrix(sz, num_qubits)

    atol = 1e-14

    results = []
    for mat in (sx_emb, sy_emb, sz_emb):
        terms: dict[str, complex] = {}
        for pauli_labels in itertools.product(_PAULI_LABELS, repeat=num_qubits):
            # Build the n-qubit Pauli matrix via Kronecker product
            p_mat = _PAULI[pauli_labels[0]]
            for lbl in pauli_labels[1:]:
                p_mat = np.kron(p_mat, _PAULI[lbl])

            coeff = np.trace(p_mat @ mat) / dim_full
            if abs(coeff) > atol:
                # Clean up near-real and near-imaginary coefficients
                if abs(coeff.imag) < atol:
                    coeff = complex(coeff.real, 0.0)
                elif abs(coeff.real) < atol:
                    coeff = complex(0.0, coeff.imag)
                label = "".join(pauli_labels)
                terms[label] = coeff
        results.append(terms)

    return tuple(results)  # type: ignore[return-value]


@lru_cache(maxsize=64)
def _penalty_decomposition_cached(spin: float) -> dict[str, complex] | None:
    """Compute the Pauli decomposition of the penalty projector for unphysical states.

    The projector is P_unphys = Σ_{k=dim}^{2ⁿ-1} |k⟩⟨k| = I - Σ_{k=0}^{dim-1} |k⟩⟨k|.

    Args:
        spin: Spin quantum number.

    Returns:
        Dict mapping Pauli string labels to coefficients, or None if dim is a power of 2.

    """
    two_s = _validate_spin(spin)
    dim = two_s + 1
    num_qubits = math.ceil(math.log2(dim)) if dim > 1 else 1
    dim_full = 1 << num_qubits

    if dim == dim_full:
        return None

    # Build projector onto unphysical subspace
    projector = np.zeros((dim_full, dim_full), dtype=np.complex128)
    for k in range(dim, dim_full):
        projector[k, k] = 1.0

    atol = 1e-14
    terms: dict[str, complex] = {}
    for pauli_labels in itertools.product(_PAULI_LABELS, repeat=num_qubits):
        p_mat = _PAULI[pauli_labels[0]]
        for lbl in pauli_labels[1:]:
            p_mat = np.kron(p_mat, _PAULI[lbl])

        coeff = np.trace(p_mat @ projector) / dim_full
        if abs(coeff) > atol:
            if abs(coeff.imag) < atol:
                coeff = complex(coeff.real, 0.0)
            elif abs(coeff.real) < atol:
                coeff = complex(0.0, coeff.imag)
            terms["".join(pauli_labels)] = coeff

    return terms


def _build_pauli_expr(terms: dict[str, complex], qubit_offset: int) -> PauliOperator:
    """Convert a dict of {pauli_label: coeff} into a PauliOperator expression.

    The label convention from the Pauli decomposition uses Kronecker-product order
    (label[0] = MSB qubit). PauliOperator uses qubit indices where lower index = LSB,
    so we reverse the mapping: label[k] → qubit (offset + num_qubits - 1 - k).

    Args:
        terms: Mapping from Pauli string labels (e.g. "XZ") to complex coefficients.
        qubit_offset: Global qubit index offset for this site.

    Returns:
        PauliOperator expression representing the sum.

    """
    _pauli_factory = {"I": PauliOperator.I, "X": PauliOperator.X, "Y": PauliOperator.Y, "Z": PauliOperator.Z}

    num_qubits = len(next(iter(terms)))  # all labels have the same length
    expr = None
    for label, coeff in terms.items():
        # Build the product of single-qubit Paulis for this term.
        # label[0] acts on the MSB → highest qubit index.
        term = coeff
        for k, ch in enumerate(label):
            term = term * _pauli_factory[ch](qubit_offset + num_qubits - 1 - k)
        expr = term if expr is None else expr + term
    return expr


class SpinSOperators:
    """Spin-S operators for a single lattice site, decomposed into Pauli operators.

    Provides Sˣ, Sʸ, Sᶻ as :class:`~qdk_chemistry.data.PauliOperator` expressions
    acting on the qubits assigned to this site. Operator matrices follow the standard
    angular momentum algebra with eigenvalues of Sᶻ being -S, -S+1, ..., S.

    Args:
        spin: Spin quantum number S (positive integer or half-integer).
        qubit_offset: Index of the first qubit for this site in the global qubit register.

    Raises:
        ValueError: If spin is not a positive integer or half-integer.

    """

    def __init__(self, spin: float, qubit_offset: int) -> None:
        two_s = _validate_spin(spin)
        self._spin = spin
        self._two_s = two_s
        self._dim = two_s + 1
        self._num_qubits = math.ceil(math.log2(self._dim)) if self._dim > 1 else 1
        self._qubit_offset = qubit_offset

        sx_terms, sy_terms, sz_terms = _pauli_decomposition_cached(spin)
        self._sx = _build_pauli_expr(sx_terms, qubit_offset)
        self._sy = _build_pauli_expr(sy_terms, qubit_offset)
        self._sz = _build_pauli_expr(sz_terms, qubit_offset)

    @property
    def spin(self) -> float:
        """Spin quantum number S."""
        return self._spin

    @property
    def dim(self) -> int:
        """Number of physical spin states (2S+1)."""
        return self._dim

    @property
    def num_qubits(self) -> int:
        """Number of qubits used to encode this site."""
        return self._num_qubits

    @property
    def qubit_offset(self) -> int:
        """Index of the first qubit for this site."""
        return self._qubit_offset

    @property
    def has_unphysical_states(self) -> bool:
        """Whether the qubit encoding has unphysical (unused) computational basis states."""
        return self._dim < (1 << self._num_qubits)

    @property
    def sx(self) -> PauliOperator:
        """Sˣ operator as a PauliOperator expression."""
        return self._sx

    @property
    def sy(self) -> PauliOperator:
        """Sʸ operator as a PauliOperator expression."""
        return self._sy

    @property
    def sz(self) -> PauliOperator:
        """Sᶻ operator as a PauliOperator expression."""
        return self._sz

    def penalty_projector(self) -> PauliOperator | None:
        """Projector onto the unphysical subspace as a PauliOperator expression.

        Returns:
            PauliOperator expression, or None if the encoding has no unphysical states.

        """
        terms = _penalty_decomposition_cached(self._spin)
        if terms is None:
            return None
        return _build_pauli_expr(terms, self._qubit_offset)


class SpinEncoding:
    """Qubit layout for a lattice with per-site spin assignments.

    Tracks which qubits encode which lattice site, supporting mixed-spin lattices
    where different sites may have different spin quantum numbers.

    Args:
        spins: Per-site spin quantum numbers. A scalar assigns the same spin to all sites (use with ``num_sites``). A sequence assigns spins individually.
        num_sites: Number of lattice sites (required when ``spins`` is a scalar).

    Raises:
        ValueError: If spins is a scalar and num_sites is not provided, or if any spin value is invalid.

    Examples:
        >>> enc = SpinEncoding(0.5, num_sites=4)       # uniform spin-1/2
        >>> enc = SpinEncoding([0.5, 1.5, 0.5, 1.5])  # mixed spins

    """

    def __init__(self, spins: float | Sequence[float], num_sites: int | None = None) -> None:
        if isinstance(spins, (int, float)):
            if num_sites is None:
                raise ValueError("num_sites is required when spins is a scalar")
            self._spins = [float(spins)] * num_sites
        else:
            self._spins = [float(s) for s in spins]

        # Validate all spins and compute qubit layout
        self._offsets: list[int] = []
        self._num_qubits_per_site: list[int] = []
        offset = 0
        for s in self._spins:
            two_s = _validate_spin(s)
            dim = two_s + 1
            nq = math.ceil(math.log2(dim)) if dim > 1 else 1
            self._offsets.append(offset)
            self._num_qubits_per_site.append(nq)
            offset += nq
        self._total_qubits = offset

    @property
    def num_sites(self) -> int:
        """Number of lattice sites."""
        return len(self._spins)

    @property
    def spins(self) -> list[float]:
        """Per-site spin quantum numbers."""
        return list(self._spins)

    @property
    def total_qubits(self) -> int:
        """Total number of qubits across all sites."""
        return self._total_qubits

    @property
    def has_unphysical_states(self) -> bool:
        """Whether any site has unphysical states in its qubit encoding."""
        return any(
            int(2 * s + 1) < (1 << nq) for s, nq in zip(self._spins, self._num_qubits_per_site)
        )

    def site_qubits(self, site: int) -> range:
        """Return the qubit index range for a given lattice site.

        Args:
            site: Lattice site index (0-based).

        Returns:
            Range of qubit indices assigned to this site.

        """
        start = self._offsets[site]
        return range(start, start + self._num_qubits_per_site[site])

    def site_operators(self, site: int) -> SpinSOperators:
        """Create a SpinSOperators instance for a given lattice site.

        Args:
            site: Lattice site index (0-based).

        Returns:
            SpinSOperators for the specified site.

        """
        return SpinSOperators(self._spins[site], self._offsets[site])

    def __repr__(self) -> str:
        spin_str = ", ".join(f"{s}" for s in self._spins)
        return f"SpinEncoding(spins=[{spin_str}], total_qubits={self._total_qubits})"
