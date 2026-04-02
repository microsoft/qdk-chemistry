"""Random wavefunction generation for benchmarking state preparation.

This module generates physically motivated random wavefunctions by
exciting electrons from a Hartree-Fock reference determinant.  It is
intended for benchmarking sparse-isometry and binary-encoding circuits
where the full qdk-chemistry pipeline is not needed.

Public API
----------
generate_determinants_matrix
    Generate a ``(n_dets, 2*n_orbitals)`` matrix of random Slater determinants.
generate_sparse_isometry_matrix
    Generate a ``(2*n_orbitals, n_dets)`` binary matrix in the format expected
    by the sparse isometry state preparation routines.
generate_sparse_isometry_matrices
    Same as above, for multiple determinant counts from a single pool.
generate_random_wavefunction
    Generate a :class:`~qdk_chemistry.data.Wavefunction` from random excitations.
determinants_to_config_strings
    Convert a determinant matrix to configuration strings.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from typing import Optional

import numpy as np
from qdk_chemistry.data import (
    BasisSet,
    CasWavefunctionContainer,
    Configuration,
    Orbitals,
    OrbitalType,
    Shell,
    Wavefunction,
)

# ---------------------------------------------------------------------------
# Determinant generation
# ---------------------------------------------------------------------------


def _hf_determinant(n_alpha: int, n_beta: int, n_orbitals: int) -> np.ndarray:
    """Build the Hartree-Fock reference determinant.

    Layout: [alpha_0, alpha_1, ..., alpha_{n_orbitals-1},
             beta_0,  beta_1,  ..., beta_{n_orbitals-1}]

    The HF state fills the lowest orbitals for each spin channel.
    """
    det = np.zeros(2 * n_orbitals, dtype=np.int8)
    det[:n_alpha] = 1  # alpha electrons in lowest orbitals
    det[n_orbitals : n_orbitals + n_beta] = 1  # beta electrons in lowest orbitals
    return det


def _random_excitation(
    det: np.ndarray,
    n_orbitals: int,
    rng: np.random.Generator,
    max_excitation_order: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Generate a single random excitation from a reference determinant.

    Independently applies excitations in the alpha and beta spin channels.
    The excitation order is randomly chosen between 1 and max_excitation_order.

    Returns None if the excitation produces a duplicate of the input.
    """
    new_det = det.copy()

    # Process alpha and beta channels independently
    for channel_start in (0, n_orbitals):
        channel = det[channel_start : channel_start + n_orbitals]
        occupied = np.where(channel == 1)[0]
        virtual = np.where(channel == 0)[0]

        if len(occupied) == 0 or len(virtual) == 0:
            continue

        max_exc = min(len(occupied), len(virtual))
        if max_excitation_order is not None:
            max_exc = min(max_exc, max_excitation_order)

        # Randomly choose excitation order for this channel (0 means no excitation)
        order = rng.integers(0, max_exc + 1)
        if order == 0:
            continue

        occ_indices = rng.choice(occupied, size=order, replace=False)
        vir_indices = rng.choice(virtual, size=order, replace=False)

        new_det[channel_start + occ_indices] = 0
        new_det[channel_start + vir_indices] = 1

    if np.array_equal(new_det, det):
        return None

    return new_det


def generate_determinants_matrix(
    n_electrons: int,
    n_orbitals: int,
    n_dets: int,
    seed: int = 0,
    max_excitation_order: Optional[int] = None,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    include_hf: bool = True,
) -> np.ndarray:
    """Generate a matrix of Slater determinants by random excitation from HF.

    Each row is a determinant represented as a binary occupation vector over
    2*n_orbitals spin-orbitals (alpha block followed by beta block).

    This function does NOT enumerate the full determinant space; instead it
    randomly samples excitations, making it suitable for large systems where
    C(n_orbitals, n_electrons) is intractable.

    Args:
        n_electrons: Total number of electrons.
        n_orbitals:  Number of spatial orbitals (spin-orbitals = 2 * n_orbitals).
        n_dets:      Number of determinants to generate (including HF if
                     ``include_hf`` is True).
        seed:        Random seed for reproducibility.
        max_excitation_order: Maximum excitation rank per spin channel.
                              None means up to the maximum possible.
        n_alpha:     Number of alpha electrons. Defaults to ``n_electrons // 2``.
        n_beta:      Number of beta electrons. Defaults to
                     ``n_electrons - n_alpha``.
        include_hf:  Whether to always include the HF determinant as the first
                     row.

    Returns:
        np.ndarray of shape ``(n_dets, 2 * n_orbitals)`` with entries 0 or 1.

    Raises:
        ValueError: If inputs are inconsistent.
    """
    if n_alpha is None:
        n_alpha = n_electrons // 2
    if n_beta is None:
        n_beta = n_electrons - n_alpha

    if n_alpha + n_beta != n_electrons:
        raise ValueError(
            f"n_alpha ({n_alpha}) + n_beta ({n_beta}) != n_electrons ({n_electrons})"
        )
    if n_alpha > n_orbitals or n_beta > n_orbitals:
        raise ValueError(
            f"Cannot place {n_alpha} alpha or {n_beta} beta electrons "
            f"in {n_orbitals} orbitals"
        )
    if n_dets < 1:
        raise ValueError("n_dets must be at least 1")

    from math import comb

    max_possible = comb(n_orbitals, n_alpha) * comb(n_orbitals, n_beta)
    if n_dets > max_possible:
        raise ValueError(
            f"Requested {n_dets} determinants but the total space has only "
            f"{max_possible}"
        )

    rng = np.random.default_rng(seed)
    hf = _hf_determinant(n_alpha, n_beta, n_orbitals)

    seen: set[bytes] = set()
    dets: list[np.ndarray] = []

    if include_hf:
        dets.append(hf)
        seen.add(hf.tobytes())

    max_attempts = n_dets * 200
    attempts = 0

    while len(dets) < n_dets and attempts < max_attempts:
        attempts += 1
        new_det = _random_excitation(hf, n_orbitals, rng, max_excitation_order)
        if new_det is None:
            continue
        key = new_det.tobytes()
        if key not in seen:
            seen.add(key)
            dets.append(new_det)

    if len(dets) < n_dets:
        raise RuntimeError(
            f"Only generated {len(dets)}/{n_dets} unique determinants after "
            f"{max_attempts} attempts. Try increasing max_excitation_order or "
            f"reducing n_dets."
        )

    return np.array(dets, dtype=np.int8)


# ---------------------------------------------------------------------------
# Sparse isometry matrix generation
# ---------------------------------------------------------------------------


def _bk_transformation_matrix(n: int) -> np.ndarray:
    """Build the n x n Bravyi-Kitaev transformation matrix.

    The BK transformation converts an occupation-number vector f (JW basis)
    to a BK qubit vector b via  b = B @ f  (mod 2).

    The matrix is built recursively:

        B_1 = [[1]]
        B_{2k} = [[B_k,  0 ],
                  [S_k, B_k]]

    where S_k has its last row all 1s (rest zeros).  For non-power-of-2
    sizes we pad to the next power of 2 and truncate.
    """
    n_padded = 1
    while n_padded < n:
        n_padded *= 2

    def _build(size: int) -> np.ndarray:
        if size == 1:
            return np.array([[1]], dtype=np.int8)
        half = size // 2
        b_half = _build(half)
        top = np.hstack([b_half, np.zeros((half, half), dtype=np.int8)])
        s_k = np.zeros((half, half), dtype=np.int8)
        s_k[half - 1, :] = 1
        bottom = np.hstack([s_k, b_half])
        return np.vstack([top, bottom])

    full = _build(n_padded)
    return full[:n, :n]


def generate_sparse_isometry_matrix(
    n_electrons: int,
    n_orbitals: int,
    n_dets: int,
    seed: int = 0,
    max_excitation_order: Optional[int] = None,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    include_hf: bool = True,
    encoding: str = "jordan-wigner",
) -> np.ndarray:
    """Generate a binary matrix in the sparse isometry format.

    Output shape is ``(2 * n_orbitals, n_dets)`` where each column is a
    determinant bitstring and each row is a qubit (spin-orbital).

    Args:
        n_electrons: Total number of electrons.
        n_orbitals:  Number of spatial orbitals.
        n_dets:      Number of determinants to generate.
        seed:        Random seed for reproducibility.
        max_excitation_order: Maximum excitation rank per spin channel.
        n_alpha:     Number of alpha electrons.
        n_beta:      Number of beta electrons.
        include_hf:  Whether to include the HF determinant as the first column.
        encoding:    ``"jordan-wigner"`` or ``"bravyi-kitaev"``.

    Returns:
        np.ndarray of shape ``(2 * n_orbitals, n_dets)`` with entries 0 or 1.
    """
    if encoding not in ("jordan-wigner", "bravyi-kitaev"):
        raise ValueError(
            f"Unknown encoding '{encoding}'. "
            "Supported: 'jordan-wigner', 'bravyi-kitaev'"
        )

    det_matrix = generate_determinants_matrix(
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        n_dets=n_dets,
        seed=seed,
        max_excitation_order=max_excitation_order,
        n_alpha=n_alpha,
        n_beta=n_beta,
        include_hf=include_hf,
    )

    # det_matrix is (n_dets, 2*n_orbitals) with [alpha|beta] layout.
    # Qubit order q[0]..q[2N-1] = alpha_0 .. alpha_{N-1}, beta_0 .. beta_{N-1}
    # which matches det_matrix columns — just transpose.
    matrix = det_matrix.T.astype(np.int8)

    if encoding == "bravyi-kitaev":
        bk = _bk_transformation_matrix(2 * n_orbitals)
        matrix = np.mod(bk @ matrix, 2).astype(np.int8)

    return matrix


def generate_sparse_isometry_matrices(
    n_electrons: int,
    n_orbitals: int,
    det_counts: list[int],
    seed: int = 0,
    max_excitation_order: Optional[int] = None,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    include_hf: bool = True,
    encoding: str = "jordan-wigner",
) -> dict[int, np.ndarray]:
    """Generate sparse isometry matrices for multiple determinant counts.

    Generates the pool once for ``max(det_counts)`` and slices prefixes.

    Returns:
        Dict mapping each count to an ``(2*n_orbitals, count)`` array.
    """
    max_dets = max(det_counts)
    full_matrix = generate_sparse_isometry_matrix(
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        n_dets=max_dets,
        seed=seed,
        max_excitation_order=max_excitation_order,
        n_alpha=n_alpha,
        n_beta=n_beta,
        include_hf=include_hf,
        encoding=encoding,
    )
    return {k: full_matrix[:, :k].copy() for k in sorted(det_counts)}


# ---------------------------------------------------------------------------
# Configuration string helpers
# ---------------------------------------------------------------------------


def determinants_to_config_strings(matrix: np.ndarray, n_orbitals: int) -> list[str]:
    """Convert a determinant matrix to configuration strings.

    Convention: '2' = doubly occupied, 'u' = alpha only,
    'd' = beta only, '0' = empty.

    Args:
        matrix: Shape ``(n_dets, 2 * n_orbitals)`` binary matrix.
        n_orbitals: Number of spatial orbitals.

    Returns:
        List of configuration strings, one per determinant.
    """
    configs = []
    for det in matrix:
        alpha = det[:n_orbitals]
        beta = det[n_orbitals:]
        chars = []
        for a, b in zip(alpha, beta):
            if a == 1 and b == 1:
                chars.append("2")
            elif a == 1 and b == 0:
                chars.append("u")
            elif a == 0 and b == 1:
                chars.append("d")
            else:
                chars.append("0")
        configs.append("".join(chars))
    return configs


# ---------------------------------------------------------------------------
# Wavefunction generation
# ---------------------------------------------------------------------------


def create_test_basis_set(num_atomic_orbitals, name="test-basis", structure=None):
    """Create a test basis set with the specified number of atomic orbitals.

    Args:
        num_atomic_orbitals: Number of atomic orbitals to generate
        name: Name for the basis set
        structure: a structure to attach

    Returns:
        qdk_chemistry.data.BasisSet: A valid basis set for testing

    """
    shells = []
    atom_index = 0
    functions_created = 0

    # Create shells to reach the desired number of atomic orbitals
    while functions_created < num_atomic_orbitals:
        remaining = num_atomic_orbitals - functions_created

        if remaining >= 3:
            # Add a P shell (3 functions: Px, Py, Pz)
            exps = np.array([1.0, 0.5])
            coefs = np.array([0.6, 0.4])
            shell = Shell(atom_index, OrbitalType.P, exps, coefs)
            shells.append(shell)
            functions_created += 3
        elif remaining >= 1:
            # Add S shells for remaining functions (1 function each)
            for _ in range(remaining):
                exps = np.array([1.0])
                coefs = np.array([1.0])
                shell = Shell(atom_index, OrbitalType.S, exps, coefs)
                shells.append(shell)
                functions_created += 1
    if structure is not None:
        return BasisSet(name, shells, structure)
    return BasisSet(name, shells)


def create_test_orbitals(num_orbitals: int):
    """Helper: construct Orbitals immutably with identity coeffs and occupations.

    Occupations are set in restricted form as total occupancy per MO (0/1/2).
    """
    coeffs = np.eye(num_orbitals)
    basis_set = create_test_basis_set(num_orbitals)
    return Orbitals(coeffs, None, None, basis_set)


def generate_random_wavefunction(
    n_electrons: int,
    n_orbitals: int,
    n_dets: int,
    seed: int = 0,
    max_excitation_order: Optional[int] = None,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    include_hf: bool = True,
) -> "Wavefunction":
    """Generate a random :class:`~qdk_chemistry.data.Wavefunction`.

    Follows the same construction pattern as the ``wavefunction_4e4o``
    test fixture in ``conftest.py``:

    1. Generate determinants via random excitation from HF.
    2. Convert to :class:`~qdk_chemistry.data.Configuration` strings.
    3. Assign random normalised coefficients.
    4. Wrap in ``CasWavefunctionContainer`` → ``Wavefunction``.

    Args:
        n_electrons: Total number of electrons.
        n_orbitals:  Number of spatial orbitals.
        n_dets:      Number of determinants.
        seed:        Random seed for reproducibility.
        max_excitation_order: Maximum excitation rank per spin channel.
        n_alpha:     Number of alpha electrons.
        n_beta:      Number of beta electrons.
        include_hf:  Whether to include the HF determinant.

    Returns:
        A normalised :class:`~qdk_chemistry.data.Wavefunction`.
    """

    det_matrix = generate_determinants_matrix(
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        n_dets=n_dets,
        seed=seed,
        max_excitation_order=max_excitation_order,
        n_alpha=n_alpha,
        n_beta=n_beta,
        include_hf=include_hf,
    )

    config_strings = determinants_to_config_strings(det_matrix, n_orbitals)
    dets = [Configuration(s) for s in config_strings]

    # Random normalised coefficients (offset seed to avoid correlation
    # with determinant generation)
    rng = np.random.default_rng(seed + 999)
    raw = rng.standard_normal(n_dets)
    coeffs = raw / np.linalg.norm(raw)

    # Identity MO coefficients — sufficient for state-preparation benchmarks
    orbitals = create_test_orbitals(n_orbitals)

    container = CasWavefunctionContainer(coeffs, dets, orbitals)
    return Wavefunction(container)
