"""Model Hamiltonian utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Sequence

import numpy as np

from qdk_chemistry._core.utils.model_hamiltonians import (
    create_hubbard_hamiltonian,
    create_huckel_hamiltonian,
    create_ppp_hamiltonian,
    mataga_nishimoto_potential,
    ohno_potential,
    pairwise_potential,
    to_pair_param,
    to_site_param,
)
from qdk_chemistry.data import LatticeGraph, PauliOperator, QubitHamiltonian
from qdk_chemistry.utils.spin_operators import SpinEncoding

__all__ = [
    "create_heisenberg_hamiltonian",
    "create_hubbard_hamiltonian",
    "create_huckel_hamiltonian",
    "create_ising_hamiltonian",
    "create_ppp_hamiltonian",
    "mataga_nishimoto_potential",
    "ohno_potential",
    "pairwise_potential",
]


def _pauli_expr_to_qubit_hamiltonian(expr, num_qubits: int) -> QubitHamiltonian:
    """Convert a simplified PauliOperator expression to a QubitHamiltonian."""
    simplified = expr.simplify()
    terms = simplified.to_canonical_terms(num_qubits)
    pauli_strings = [t[1][::-1] for t in terms]
    coefficients = np.array([complex(t[0]) for t in terms])
    return QubitHamiltonian(pauli_strings, coefficients)


def create_heisenberg_hamiltonian(
    graph: LatticeGraph,
    jx: np.ndarray | float,
    jy: np.ndarray | float,
    jz: np.ndarray | float,
    hx: np.ndarray | float = 0.0,
    hy: np.ndarray | float = 0.0,
    hz: np.ndarray | float = 0.0,
    spins: float | Sequence[float] = 0.5,
    j_biquadratic: np.ndarray | float = 0.0,
    penalty_strength: float | None = None,
) -> QubitHamiltonian:
    r"""Create the anisotropic Heisenberg model Hamiltonian on a lattice.

    For spin-1/2 (default), the Hamiltonian is:

    .. math::

        H = \sum_{\langle i,j \rangle} w_{ij}\,\bigl[
                J_x^{ij}\,S_i^x S_j^x
              + J_y^{ij}\,S_i^y S_j^y
              + J_z^{ij}\,S_i^z S_j^z
            \bigr]
          + \sum_{\langle i,j \rangle} w_{ij}\,J_\mathrm{bq}^{ij}\,
            (\mathbf{S}_i \cdot \mathbf{S}_j)^2
          + \sum_i \bigl[
                h_x^{i}\,S_i^x + h_y^{i}\,S_i^y + h_z^{i}\,S_i^z
            \bigr]

    where :math:`w_{ij}` is the edge weight from the lattice adjacency matrix and
    :math:`S_i^\alpha` are spin-S operators with eigenvalues :math:`-S, -S+1, \ldots, S`.

    Each lattice site is encoded into :math:`\lceil \log_2(2S+1) \rceil` qubits.
    When :math:`2S+1` is not a power of 2, the encoding introduces unphysical
    computational basis states. Use ``penalty_strength`` to add an energy penalty
    that pushes these states out of the low-energy spectrum.

    .. note::

        This function uses physical spin operators :math:`S^\alpha` (not Pauli matrices
        :math:`\sigma^\alpha`). For spin-1/2, :math:`S^\alpha = \sigma^\alpha / 2`.

    Args:
        graph: Lattice graph defining the connectivity.
        jx: Coupling constant for :math:`S^x S^x` interactions. Scalar (uniform) or ``(n, n)`` array for per-pair values.
        jy: Coupling constant for :math:`S^y S^y` interactions (same format as *jx*).
        jz: Coupling constant for :math:`S^z S^z` interactions (same format as *jx*).
        hx: External magnetic field in the x direction. Scalar or length-n array. Defaults to 0.
        hy: External magnetic field in the y direction. Defaults to 0.
        hz: External magnetic field in the z direction. Defaults to 0.
        spins: Spin quantum number per site. Scalar for uniform spin, or a sequence of length ``n`` for mixed-spin lattices. Defaults to 0.5.
        j_biquadratic: Biquadratic coupling constant for :math:`(\mathbf{S}_i \cdot \mathbf{S}_j)^2`. Scalar or ``(n, n)`` array. Defaults to 0.
        penalty_strength: Energy penalty applied to unphysical qubit states. Required when any site has :math:`2S+1 \neq 2^k`. Set to ``None`` (default) for power-of-2 encodings.

    Returns:
        QubitHamiltonian: The Heisenberg model as a qubit Hamiltonian.

    Raises:
        ValueError: If any site has unphysical states and ``penalty_strength`` is not set.

    """
    if not graph.is_symmetric:
        raise ValueError("Lattice graph must be symmetric for a valid Hamiltonian.")

    n = graph.num_sites
    adj = graph.adjacency_matrix()

    jx_mat = to_pair_param(jx, graph, "jx")
    jy_mat = to_pair_param(jy, graph, "jy")
    jz_mat = to_pair_param(jz, graph, "jz")
    hx_vec = to_site_param(hx, graph, "hx")
    hy_vec = to_site_param(hy, graph, "hy")
    hz_vec = to_site_param(hz, graph, "hz")

    # Build qubit layout from spin assignments
    encoding = SpinEncoding(spins, num_sites=n)

    # Check for unphysical states
    if encoding.has_unphysical_states and penalty_strength is None:
        raise ValueError(
            "Some sites have 2S+1 ≠ 2^k, introducing unphysical qubit states. "
            "Set penalty_strength to a positive value to penalize these states, "
            "e.g. penalty_strength=10.0."
        )

    # Build site operators
    site_ops = [encoding.site_operators(i) for i in range(n)]

    h = 0.0 * PauliOperator.I(0)  # seed with zero expression

    # Parse biquadratic coupling
    has_biquadratic = not (isinstance(j_biquadratic, (int, float)) and j_biquadratic == 0.0)
    if has_biquadratic:
        jbq_mat = to_pair_param(j_biquadratic, graph, "j_biquadratic")

    # Two-body interaction terms (each edge counted once: i < j), scaled by the lattice edge weight.
    for i in range(n):
        for j in range(i + 1, n):
            edge_weight = adj[i, j]
            if edge_weight == 0.0:
                continue

            # Bilinear terms: Jα * Sᵢα * Sⱼα
            if jx_mat[i, j] != 0.0:
                h = h + jx_mat[i, j] * edge_weight * site_ops[i].sx * site_ops[j].sx
            if jy_mat[i, j] != 0.0:
                h = h + jy_mat[i, j] * edge_weight * site_ops[i].sy * site_ops[j].sy
            if jz_mat[i, j] != 0.0:
                h = h + jz_mat[i, j] * edge_weight * site_ops[i].sz * site_ops[j].sz

            # Biquadratic term: Jbq * (Sᵢ·Sⱼ)²
            if has_biquadratic and jbq_mat[i, j] != 0.0:
                dot_ij = (
                    site_ops[i].sx * site_ops[j].sx
                    + site_ops[i].sy * site_ops[j].sy
                    + site_ops[i].sz * site_ops[j].sz
                )
                h = h + jbq_mat[i, j] * edge_weight * dot_ij * dot_ij

    # Single-body field terms
    for i in range(n):
        if hx_vec[i] != 0.0:
            h = h + hx_vec[i] * site_ops[i].sx
        if hy_vec[i] != 0.0:
            h = h + hy_vec[i] * site_ops[i].sy
        if hz_vec[i] != 0.0:
            h = h + hz_vec[i] * site_ops[i].sz

    # Penalty terms for unphysical states
    if penalty_strength is not None and penalty_strength != 0.0:
        for i in range(n):
            pen = site_ops[i].penalty_projector()
            if pen is not None:
                h = h + penalty_strength * pen

    return _pauli_expr_to_qubit_hamiltonian(h, encoding.total_qubits)


def create_ising_hamiltonian(
    graph: LatticeGraph,
    j: np.ndarray | float,
    h: np.ndarray | float = 0.0,
    spins: float | Sequence[float] = 0.5,
    penalty_strength: float | None = None,
) -> QubitHamiltonian:
    r"""Create the Ising model Hamiltonian on a lattice.

    .. math::

        H = \sum_{\langle i,j \rangle} w_{ij}\,J^{ij}\,S_i^z S_j^z
          + \sum_i h^{i}\,S_i^x

    where :math:`w_{ij}` is the edge weight from the lattice adjacency matrix and
    :math:`S_i^\alpha` are spin-S operators.

    This is a special case of the Heisenberg model with only :math:`S^z S^z`
    coupling and a transverse :math:`S^x` field.

    Args:
        graph: Lattice graph defining the connectivity.
        j: Coupling constant for :math:`S^z S^z` interactions. Scalar or ``(n, n)`` array.
        h: Transverse field strength (x direction). Scalar or length-n array.  Defaults to 0.
        spins: Spin quantum number per site. Scalar or sequence. Defaults to 0.5.
        penalty_strength: Energy penalty for unphysical qubit states. See :func:`create_heisenberg_hamiltonian`.

    Returns:
        QubitHamiltonian: The Ising model as a qubit Hamiltonian.

    """
    return create_heisenberg_hamiltonian(
        graph, jx=0.0, jy=0.0, jz=j, hx=h, spins=spins, penalty_strength=penalty_strength
    )
