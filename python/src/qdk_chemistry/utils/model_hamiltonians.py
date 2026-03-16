"""Model Hamiltonian utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

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


def _pauli_expr_to_qubit_hamiltonian(expr) -> QubitHamiltonian:
    """Convert a simplified PauliOperator expression to a QubitHamiltonian."""
    simplified = expr.simplify()
    n_qubits = simplified.num_qubits()
    terms = simplified.to_canonical_terms(n_qubits)
    pauli_strings = [t[1] for t in terms]
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
) -> QubitHamiltonian:
    r"""Create the anisotropic Heisenberg model Hamiltonian on a lattice.

    .. math::

        H = \sum_{\langle i,j \rangle} w_{ij}\,\bigl[
                J_x^{ij}\,\sigma_i^x \sigma_j^x
              + J_y^{ij}\,\sigma_i^y \sigma_j^y
              + J_z^{ij}\,\sigma_i^z \sigma_j^z
            \bigr]
          + \sum_i \bigl[
                h_x^{i}\,\sigma_i^x
              + h_y^{i}\,\sigma_i^y
              + h_z^{i}\,\sigma_i^z
            \bigr]

    where :math:`w_{ij}` is the edge weight from the lattice adjacency matrix.

    Each qubit corresponds to a lattice site.

    Args:
        graph: Lattice graph defining the connectivity.
        jx: Coupling constant for XX interactions. Scalar (uniform) or ``(n, n)`` array for per-pair values.
        jy: Coupling constant for YY interactions (same format as *jx*).
        jz: Coupling constant for ZZ interactions (same format as *jx*).
        hx: External magnetic field in the x direction. Scalar or length-n array. Defaults to 0.
        hy: External magnetic field in the y direction. Defaults to 0.
        hz: External magnetic field in the z direction. Defaults to 0.

    Returns:
        QubitHamiltonian: The Heisenberg model as a qubit Hamiltonian.

    """
    n = graph.num_sites
    adj = graph.adjacency_matrix()

    jx_mat = to_pair_param(jx, graph, "jx")
    jy_mat = to_pair_param(jy, graph, "jy")
    jz_mat = to_pair_param(jz, graph, "jz")
    hx_vec = to_site_param(hx, graph, "hx")
    hy_vec = to_site_param(hy, graph, "hy")
    hz_vec = to_site_param(hz, graph, "hz")

    h = 0.0 * PauliOperator.I(0)  # seed with zero expression

    # Two-body interaction terms (each edge counted once: i < j), scaled by the lattice edge weight.
    for i in range(n):
        for j in range(i + 1, n):
            edge_weight = adj[i, j]
            if edge_weight == 0.0:
                continue
            if jx_mat[i, j] != 0.0:
                h = h + jx_mat[i, j] * edge_weight * PauliOperator.X(i) * PauliOperator.X(j)
            if jy_mat[i, j] != 0.0:
                h = h + jy_mat[i, j] * edge_weight * PauliOperator.Y(i) * PauliOperator.Y(j)
            if jz_mat[i, j] != 0.0:
                h = h + jz_mat[i, j] * edge_weight * PauliOperator.Z(i) * PauliOperator.Z(j)

    # Single-body field terms
    for i in range(n):
        if hx_vec[i] != 0.0:
            h = h + hx_vec[i] * PauliOperator.X(i)
        if hy_vec[i] != 0.0:
            h = h + hy_vec[i] * PauliOperator.Y(i)
        if hz_vec[i] != 0.0:
            h = h + hz_vec[i] * PauliOperator.Z(i)

    return _pauli_expr_to_qubit_hamiltonian(h)


def create_ising_hamiltonian(
    graph: LatticeGraph,
    j: np.ndarray | float,
    h: np.ndarray | float = 0.0,
) -> QubitHamiltonian:
    r"""Create the Ising model Hamiltonian on a lattice.

    .. math::

        H = \sum_{\langle i,j \rangle} J^{ij}\,\sigma_i^z \sigma_j^z + \sum_i h^{i}\,\sigma_i^x

    Args:
        graph: Lattice graph defining the connectivity.
        j: Coupling constant for ZZ interactions. Scalar or ``(n, n)`` array.
        h: Transverse field strength (x direction). Scalar or length-n array.  Defaults to 0.

    Returns:
        QubitHamiltonian: The Ising model as a qubit Hamiltonian.

    """
    return create_heisenberg_hamiltonian(graph, jx=0.0, jy=0.0, jz=j, hx=h)
