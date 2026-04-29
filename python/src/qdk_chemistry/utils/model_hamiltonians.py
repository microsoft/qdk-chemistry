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
from qdk_chemistry.geometry.hypergraph import Hypergraph, HypergraphEdgeColoring

__all__ = [
    "create_heisenberg_hamiltonian",
    "create_hubbard_hamiltonian",
    "create_huckel_hamiltonian",
    "create_ising_hamiltonian",
    "create_ppp_hamiltonian",
    "heisenberg_term_groups",
    "ising_term_groups",
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

    return _pauli_expr_to_qubit_hamiltonian(h, n)


def create_ising_hamiltonian(
    graph: LatticeGraph,
    j: np.ndarray | float,
    h: np.ndarray | float = 0.0,
) -> QubitHamiltonian:
    r"""Create the Ising model Hamiltonian on a lattice.

    .. math::

        H = \sum_{\langle i,j \rangle} w_{ij}\,J^{ij}\,\sigma_i^z \sigma_j^z
          + \sum_i h^{i}\,\sigma_i^x

    where :math:`w_{ij}` is the edge weight from the lattice adjacency matrix.

    Args:
        graph: Lattice graph defining the connectivity.
        j: Coupling constant for ZZ interactions. Scalar or ``(n, n)`` array.
        h: Transverse field strength (x direction). Scalar or length-n array.  Defaults to 0.

    Returns:
        QubitHamiltonian: The Ising model as a qubit Hamiltonian.

    """
    return create_heisenberg_hamiltonian(graph, jx=0.0, jy=0.0, jz=j, hx=h)


def _build_edge_groups(
    graph: LatticeGraph,
    coloring: HypergraphEdgeColoring,
    pauli_label: str,
    coupling: np.ndarray | float,
) -> list[QubitHamiltonian]:
    """Build parallelizable layers for one interaction type using edge coloring.

    Each color in the coloring corresponds to a set of edges with disjoint
    vertex sets.  Operators on those edges can be executed in parallel.

    Args:
        graph: Lattice graph defining the connectivity.
        coloring: A :class:`~qdk_chemistry.geometry.HypergraphEdgeColoring`.
        pauli_label: Two-character Pauli label (e.g., ``"XX"``).
        coupling: Scalar or ``(n, n)`` coupling matrix.

    Returns:
        A list of :class:`QubitHamiltonian`, one per color (parallel layer).

    """
    assert isinstance(coloring, HypergraphEdgeColoring)

    n = graph.num_sites
    adj = graph.adjacency_matrix()
    coupling_mat = to_pair_param(coupling, graph, "coupling")

    # Collect operators per color
    color_terms: dict[int, tuple[list[str], list[complex]]] = {}

    for edge in coloring.hypergraph.edges():
        verts = edge.vertices
        if len(verts) != 2:
            continue
        i, j = verts
        edge_weight = adj[i, j]
        if edge_weight == 0.0:
            continue
        coeff_val = coupling_mat[i, j] * edge_weight
        if coeff_val == 0.0:
            continue
        c = coloring.color(verts)
        if c is None or c < 0:
            continue

        # Build Pauli string: each character acts on the corresponding qubit
        ps = ["I"] * n
        ps[i] = pauli_label[0]
        ps[j] = pauli_label[1]
        ps_str = "".join(ps[::-1])  # little-endian: rightmost = qubit 0

        if c not in color_terms:
            color_terms[c] = ([], [])
        color_terms[c][0].append(ps_str)
        color_terms[c][1].append(coeff_val)

    layers: list[QubitHamiltonian] = []
    for c in sorted(color_terms):
        labels, coeffs = color_terms[c]
        layers.append(QubitHamiltonian(labels, np.array(coeffs)))
    return layers


def _build_field_group(
    graph: LatticeGraph,
    pauli_char: str,
    field: np.ndarray | float,
) -> list[QubitHamiltonian]:
    """Build a single-layer group for single-site field terms.

    All single-site field terms commute and have disjoint qubit support,
    so they form a single parallelizable layer.

    Args:
        graph: Lattice graph defining the number of sites.
        pauli_char: Single Pauli character (``"X"``, ``"Y"``, or ``"Z"``).
        field: Scalar or length-n field array.

    Returns:
        A list containing one :class:`QubitHamiltonian` (or empty if all zero).

    """
    n = graph.num_sites
    field_vec = to_site_param(field, graph, "field")

    labels: list[str] = []
    coeffs: list[complex] = []
    for i in range(n):
        if field_vec[i] == 0.0:
            continue
        ps = ["I"] * n
        ps[i] = pauli_char
        labels.append("".join(ps[::-1]))
        coeffs.append(field_vec[i])

    if not labels:
        return []
    return [QubitHamiltonian(labels, np.array(coeffs))]


def heisenberg_term_groups(
    graph: LatticeGraph,
    jx: np.ndarray | float,
    jy: np.ndarray | float,
    jz: np.ndarray | float,
    hx: np.ndarray | float = 0.0,
    hy: np.ndarray | float = 0.0,
    hz: np.ndarray | float = 0.0,
    geometry: Hypergraph | None = None,
) -> list[list[QubitHamiltonian]]:
    r"""Compute geometry-aware Trotter term groups for a Heisenberg Hamiltonian.

    Uses edge coloring of the lattice geometry to partition interaction
    terms into parallelizable layers. Each interaction type (XX, YY, ZZ)
    becomes a separate Trotter term group, and within each group, edges
    of the same color have disjoint qubit supports and form a single
    parallelizable layer.

    The resulting structure is ``list[list[QubitHamiltonian]]`` suitable
    for passing to :class:`~qdk_chemistry.algorithms.time_evolution.builder.trotter.Trotter`
    as ``term_groups``.

    Args:
        graph: Lattice graph defining the connectivity.
        jx: Coupling constant for XX interactions.
        jy: Coupling constant for YY interactions.
        jz: Coupling constant for ZZ interactions.
        hx: External magnetic field in x direction. Defaults to 0.
        hy: External magnetic field in y direction. Defaults to 0.
        hz: External magnetic field in z direction. Defaults to 0.
        geometry: Lattice geometry for edge coloring. If ``None``, built from the graph via greedy coloring.

    Returns:
        Term groups for use with the Trotter builder's ``term_groups`` parameter.

    """
    if geometry is None:
        geometry = Hypergraph.from_lattice_graph(graph)

    coloring = geometry.edge_coloring()

    field_groups: list[list[QubitHamiltonian]] = []
    coupling_groups: list[list[QubitHamiltonian]] = []

    # Each interaction type is a separate Trotter term group.
    # Within each, the edge coloring gives parallelizable layers.
    for pauli_label, coupling_val in [("XX", jx), ("YY", jy), ("ZZ", jz)]:
        layers = _build_edge_groups(graph, coloring, pauli_label, coupling_val)
        if layers:
            coupling_groups.append(layers)

    # Field terms: each direction is a group with a single layer
    for pauli_char, field in [("X", hx), ("Y", hy), ("Z", hz)]:
        field_layers = _build_field_group(graph, pauli_char, field)
        if field_layers:
            field_groups.append(field_layers)

    # Order: field groups first (outer in Strang), coupling groups last
    # (middle in Strang). This maximises merging at Suzuki boundaries
    # because the outermost group is repeated symmetrically and adjacent
    # identical terms are combined.
    groups: list[list[QubitHamiltonian]] = field_groups + coupling_groups

    return groups


def ising_term_groups(
    graph: LatticeGraph,
    j: np.ndarray | float,
    h: np.ndarray | float = 0.0,
    geometry: Hypergraph | None = None,
) -> list[list[QubitHamiltonian]]:
    r"""Compute geometry-aware Trotter term groups for an Ising Hamiltonian.

    Equivalent to :func:`heisenberg_term_groups` with ``jx=0, jy=0, jz=j, hx=h``.

    Args:
        graph: Lattice graph defining the connectivity.
        j: Coupling constant for ZZ interactions.
        h: Transverse field strength (x direction). Defaults to 0.
        geometry: Lattice geometry for edge coloring. If ``None``, built from the graph.

    Returns:
        Term groups for use with the Trotter builder's ``term_groups`` parameter.

    """
    return heisenberg_term_groups(graph, jx=0.0, jy=0.0, jz=j, hx=h, geometry=geometry)
