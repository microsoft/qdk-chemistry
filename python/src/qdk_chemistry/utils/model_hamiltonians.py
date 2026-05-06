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
from qdk_chemistry.data import LatticeGraph, LayeredPartition, PauliOperator, QubitHamiltonian
from qdk_chemistry.utils import Logger

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


def _build_geometry_grouped_hamiltonian(
    graph: LatticeGraph,
    *,
    couplings: list[tuple[str, np.ndarray | float]],
    fields: list[tuple[str, np.ndarray | float]],
    coloring: dict[tuple[int, int], int] | None = None,
) -> QubitHamiltonian:
    r"""Assemble a Heisenberg-like Hamiltonian with a populated ``term_partition``.

    This helper exists separately from the ungrouped construction path
    because it builds the Pauli-string list in a specific order dictated
    by the lattice edge coloring, then records that order as a
    :class:`~qdk_chemistry.data.LayeredPartition`.  The ungrouped path
    constructs terms from the adjacency matrix directly without regard
    to color structure.

    Groups are organised first by single-body field direction (one group
    per direction, each containing a single layer because field terms
    have disjoint support), then by two-body coupling type (one group
    per ``XX``/``YY``/``ZZ`` block, each split into layers by edge
    color).  Term indices in
    :attr:`~qdk_chemistry.data.QubitHamiltonian.pauli_strings` align with the
    indices stored in the returned :class:`LayeredPartition`.

    Args:
        graph: Lattice graph defining connectivity.
        couplings: ``[(label, value), ...]`` for two-body terms (e.g. ``[(\"XX\", jx)]``).
        fields: ``[(char, value), ...]`` for single-body terms (e.g. ``[(\"X\", hx)]``).
        coloring: Optional edge coloring ``{(i, j): color}`` (``i < j``). Reads ``graph.edge_coloring`` when ``None``.

    Returns:
        QubitHamiltonian: The assembled Hamiltonian carrying a ``LayeredPartition``
        with ``strategy=\"geometry_coloring\"``.

    """
    n = graph.num_sites

    if coloring is None:
        coloring = graph.edge_coloring
    if coloring is None:
        raise ValueError(
            "No edge coloring available on the lattice graph. "
            "Use a factory method that provides one, or pass an explicit coloring."
        )

    pauli_strings: list[str] = []
    coefficients: list[complex] = []
    groups_layers: list[tuple[tuple[int, ...], ...]] = []

    # Field groups: one group per direction, single layer each.
    for pauli_char, field in fields:
        field_vec = to_site_param(field, graph, "field")
        layer_indices: list[int] = []
        for i in range(n):
            if field_vec[i] == 0.0:
                continue
            ps = ["I"] * n
            ps[i] = pauli_char
            pauli_strings.append("".join(ps[::-1]))
            coefficients.append(complex(field_vec[i]))
            layer_indices.append(len(pauli_strings) - 1)
        if layer_indices:
            groups_layers.append((tuple(layer_indices),))

    # Coupling groups: one group per (XX/YY/ZZ) block; layers given by edge colors.
    for pauli_label, coupling in couplings:
        coupling_mat = to_pair_param(coupling, graph, "coupling")
        color_to_indices: dict[int, list[int]] = {}
        for (i, j), c in coloring.items():
            edge_weight = graph.weight(i, j)
            if edge_weight == 0.0:
                continue
            coeff_val = coupling_mat[i, j] * edge_weight
            if coeff_val == 0.0:
                continue
            ps = ["I"] * n
            ps[i] = pauli_label[0]
            ps[j] = pauli_label[1]
            pauli_strings.append("".join(ps[::-1]))
            coefficients.append(complex(coeff_val))
            color_to_indices.setdefault(c, []).append(len(pauli_strings) - 1)
        if color_to_indices:
            layers = tuple(tuple(color_to_indices[c]) for c in sorted(color_to_indices))
            groups_layers.append(layers)

    if not pauli_strings:
        # Empty Hamiltonian: emit a single all-identity term with zero coefficient
        # so the resulting QubitHamiltonian remains constructible.
        pauli_strings = ["I" * n]
        coefficients = [0.0 + 0.0j]
        groups_layers = [((0,),)]

    partition = LayeredPartition(strategy="geometry_coloring", groups=tuple(groups_layers))
    return QubitHamiltonian(
        pauli_strings=pauli_strings,
        coefficients=np.array(coefficients),
        term_partition=partition,
    )


def create_heisenberg_hamiltonian(
    graph: LatticeGraph,
    jx: np.ndarray | float,
    jy: np.ndarray | float,
    jz: np.ndarray | float,
    hx: np.ndarray | float = 0.0,
    hy: np.ndarray | float = 0.0,
    hz: np.ndarray | float = 0.0,
    *,
    include_term_groups: bool = True,
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
        include_term_groups: When ``True`` (default), attach a geometry-coloring term partition to the result.

    Returns:
        QubitHamiltonian: The Heisenberg model as a qubit Hamiltonian; carries a ``LayeredPartition`` when grouped.

    """
    if not graph.is_symmetric:
        raise ValueError("Lattice graph must be symmetric for a valid Hamiltonian.")

    if include_term_groups:
        if graph.edge_coloring is not None:
            return _build_geometry_grouped_hamiltonian(
                graph,
                couplings=[("XX", jx), ("YY", jy), ("ZZ", jz)],
                fields=[("X", hx), ("Y", hy), ("Z", hz)],
            )
        Logger.debug("No edge coloring on lattice graph; falling back to ungrouped Hamiltonian construction.")

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
    *,
    include_term_groups: bool = True,
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
        include_term_groups: When ``True`` (default), attach a geometry-coloring term partition to the result.

    Returns:
        QubitHamiltonian: The Ising model as a qubit Hamiltonian.

    """
    return create_heisenberg_hamiltonian(graph, jx=0.0, jy=0.0, jz=j, hx=h, include_term_groups=include_term_groups)
