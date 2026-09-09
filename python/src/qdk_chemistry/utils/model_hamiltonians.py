"""Model Hamiltonian utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from array import array
from itertools import repeat

import numpy as np
import scipy.sparse

from qdk_chemistry._core.data import greedy_edge_coloring
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
from qdk_chemistry.data import LatticeGraph, LayeredPartition, QubitOperator
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


def _build_sparse_hamiltonian(
    graph: LatticeGraph,
    *,
    couplings: list[tuple[str, np.ndarray | float]],
    fields: list[tuple[str, np.ndarray | float]],
    coloring: dict[tuple[int, int], int] | None,
) -> QubitOperator:
    """Assemble equal-axis spin interactions without dense matrices or Pauli labels.

    Grouped terms retain the field-axis, coupling-axis, and lexicographic
    edge order of the geometry builder. Without a coloring, terms retain
    the expression builder's edge-major, then site-major order. Identical
    active coupling supports share one coloring; the full factory coloring
    is reused unchanged. Packed assembly storage is linear in sites and edges,
    apart from any matrices supplied by the caller.

    Args:
        graph: Lattice graph defining connectivity.
        couplings: Axis and scalar or matrix coupling for each equal-axis two-body block.
        fields: Axis and scalar or vector coefficient for each single-body block.
        coloring: Factory edge coloring, or ``None`` to leave terms ungrouped.

    Returns:
        QubitOperator: Packed operator with an optional :class:`~qdk_chemistry.data.LayeredPartition`.

    """
    n = graph.num_sites
    offsets = array("Q", [0])
    qubits = array("I")
    paulis = array("B")
    coefficients = array("d")
    pauli_code = {"X": 1, "Y": 2, "Z": 3}
    groups_layers: list[tuple[tuple[int, ...], ...]] = []

    def append_term(sites: tuple[int, ...], code: int, coefficient: float) -> int:
        qubits.extend(sites)
        paulis.extend([code] * len(sites))
        offsets.append(len(qubits))
        coefficients.append(coefficient)
        return len(coefficients) - 1

    adjacency = scipy.sparse.triu(graph.sparse_adjacency_matrix(), k=1, format="csr")
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    edges = adjacency.tocoo(copy=False)

    edge_couplings: list[tuple[int, np.ndarray]] = []
    pair_values: np.ndarray | float
    for axis, coupling in couplings:
        if isinstance(coupling, int | float | np.integer | np.floating):
            pair_values = float(coupling)
            if pair_values == 0.0:
                continue
        else:
            matrix = (
                coupling
                if isinstance(coupling, np.ndarray) and coupling.dtype.kind in "biuf" and coupling.shape == (n, n)
                else to_pair_param(coupling, graph, f"j{axis.lower()}")
            )
            # Select only edges, even for a broadcast n-by-n array.
            pair_values = matrix[edges.row, edges.col]
        # Cast before multiplying, preserving Python-double arithmetic and its nonfinite behavior.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            edge_couplings.append((pauli_code[axis], np.multiply(pair_values, edges.data, dtype=float)))

    field_values: list[tuple[int, np.ndarray | float]] = []
    for axis, field in fields:
        values = (
            float(field)
            if isinstance(field, int | float | np.integer | np.floating)
            else to_site_param(field, graph, f"h{axis.lower()}")
        )
        if isinstance(values, float) and values == 0.0:
            continue
        field_values.append((pauli_code[axis], values))

    if coloring is None:
        for index, (i, j) in enumerate(zip(edges.row, edges.col, strict=True)):
            edge = (int(i), int(j))
            for code, coupling_terms in edge_couplings:
                coefficient = float(coupling_terms[index])
                if coefficient != 0.0:
                    append_term(edge, code, coefficient)
        for site in range(n):
            for code, field in field_values:
                coefficient = field if isinstance(field, float) else float(field[site])
                if coefficient != 0.0:
                    append_term((site,), code, coefficient)
    else:
        for code, field in field_values:
            site_values = repeat(field, n) if isinstance(field, float) else field
            layer = tuple(
                append_term((site,), code, float(value)) for site, value in enumerate(site_values) if value != 0.0
            )
            if layer:
                groups_layers.append((layer,))

        coloring_cache = {np.arange(edges.nnz, dtype=np.intp).tobytes(): coloring}
        for code, coupling_terms in edge_couplings:
            active = np.flatnonzero(coupling_terms)
            if not active.size:
                continue
            support_key = active.tobytes()
            coupling_coloring = coloring_cache.get(support_key)
            if coupling_coloring is None:
                support = scipy.sparse.csr_matrix(
                    (np.ones(active.size), (edges.row[active], edges.col[active])), shape=(n, n)
                )
                coupling_coloring = greedy_edge_coloring(support + support.T, seed=0, trials=32)
                coloring_cache[support_key] = coupling_coloring
            color_to_indices: dict[int, list[int]] = {}
            for index in active:
                edge = (int(edges.row[index]), int(edges.col[index]))
                coefficient = float(coupling_terms[index])
                color_to_indices.setdefault(coupling_coloring[edge], []).append(append_term(edge, code, coefficient))
            groups_layers.append(tuple(tuple(color_to_indices[color]) for color in sorted(color_to_indices)))

    if not coefficients:
        append_term((), 0, 0.0)
        groups_layers = [((0,),)]

    partition = (
        LayeredPartition(strategy="geometry_coloring", groups=tuple(groups_layers)) if coloring is not None else None
    )
    return QubitOperator.from_sparse_arrays(
        n,
        np.frombuffer(offsets, dtype=np.uint64),
        np.frombuffer(qubits, dtype=np.uint32),
        np.frombuffer(paulis, dtype=np.uint8),
        np.asarray(coefficients, dtype=complex),
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
) -> QubitOperator:
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
        QubitOperator: The Heisenberg model as a qubit Hamiltonian; carries a ``LayeredPartition`` when grouped.

    """
    if not graph.is_symmetric:
        raise ValueError("Lattice graph must be symmetric for a valid Hamiltonian.")

    coloring = graph.edge_coloring if include_term_groups else None
    if include_term_groups and coloring is None:
        Logger.debug("No edge coloring on lattice graph; falling back to ungrouped Hamiltonian construction.")

    return _build_sparse_hamiltonian(
        graph,
        couplings=[("X", jx), ("Y", jy), ("Z", jz)],
        fields=[("X", hx), ("Y", hy), ("Z", hz)],
        coloring=coloring,
    )


def create_ising_hamiltonian(
    graph: LatticeGraph,
    j: np.ndarray | float,
    h: np.ndarray | float = 0.0,
    *,
    include_term_groups: bool = True,
) -> QubitOperator:
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
        QubitOperator: The Ising model as a qubit Hamiltonian.

    """
    return create_heisenberg_hamiltonian(graph, jx=0.0, jy=0.0, jz=j, hx=h, include_term_groups=include_term_groups)
