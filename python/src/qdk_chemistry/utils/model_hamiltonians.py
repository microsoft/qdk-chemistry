"""Model Hamiltonian utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Iterable, Mapping
from numbers import Integral

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


def _build_geometry_grouped_hamiltonian(
    graph: LatticeGraph,
    *,
    couplings: list[tuple[str, np.ndarray | float]],
    fields: list[tuple[str, np.ndarray | float]],
    coloring: dict[tuple[int, int], int] | None = None,
) -> QubitOperator:
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
    :attr:`~qdk_chemistry.data.QubitOperator.pauli_strings` align with the
    indices stored in the returned :class:`LayeredPartition`.

    Args:
        graph: Lattice graph defining connectivity.
        couplings: ``[(label, value), ...]`` for two-body terms (e.g. ``[(\"XX\", jx)]``).
        fields: ``[(char, value), ...]`` for single-body terms (e.g. ``[(\"X\", hx)]``).
        coloring: Optional edge coloring ``{(i, j): color}`` (``i < j``). Reads ``graph.edge_coloring`` when ``None``.

    Returns:
        QubitOperator: The assembled Hamiltonian carrying a ``LayeredPartition``
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
        # so the resulting QubitOperator remains constructible.
        pauli_strings = ["I" * n]
        coefficients = [0.0 + 0.0j]
        groups_layers = [((0,),)]

    partition = LayeredPartition(strategy="geometry_coloring", groups=tuple(groups_layers))
    return QubitOperator(
        pauli_strings=pauli_strings,
        coefficients=np.array(coefficients),
        term_partition=partition,
    )


def create_heisenberg_hamiltonian(
    graph: LatticeGraph,
    jx: np.ndarray | float | Mapping[int, np.ndarray | float],
    jy: np.ndarray | float | Mapping[int, np.ndarray | float],
    jz: np.ndarray | float | Mapping[int, np.ndarray | float],
    hx: np.ndarray | float = 0.0,
    hy: np.ndarray | float = 0.0,
    hz: np.ndarray | float = 0.0,
    *,
    include_term_groups: bool = True,
) -> QubitOperator:
    r"""Create the anisotropic Heisenberg model Hamiltonian on a lattice.

    .. math::

                H = \sum_{i<j} \bigl[
                                K_x^{ij}\,\sigma_i^x \sigma_j^x
                            + K_y^{ij}\,\sigma_i^y \sigma_j^y
                            + K_z^{ij}\,\sigma_i^z \sigma_j^z
                        \bigr]
          + \sum_i \bigl[
                h_x^{i}\,\sigma_i^x
              + h_y^{i}\,\sigma_i^y
              + h_z^{i}\,\sigma_i^z
            \bigr]

    For scalar and array couplings, :math:`K_a^{ij}=w_{ij}J_a^{ij}` on
    adjacency edges. A mapping from one-based geometric shell indices to
    couplings instead defines :math:`K_a^{ij}` on those shells independently
    of adjacency weights.

    Each qubit corresponds to a lattice site.

    Args:
        graph: Lattice graph defining the connectivity.
        jx: XX coupling as a scalar, ``(n, n)`` array, or ``{m: coupling}`` geometric-shell mapping.
        jy: Coupling constant for YY interactions (same format as *jx*).
        jz: Coupling constant for ZZ interactions (same format as *jx*).
        hx: External magnetic field in the x direction. Scalar or length-n array. Defaults to 0.
        hy: External magnetic field in the y direction. Defaults to 0.
        hz: External magnetic field in the z direction. Defaults to 0.
        include_term_groups: Attach a term partition for adjacency-based couplings when ``True``. Defaults to ``True``.

    Returns:
        QubitOperator: The Heisenberg model as a qubit Hamiltonian; carries a ``LayeredPartition`` when grouped.

    """
    if not graph.is_symmetric:
        raise ValueError("Lattice graph must be symmetric for a valid Hamiltonian.")

    shell_couplings = any(isinstance(coupling, Mapping) for coupling in (jx, jy, jz))
    if include_term_groups and not shell_couplings:
        if graph.edge_coloring is not None:
            return _build_geometry_grouped_hamiltonian(
                graph,
                couplings=[("XX", jx), ("YY", jy), ("ZZ", jz)],
                fields=[("X", hx), ("Y", hy), ("Z", hz)],
            )
        Logger.debug("No edge coloring on lattice graph; falling back to ungrouped Hamiltonian construction.")
    elif include_term_groups:
        Logger.debug("Geometric shell couplings use ungrouped Hamiltonian construction.")

    n = graph.num_sites
    hx_vec = to_site_param(hx, graph, "hx")
    hy_vec = to_site_param(hy, graph, "hy")
    hz_vec = to_site_param(hz, graph, "hz")
    pauli_strings: list[str] = []
    coefficients: list[complex] = []

    def pair_parameter(value: np.ndarray | float, name: str) -> np.ndarray | float:
        if isinstance(value, int | float | np.integer | np.floating):
            return float(value)
        return to_pair_param(value, graph, name)

    def append_pair_terms(
        pauli_char: str,
        pairs: Iterable[tuple[int, int]],
        values: np.ndarray | float,
        weights: np.ndarray | None = None,
    ) -> None:
        for i, j in pairs:
            coefficient = values if isinstance(values, float) else values[i, j]
            if weights is not None:
                coefficient *= weights[i, j]
            if coefficient == 0.0:
                continue
            pauli = ["I"] * n
            pauli[i] = pauli[j] = pauli_char
            pauli_strings.append("".join(pauli[::-1]))
            coefficients.append(complex(coefficient))

    coupling_specs = [("X", "jx", jx), ("Y", "jy", jy), ("Z", "jz", jz)]
    if not shell_couplings:
        adjacency = graph.adjacency_matrix()
        coupling_matrices = [to_pair_param(coupling, graph, name) for _, name, coupling in coupling_specs]
        for i in range(n):
            for j in range(i + 1, n):
                edge_weight = adjacency[i, j]
                if edge_weight == 0.0:
                    continue
                for (pauli_char, _, _), coupling_matrix in zip(coupling_specs, coupling_matrices, strict=True):
                    coefficient = coupling_matrix[i, j] * edge_weight
                    if coefficient == 0.0:
                        continue
                    pauli = ["I"] * n
                    pauli[i] = pauli[j] = pauli_char
                    pauli_strings.append("".join(pauli[::-1]))
                    coefficients.append(complex(coefficient))
    else:
        normalized_shell_couplings: dict[str, list[tuple[int, np.ndarray | float]]] = {}
        requested_shells: set[int] = set()
        for _, name, coupling in coupling_specs:
            if not isinstance(coupling, Mapping):
                continue

            normalized_mapping: list[tuple[int, np.ndarray | float]] = []
            for shell, shell_coupling in coupling.items():
                if isinstance(shell, bool) or not isinstance(shell, Integral) or shell < 1:
                    raise ValueError(f"{name} shell indices must be positive integers; got {shell!r}.")
                shell_index = int(shell)
                normalized_mapping.append((shell_index, shell_coupling))
                requested_shells.add(shell_index)
            normalized_mapping.sort(key=lambda item: item[0])
            normalized_shell_couplings[name] = normalized_mapping

        shell_pairs = graph.nearest_neighbor_shells(sorted(requested_shells)) if requested_shells else {}
        adjacency = graph.adjacency_matrix() if len(normalized_shell_couplings) != len(coupling_specs) else None

        for pauli_char, name, coupling in coupling_specs:
            if name in normalized_shell_couplings:
                for shell_index, shell_coupling in normalized_shell_couplings[name]:
                    values = pair_parameter(shell_coupling, f"{name}[{shell_index}]")
                    append_pair_terms(pauli_char, shell_pairs[shell_index], values)
            else:
                assert adjacency is not None
                pairs = ((i, j) for i in range(n) for j in range(i + 1, n) if adjacency[i, j] != 0.0)
                append_pair_terms(pauli_char, pairs, pair_parameter(coupling, name), adjacency)

    for i in range(n):
        for pauli_char, field in [("X", hx_vec), ("Y", hy_vec), ("Z", hz_vec)]:
            if field[i] == 0.0:
                continue
            pauli = ["I"] * n
            pauli[i] = pauli_char
            pauli_strings.append("".join(pauli[::-1]))
            coefficients.append(complex(field[i]))

    if not pauli_strings:
        pauli_strings = ["I" * n]
        coefficients = [0.0 + 0.0j]
    return QubitOperator(pauli_strings, np.array(coefficients))


def create_ising_hamiltonian(
    graph: LatticeGraph,
    j: np.ndarray | float | Mapping[int, np.ndarray | float],
    h: np.ndarray | float = 0.0,
    *,
    include_term_groups: bool = True,
) -> QubitOperator:
    r"""Create the Ising model Hamiltonian on a lattice.

    .. math::

        H = \sum_{\langle i,j \rangle} w_{ij}\,J^{ij}\,\sigma_i^z \sigma_j^z
          + \sum_i h^{i}\,\sigma_i^x

    Scalar and array couplings use adjacency edges and their weights. A mapping
    ``{m: coupling}`` instead applies couplings to geometric neighbor shells
    independently of adjacency weights.

    Args:
        graph: Lattice graph defining the connectivity.
        j: ZZ coupling as a scalar, ``(n, n)`` array, or ``{m: coupling}`` geometric-shell mapping.
        h: Transverse field strength (x direction). Scalar or length-n array.  Defaults to 0.
        include_term_groups: When ``True`` (default), attach a geometry-coloring term partition to the result.

    Returns:
        QubitOperator: The Ising model as a qubit Hamiltonian.

    """
    return create_heisenberg_hamiltonian(graph, jx=0.0, jy=0.0, jz=j, hx=h, include_term_groups=include_term_groups)
