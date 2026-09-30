"""Model Hamiltonian utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from array import array
from collections.abc import Mapping
from enum import IntEnum
from numbers import Integral

import numpy as np
import scipy.sparse

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
from qdk_chemistry.data import (
    BondFlavorDefinition,
    LatticeGraph,
    LayeredPartition,
    QubitOperator,
)
from qdk_chemistry.utils import Logger

__all__ = [
    "KitaevBondFlavor",
    "create_heisenberg_hamiltonian",
    "create_hubbard_hamiltonian",
    "create_huckel_hamiltonian",
    "create_ising_hamiltonian",
    "create_kitaev_hamiltonian",
    "create_ppp_hamiltonian",
    "kitaev_honeycomb_bond_flavors",
    "mataga_nishimoto_potential",
    "ohno_potential",
    "pairwise_potential",
]


class KitaevBondFlavor(IntEnum):
    """Spin component selected by a Kitaev bond."""

    X = 0
    Y = 1
    Z = 2


def kitaev_honeycomb_bond_flavors() -> list[BondFlavorDefinition]:
    """Return the standard honeycomb shell-axis flavor mapping."""
    root_three = np.sqrt(3.0)
    return [
        BondFlavorDefinition(1, np.array([0.5, root_three / 2]), KitaevBondFlavor.X),
        BondFlavorDefinition(1, np.array([0.5, -root_three / 2]), KitaevBondFlavor.Y),
        BondFlavorDefinition(1, np.array([1.0, 0.0]), KitaevBondFlavor.Z),
        BondFlavorDefinition(2, np.array([1.5, -root_three / 2]), KitaevBondFlavor.X),
        BondFlavorDefinition(2, np.array([1.5, root_three / 2]), KitaevBondFlavor.Y),
        BondFlavorDefinition(2, np.array([0.0, root_three]), KitaevBondFlavor.Z),
        BondFlavorDefinition(3, np.array([1.0, root_three]), KitaevBondFlavor.X),
        BondFlavorDefinition(3, np.array([1.0, -root_three]), KitaevBondFlavor.Y),
        BondFlavorDefinition(3, np.array([2.0, 0.0]), KitaevBondFlavor.Z),
    ]


def _pair_parameter(value: np.ndarray | float, graph: LatticeGraph, name: str) -> np.ndarray | float:
    """Validate supplied matrices without expanding scalar couplings."""
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    return to_pair_param(value, graph, name)


def _shell_couplings(
    coupling: np.ndarray | float | Mapping[int, np.ndarray | float], graph: LatticeGraph, name: str
) -> dict[int, np.ndarray | float]:
    """Validate a ``{m: coupling}`` shell mapping; a scalar or array coupling ``J`` means ``{1: J}``."""
    if not isinstance(coupling, Mapping):
        return {1: _pair_parameter(coupling, graph, name)}
    normalized: dict[int, np.ndarray | float] = {}
    for shell, value in coupling.items():
        if isinstance(shell, bool) or not isinstance(shell, Integral) or shell < 1:
            raise ValueError(f"{name} shell indices must be positive integers; got {shell!r}.")
        normalized[int(shell)] = _pair_parameter(value, graph, f"{name}[{int(shell)}]")
    return normalized


def _edge_value(value: np.ndarray | float, pair: tuple[int, int], weight: float) -> float:
    """Return a scalar or per-pair coupling on one edge, multiplied by the edge weight."""
    return float((value if isinstance(value, float) else value[pair]) * weight)


def _selected_edges(graph: LatticeGraph, shells: set[int]) -> list[tuple[tuple[int, int], int, int | None, float]]:
    """Return sorted ``(pair, shell, flavor, weight)`` records for weighted edges in the active shells.

    Every edge of a graph without edge labels, such as a factory lattice, is an unflavored shell-1 edge.
    The graph's edges are used as stored; none are discovered or added.

    """
    if not shells:
        return []
    labels = graph.edge_labels
    missing_shells = shells - ({label.shell for label in labels.values()} if labels else {1})
    if missing_shells:
        if not labels and graph.num_nonzeros != 0:
            raise ValueError(
                f"Neighbor shells {sorted(missing_shells)} require edge-label metadata; "
                "build the graph with LatticeGraph.from_geometry or pass edge_labels."
            )
        raise ValueError(
            f"Requested neighbor shells {sorted(missing_shells)} are not selected in the lattice graph. "
            "Select them with LatticeGraph.from_geometry or edge_labels before constructing the Hamiltonian."
        )
    adjacency = scipy.sparse.triu(graph.sparse_adjacency_matrix(), k=1).tocoo()
    edges = []
    for site_i, site_j, weight in sorted(
        zip(adjacency.row.tolist(), adjacency.col.tolist(), adjacency.data.tolist(), strict=True)
    ):
        label = labels.get((site_i, site_j))
        shell, flavor = (1, None) if label is None else (label.shell, label.flavor)
        if weight != 0.0 and shell in shells:
            edges.append(((site_i, site_j), shell, flavor, weight))
    return edges


def _build_sparse_hamiltonian(
    graph: LatticeGraph,
    *,
    couplings: list[tuple[str, dict[tuple[int, int], float]]],
    fields: list[tuple[str, np.ndarray | float]],
    grouped: bool,
    sparse_output: bool,
) -> QubitOperator:
    """Assemble sparse words, preserving the model's grouped or ungrouped ordering.

    Grouped families restrict the graph's stored coloring to nonzero coefficients,
    ordered by color and then pair; they never recolor their individual supports.
    Same-axis fields and interactions share a commuting group, with fields in
    their own disjoint layer. This merge changes metadata, not Pauli term order.
    Ungrouped terms use lexicographic pairs before coupling families.

    """
    n = graph.num_sites
    field_values = [(pauli, to_site_param(field, graph, "field")) for pauli, field in fields]
    words: list[tuple[tuple[int, str], ...]] = []
    coefficients = array("d")
    groups_layers: list[tuple[tuple[int, ...], ...]] = []

    def append_term(factors: tuple[tuple[int, str], ...], coefficient: float) -> int:
        words.append(tuple(sorted(factors)))
        coefficients.append(coefficient)
        return len(coefficients) - 1

    if grouped:
        coloring = graph.edge_coloring
        assert coloring is not None
        colored_pairs = sorted(coloring.items(), key=lambda item: (item[1], item[0]))
        field_groups: dict[str, int] = {}
        for pauli, values in field_values:
            layer = tuple(
                append_term(((site, pauli),), float(value)) for site, value in enumerate(values) if value != 0.0
            )
            if layer:
                field_groups[pauli] = len(groups_layers)
                groups_layers.append((layer,))

        for label, coeff_by_pair in couplings:
            if not coeff_by_pair:
                continue
            color_to_indices: dict[int, list[int]] = {}
            for pair, color in colored_pairs:
                coefficient = coeff_by_pair.get(pair, 0.0)
                if coefficient != 0.0:
                    color_to_indices.setdefault(color, []).append(
                        append_term(((pair[0], label[0]), (pair[1], label[1])), coefficient)
                    )
            if color_to_indices:
                layers = tuple(tuple(color_to_indices[color]) for color in sorted(color_to_indices))
                # Equal-axis terms commute across layers; mixed-axis terms need disjoint groups.
                if label[0] == label[1]:
                    field_group = field_groups.get(label[0])
                    if field_group is None:
                        groups_layers.append(layers)
                    else:
                        groups_layers[field_group] += layers
                else:
                    groups_layers.extend((layer,) for layer in layers)
    else:
        pairs = sorted({pair for _, coupling in couplings for pair, value in coupling.items() if value != 0.0})
        terms = ((label, pair, coupling.get(pair, 0.0)) for pair in pairs for label, coupling in couplings)
        for label, pair, coefficient in terms:
            if coefficient != 0.0:
                append_term(((pair[0], label[0]), (pair[1], label[1])), coefficient)
        for site in range(n):
            for pauli, values in field_values:
                if values[site] != 0.0:
                    append_term(((site, pauli),), float(values[site]))

    if not coefficients:
        append_term((), 0.0)
        groups_layers = [((0,),)]
    partition = LayeredPartition(strategy="geometry_coloring", groups=tuple(groups_layers)) if grouped else None
    operator = QubitOperator.from_sparse_terms(
        n,
        words,
        np.asarray(coefficients, dtype=complex),
        term_partition=partition,
    )
    # Terms are accumulated once as sparse words; dense output converts them to register-width labels.
    if not sparse_output:
        return QubitOperator(
            list(operator.pauli_strings),
            operator.coefficients,
            term_partition=partition,
        )
    return operator


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
    sparse_terms: bool = False,
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

    A mapping ``{m: J_m}`` from one-based geometric shells to couplings gives
    :math:`K_a^{ij}=w_{ij}J_{a,m}^{ij}` on the graph's shell-``m`` edges, where
    :math:`w_{ij}` is the edge weight. A scalar or array coupling ``J`` is the same
    as ``{1: J}``, and every edge of a graph without edge labels, such as a factory
    lattice, is a shell-1 edge. Select other shells with
    :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` or custom edge labels before
    constructing the Hamiltonian; the builder never discovers or adds edges. Empty or
    all-zero mappings require no edge labels.

    Each qubit corresponds to a lattice site.

    Args:
        graph: Lattice graph defining the connectivity.
        jx: XX coupling as a scalar, ``(n, n)`` array, or ``{m: coupling}`` geometric-shell mapping.
        jy: Coupling constant for YY interactions (same format as *jx*).
        jz: Coupling constant for ZZ interactions (same format as *jx*).
        hx: External magnetic field in the x direction. Scalar or length-n array. Defaults to 0.
        hy: External magnetic field in the y direction. Defaults to 0.
        hz: External magnetic field in the z direction. Defaults to 0.
        include_term_groups: Attach a geometry-coloring term partition when ``True``. Defaults to ``True``.
        sparse_terms: Store terms by their non-identity factors instead of register-width labels. Defaults to ``False``.

    Returns:
        QubitOperator: The Heisenberg model as a qubit Hamiltonian; carries a ``LayeredPartition`` when grouped.

    Raises:
        ValueError: If the graph is asymmetric, a shell index is invalid, or an active shell has no labelled edges.

    """
    if not graph.is_symmetric:
        raise ValueError("Lattice graph must be symmetric for a valid Hamiltonian.")

    families = (("XX", "jx", jx), ("YY", "jy", jy), ("ZZ", "jz", jz))
    shell_couplings = {name: _shell_couplings(coupling, graph, name) for _, name, coupling in families}
    requested_shells = {
        shell for values in shell_couplings.values() for shell, value in values.items() if np.any(value != 0.0)
    }
    edges = _selected_edges(graph, requested_shells)
    couplings: list[tuple[str, dict[tuple[int, int], float]]] = []
    for label, name, _ in families:
        records: dict[tuple[int, int], float] = {}
        for pair, shell, _, weight in edges:
            coefficient = _edge_value(shell_couplings[name].get(shell, 0.0), pair, weight)
            if coefficient != 0.0:
                records[pair] = coefficient
        couplings.append((label, records))

    grouped = include_term_groups and graph.edge_coloring is not None
    if include_term_groups and not grouped:
        Logger.debug("No edge coloring on lattice graph; falling back to ungrouped Hamiltonian construction.")
    return _build_sparse_hamiltonian(
        graph,
        couplings=couplings,
        fields=[("X", hx), ("Y", hy), ("Z", hz)],
        grouped=grouped,
        sparse_output=sparse_terms,
    )


def create_kitaev_hamiltonian(
    graph: LatticeGraph,
    kx: np.ndarray | float | Mapping[int, np.ndarray | float],
    ky: np.ndarray | float | Mapping[int, np.ndarray | float],
    kz: np.ndarray | float | Mapping[int, np.ndarray | float],
    j: np.ndarray | float | Mapping[int, np.ndarray | float] = 0.0,
    gamma: np.ndarray | float = 0.0,
    gamma_prime: np.ndarray | float = 0.0,
    *,
    gamma_x: np.ndarray | float | None = None,
    gamma_y: np.ndarray | float | None = None,
    gamma_z: np.ndarray | float | None = None,
    gamma_prime_x: np.ndarray | float | None = None,
    gamma_prime_y: np.ndarray | float | None = None,
    gamma_prime_z: np.ndarray | float | None = None,
    magnetic_field_abc: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    g_factors_abc: np.ndarray | tuple[float, float, float] = (1.0, 1.0, 1.0),
    bohr_magneton: float = 1.0,
    crystallographic_transform: np.ndarray | None = None,
    spin_basis_transform: np.ndarray | None = None,
    include_term_groups: bool = True,
    sparse_terms: bool = False,
) -> QubitOperator:
    r"""Create a flavored Kitaev-Heisenberg-Gamma model on a lattice.

    For a connection of flavor :math:`\gamma`, with :math:`(\alpha,\beta)` denoting the other two spin components,

    .. math::

        H_{ij,\gamma} = J\,\mathbf{S}_i\cdot\mathbf{S}_j
        + K_\gamma S_i^\gamma S_j^\gamma
        + \Gamma_\gamma(S_i^\alpha S_j^\beta + S_i^\beta S_j^\alpha)
        + \Gamma'_\gamma(S_i^\alpha S_j^\gamma + S_i^\gamma S_j^\alpha
        + S_i^\beta S_j^\gamma + S_i^\gamma S_j^\beta).

    A mapping ``{m: coupling}`` applies ``kx``, ``ky``, ``kz``, or ``j`` to already-selected shell-``m`` edges,
    multiplied by their weights; a scalar or array coupling ``J`` is the same as ``{1: J}``.
    Select active shells with :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` or custom edge labels first; the
    model never discovers or adds edges. Every active edge needs an X, Y, or Z flavor; pass
    :func:`kitaev_honeycomb_bond_flavors` to ``from_geometry`` for the standard honeycomb assignment.
    ``gamma_x``, ``gamma_y``, ``gamma_z`` and their primed counterparts apply to first-neighbor bonds; omitted
    flavor-specific values fall back to ``gamma`` or ``gamma_prime``.

    The magnetic field is specified in the crystallographic :math:`(a,b,c)` frame and contributes

    .. math::

        \mu_B \sum_i (g_a H_a S_i^a + g_b H_b S_i^b + g_c H_c S_i^c).

    ``bohr_magneton`` converts the field units to the energy units used by the exchange parameters and defaults to
    one for reduced-unit calculations. For fields in tesla and exchange parameters in meV, pass
    ``BOHR_MAGNETON / ELEMENTARY_CHARGE * 1e3`` from :mod:`qdk_chemistry.constants`.
    ``crystallographic_transform`` must be supplied for a nonzero field and
    defines :math:`\mathbf{S}_{abc}=D\mathbf{S}_{xyz}` for the lattice-specific crystallographic frame.
    The returned operator uses Pauli matrices, so two-body coefficients include
    :math:`S_i^\mu S_j^\nu=\sigma_i^\mu\sigma_j^\nu/4`, while field coefficients include
    :math:`S_i^\mu=\sigma_i^\mu/2`.

    Args:
        graph: Interaction graph whose active edges carry X/Y/Z flavors.
        kx: Kitaev coupling on X-flavor bonds as a scalar, ``(n, n)`` array, or shell mapping.
        ky: Kitaev coupling on Y-flavor bonds in the same format as ``kx``.
        kz: Kitaev coupling on Z-flavor bonds in the same format as ``kx``.
        j: Heisenberg coupling in the same format as ``kx``. Defaults to 0.
        gamma: Shared nearest-neighbor Gamma coupling used when a flavor-specific value is omitted. Defaults to 0.
        gamma_prime: Shared nearest-neighbor Gamma-prime coupling used when no flavor-specific value is given.
        gamma_x: Gamma coupling on X-flavor nearest-neighbor bonds. Defaults to ``gamma``.
        gamma_y: Gamma coupling on Y-flavor nearest-neighbor bonds. Defaults to ``gamma``.
        gamma_z: Gamma coupling on Z-flavor nearest-neighbor bonds. Defaults to ``gamma``.
        gamma_prime_x: Gamma-prime coupling on X-flavor nearest-neighbor bonds. Defaults to ``gamma_prime``.
        gamma_prime_y: Gamma-prime coupling on Y-flavor nearest-neighbor bonds. Defaults to ``gamma_prime``.
        gamma_prime_z: Gamma-prime coupling on Z-flavor nearest-neighbor bonds. Defaults to ``gamma_prime``.
        magnetic_field_abc: Magnetic-field vector ``(H_a, H_b, H_c)`` in the crystallographic frame. Defaults to zero.
        g_factors_abc: Diagonal ``(g_a, g_b, g_c)`` factors in the crystallographic frame. Defaults to one.
        bohr_magneton: Factor converting magnetic-field units to exchange-energy units. Defaults to 1.
        crystallographic_transform: Proper rotation ``D`` from cubic spin components to crystallographic components.
        spin_basis_transform: Proper rotation from Cartesian to output spin components. Defaults to identity.
        include_term_groups: Attach a geometry-coloring term partition. Defaults to ``True``.
        sparse_terms: Store terms by their non-identity factors instead of register-width labels. Defaults to ``False``.

    Returns:
        QubitOperator: The flavored spin model represented in the requested spin basis.

    Raises:
        ValueError: If shell metadata or flavors are missing, an active shell is unselected, or inputs are invalid.

    """
    if not graph.is_symmetric:
        raise ValueError("Lattice graph must be symmetric for a valid Hamiltonian.")

    def validate_transform(value: np.ndarray, name: str) -> np.ndarray:
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"{name} must have shape (3, 3).")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain only finite values.")
        if not np.allclose(matrix @ matrix.T, np.eye(3), rtol=1e-12, atol=1e-12):
            raise ValueError(f"{name} must be orthogonal.")
        if not np.isclose(np.linalg.det(matrix), 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{name} must be right-handed with determinant +1.")
        return matrix

    transform = (
        np.eye(3) if spin_basis_transform is None else validate_transform(spin_basis_transform, "spin_basis_transform")
    )

    prepared = {
        name: _shell_couplings(value, graph, name) for name, value in (("j", j), ("kx", kx), ("ky", ky), ("kz", kz))
    }
    requested_shells = {
        shell for values in prepared.values() for shell, value in values.items() if np.any(value != 0.0)
    }

    gamma_parameters: dict[KitaevBondFlavor, np.ndarray | float] = {}
    gamma_prime_parameters: dict[KitaevBondFlavor, np.ndarray | float] = {}
    for name, shared, overrides, values in (
        ("gamma", gamma, (gamma_x, gamma_y, gamma_z), gamma_parameters),
        (
            "gamma_prime",
            gamma_prime,
            (gamma_prime_x, gamma_prime_y, gamma_prime_z),
            gamma_prime_parameters,
        ),
    ):
        shared_value = _pair_parameter(shared, graph, name) if any(value is None for value in overrides) else 0.0
        for flavor, override in zip(KitaevBondFlavor, overrides, strict=True):
            values[flavor] = (
                shared_value if override is None else _pair_parameter(override, graph, f"{name}_{flavor.name.lower()}")
            )
    if any(np.any(value != 0.0) for value in (*gamma_parameters.values(), *gamma_prime_parameters.values())):
        requested_shells.add(1)

    magnetic_field = np.asarray(magnetic_field_abc, dtype=float)
    g_factors = np.asarray(g_factors_abc, dtype=float)
    if magnetic_field.shape != (3,) or g_factors.shape != (3,):
        raise ValueError("magnetic_field_abc and g_factors_abc must have shape (3,).")
    if not np.all(np.isfinite(magnetic_field)) or not np.all(np.isfinite(g_factors)):
        raise ValueError("magnetic_field_abc and g_factors_abc must contain only finite values.")
    if not np.isfinite(bohr_magneton):
        raise ValueError("bohr_magneton must be finite.")
    if np.any(magnetic_field != 0.0) and crystallographic_transform is None:
        raise ValueError("crystallographic_transform is required for a nonzero magnetic_field_abc.")
    if crystallographic_transform is None:
        output_field = np.zeros(3)
    else:
        crystal_transform = validate_transform(crystallographic_transform, "crystallographic_transform")
        weighted_field_abc = g_factors * magnetic_field
        # S_abc = D S_xyz and S_out = C S_xyz, hence
        # h_abc^T S_abc = (C D^T h_abc)^T S_out.
        weighted_field_xyz = crystal_transform.T @ weighted_field_abc
        output_field = bohr_magneton * transform @ weighted_field_xyz / 2.0

    edges = _selected_edges(graph, requested_shells)
    flavor_ids = set(KitaevBondFlavor)
    exchange_by_pair: dict[tuple[int, int], np.ndarray] = {}
    for pair, shell, flavor_id, weight in edges:
        if flavor_id is None or flavor_id not in flavor_ids:
            raise ValueError(
                "The Kitaev Hamiltonian requires X, Y, or Z flavor IDs for every requested geometric connection; "
                "build the graph with LatticeGraph.from_geometry and suitable bond flavors, or pass edge_labels."
            )
        flavor = KitaevBondFlavor(flavor_id)
        flavor_index = int(flavor)
        other_indices = tuple(index for index in range(3) if index != flavor_index)
        exchange = np.eye(3) * _edge_value(prepared["j"].get(shell, 0.0), pair, weight)
        exchange[flavor_index, flavor_index] += _edge_value(
            prepared["k" + flavor.name.lower()].get(shell, 0.0), pair, weight
        )
        gamma_value = _edge_value(gamma_parameters[flavor], pair, weight) if shell == 1 else 0.0
        gamma_prime_value = _edge_value(gamma_prime_parameters[flavor], pair, weight) if shell == 1 else 0.0
        exchange[other_indices[0], other_indices[1]] = gamma_value
        exchange[other_indices[1], other_indices[0]] = gamma_value
        for other_index in other_indices:
            exchange[flavor_index, other_index] = gamma_prime_value
            exchange[other_index, flavor_index] = gamma_prime_value
        transformed = transform @ exchange @ transform.T / 4.0
        scale = np.max(np.abs(transformed))
        if scale != 0.0:
            transformed[np.abs(transformed) < 100 * np.finfo(float).eps * scale] = 0.0
        exchange_by_pair[pair] = transformed

    pauli_components = ("X", "Y", "Z")
    couplings: list[tuple[str, dict[tuple[int, int], float]]] = []
    for first_index, first in enumerate(pauli_components):
        for second_index, second in enumerate(pauli_components):
            records = {
                pair: float(exchange[first_index, second_index])
                for pair, exchange in exchange_by_pair.items()
                if exchange[first_index, second_index] != 0.0
            }
            couplings.append((first + second, records))

    grouped = include_term_groups and graph.edge_coloring is not None
    if include_term_groups and not grouped:
        Logger.debug("No edge coloring on lattice graph; falling back to ungrouped Hamiltonian construction.")
    return _build_sparse_hamiltonian(
        graph,
        couplings=couplings,
        fields=list(zip(pauli_components, output_field, strict=True)),
        grouped=grouped,
        sparse_output=sparse_terms,
    )


def create_ising_hamiltonian(
    graph: LatticeGraph,
    j: np.ndarray | float | Mapping[int, np.ndarray | float],
    h: np.ndarray | float = 0.0,
    *,
    include_term_groups: bool = True,
    sparse_terms: bool = False,
) -> QubitOperator:
    r"""Create the Ising model Hamiltonian on a lattice.

    .. math::

        H = \sum_{\langle i,j \rangle} w_{ij}\,J^{ij}\,\sigma_i^z \sigma_j^z
          + \sum_i h^{i}\,\sigma_i^x

    A mapping ``{m: coupling}`` applies to the graph's shell-``m`` edges, multiplied
    by their weights; a scalar or array coupling ``J`` is the same as ``{1: J}``, and
    every edge of a graph without edge labels is a shell-1 edge. Select other shells
    with :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` or custom edge labels
    first; no edges are added by the model.

    Args:
        graph: Lattice graph defining the connectivity.
        j: ZZ coupling as a scalar, ``(n, n)`` array, or ``{m: coupling}`` geometric-shell mapping.
        h: Transverse field strength (x direction). Scalar or length-n array.  Defaults to 0.
        include_term_groups: When ``True`` (default), attach a geometry-coloring term partition to the result.
        sparse_terms: Store terms by their non-identity factors instead of register-width labels. Defaults to ``False``.

    Returns:
        QubitOperator: The Ising model as a qubit Hamiltonian.

    """
    return create_heisenberg_hamiltonian(
        graph, jx=0.0, jy=0.0, jz=j, hx=h, include_term_groups=include_term_groups, sparse_terms=sparse_terms
    )
