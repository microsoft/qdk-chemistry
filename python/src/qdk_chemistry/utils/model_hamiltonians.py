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
    NeighborConnection,
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


def _selected_connections(
    graph: LatticeGraph,
    shells: set[int],
    *,
    required_shells: set[int] | None = None,
) -> list[NeighborConnection]:
    """Validate active shell requests without discovering or adding graph edges."""
    if not shells:
        return []
    selected_shells = set(graph.selected_shells)
    if not selected_shells and (graph.geometry is None or graph.num_nonzeros != 0):
        raise ValueError("Shell interactions require lattice geometry or explicit neighbor-shell metadata.")
    missing_shells = (shells if required_shells is None else required_shells) - selected_shells
    if missing_shells:
        raise ValueError(
            f"Requested neighbor shells {sorted(missing_shells)} are not selected in the lattice graph. "
            "Select them explicitly with LatticeGraph.from_geometry before constructing the Hamiltonian."
        )
    return [connection for connection in graph.connections if connection.bond_class.shell in shells]


def _build_sparse_hamiltonian(
    graph: LatticeGraph,
    *,
    couplings: list[tuple[str, dict[tuple[int, int], float]]],
    fields: list[tuple[str, np.ndarray | float]],
    grouped: bool,
    sparse_output: bool,
    pair_first: bool = True,
) -> QubitOperator:
    """Assemble sparse words, preserving the model's grouped or ungrouped ordering.

    Grouped families restrict the graph's stored coloring to nonzero coefficients,
    ordered by color and then pair; they never recolor their individual supports.
    Same-axis fields and interactions share a commuting group, with fields in
    their own disjoint layer. This merge changes metadata, not Pauli term order.
    Ungrouped mapped Heisenberg terms retain family, ascending-shell, and pair order;
    other ungrouped models use lexicographic pairs before coupling families.

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
        if pair_first:
            pairs = sorted({pair for _, coupling in couplings for pair, value in coupling.items() if value != 0.0})
            terms = ((label, pair, coupling.get(pair, 0.0)) for pair in pairs for label, coupling in couplings)
        else:
            terms = ((label, pair, value) for label, coupling in couplings for pair, value in coupling.items())
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
    # Content hashes distinguish dense and sparse storage. Preserve the legacy
    # output boundary without duplicating accumulation or allocating pair matrices.
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
    couplings instead defines :math:`K_a^{ij}` on already-selected graph edges
    independently of adjacency weights. Select nonzero requested shells with
    :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` before constructing
    the Hamiltonian; mappings never discover or add edges. Periodic shell mappings
    remain unsupported. Empty or all-zero mappings require no geometry.

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

    Returns:
        QubitOperator: The Heisenberg model as a qubit Hamiltonian; carries a ``LayeredPartition`` when grouped.

    Raises:
        ValueError: If the graph is asymmetric, shell metadata is absent, or an active mapped shell is unselected.
        RuntimeError: If active shell mappings are used with periodic geometry.

    """
    if not graph.is_symmetric:
        raise ValueError("Lattice graph must be symmetric for a valid Hamiltonian.")

    shell_couplings = any(isinstance(coupling, Mapping) for coupling in (jx, jy, jz))
    coupling_specs = [("XX", "jx", jx), ("YY", "jy", jy), ("ZZ", "jz", jz)]
    normalized_shell_couplings: dict[str, dict[int, np.ndarray | float]] = {}
    adjacency_couplings: dict[str, np.ndarray | float] = {}
    requested_shells: set[int] = set()
    for _, name, coupling in coupling_specs:
        if isinstance(coupling, Mapping):
            normalized: dict[int, np.ndarray | float] = {}
            for shell, shell_coupling in coupling.items():
                if isinstance(shell, bool) or not isinstance(shell, Integral) or shell < 1:
                    raise ValueError(f"{name} shell indices must be positive integers; got {shell!r}.")
                shell_index = int(shell)
                normalized[shell_index] = _pair_parameter(shell_coupling, graph, f"{name}[{shell_index}]")
                if np.any(normalized[shell_index] != 0.0):
                    requested_shells.add(shell_index)
            normalized_shell_couplings[name] = normalized
        else:
            adjacency_couplings[name] = _pair_parameter(coupling, graph, name)

    connections = _selected_connections(graph, requested_shells)
    geometry = graph.geometry
    if requested_shells and (
        (geometry is not None and geometry.periods is not None)
        or any(any(connection.image_shift) for connection in connections)
    ):
        raise RuntimeError("Heisenberg shell mappings support open lattices only.")
    shell_pairs: dict[int, set[tuple[int, int]]] = {}
    for connection in connections:
        if connection.site_i == connection.site_j:
            raise ValueError("Heisenberg interactions cannot connect a site to its own periodic image.")
        shell_pairs.setdefault(connection.bond_class.shell, set()).add((connection.site_i, connection.site_j))

    adjacency_edges = []
    if adjacency_couplings:
        adjacency = scipy.sparse.triu(graph.sparse_adjacency_matrix(), k=1).tocoo()
        adjacency_edges = sorted(zip(adjacency.row, adjacency.col, adjacency.data, strict=True))
    couplings: list[tuple[str, dict[tuple[int, int], float]]] = []
    for label, name, _ in coupling_specs:
        records: dict[tuple[int, int], float] = {}
        if name in normalized_shell_couplings:
            for shell_index, values in sorted(normalized_shell_couplings[name].items()):
                for site_i, site_j in sorted(shell_pairs.get(shell_index, ())):
                    value = values if isinstance(values, float) else values[site_i, site_j]
                    if value != 0.0:
                        records[site_i, site_j] = float(value)
        else:
            values = adjacency_couplings[name]
            for site_i, site_j, weight in adjacency_edges:
                if weight == 0.0:
                    continue
                value = values if isinstance(values, float) else values[site_i, site_j]
                coefficient = float(value * weight)
                if coefficient != 0.0:
                    records[int(site_i), int(site_j)] = coefficient
        couplings.append((label, records))

    grouped = include_term_groups and graph.edge_coloring is not None
    if include_term_groups and not grouped:
        Logger.debug("No edge coloring on lattice graph; falling back to ungrouped Hamiltonian construction.")
    return _build_sparse_hamiltonian(
        graph,
        couplings=couplings,
        fields=[("X", hx), ("Y", hy), ("Z", hz)],
        grouped=grouped,
        sparse_output=grouped and shell_couplings,
        pair_first=not shell_couplings,
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
) -> QubitOperator:
    r"""Create a flavored Kitaev-Heisenberg-Gamma model on a lattice.

    For a connection of flavor :math:`\gamma`, with :math:`(\alpha,\beta)` denoting the other two spin components,

    .. math::

        H_{ij,\gamma} = J\,\mathbf{S}_i\cdot\mathbf{S}_j
        + K_\gamma S_i^\gamma S_j^\gamma
        + \Gamma_\gamma(S_i^\alpha S_j^\beta + S_i^\beta S_j^\alpha)
        + \Gamma'_\gamma(S_i^\alpha S_j^\gamma + S_i^\gamma S_j^\alpha
        + S_i^\beta S_j^\gamma + S_i^\gamma S_j^\beta).

    Scalars and arrays use selected first-neighbor connections and their weights. A mapping ``{m: coupling}``
    applies ``kx``, ``ky``, ``kz``, or ``j`` to already-selected shell ``m`` edges independently of their weights.
    Select active mapped shells with :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` first; the model never
    discovers or adds edges. Entirely unflavored active connections use :func:`kitaev_honeycomb_bond_flavors`;
    partial or invalid explicit flavor labels are rejected rather than replaced.
    ``gamma_x``, ``gamma_y``, ``gamma_z`` and their primed counterparts apply to first-neighbor bonds; omitted
    flavor-specific values fall back to ``gamma`` or ``gamma_prime``. Distinct periodic-image connections are
    accumulated when they collapse onto the same finite-lattice site pair.

    The magnetic field is specified in the crystallographic :math:`(a,b,c)` frame and contributes

    .. math::

        \mu_B \sum_i (g_a H_a S_i^a + g_b H_b S_i^b + g_c H_c S_i^c).

    ``bohr_magneton`` converts the field units to the energy units used by the exchange parameters and defaults to
    one for reduced-unit calculations. ``crystallographic_transform`` must be supplied for a nonzero field and
    defines :math:`\mathbf{S}_{abc}=D\mathbf{S}_{xyz}` for the lattice-specific crystallographic frame.
    The returned operator uses Pauli matrices, so two-body coefficients include
    :math:`S_i^\mu S_j^\nu=\sigma_i^\mu\sigma_j^\nu/4`, while field coefficients include
    :math:`S_i^\mu=\sigma_i^\mu/2`.

    Args:
        graph: Selected interaction graph with X/Y/Z flavors, or unflavored connections matching the honeycomb defaults.
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

    parameters = {"j": j, "kx": kx, "ky": ky, "kz": kz}
    mapped_parameters = {name for name, value in parameters.items() if isinstance(value, Mapping)}
    prepared: dict[str, dict[int, np.ndarray | float]] = {}
    requested_shells: set[int] = set()
    mapped_shells: set[int] = set()
    for name, parameter in parameters.items():
        shell_values = parameter.items() if isinstance(parameter, Mapping) else [(1, parameter)]
        normalized: dict[int, np.ndarray | float] = {}
        for shell, value in shell_values:
            if isinstance(shell, bool) or not isinstance(shell, Integral) or shell < 1:
                raise ValueError(f"{name} shell indices must be positive integers; got {shell!r}.")
            shell_index = int(shell)
            normalized[shell_index] = _pair_parameter(value, graph, f"{name}[{shell_index}]")
            if np.any(normalized[shell_index] != 0.0):
                requested_shells.add(shell_index)
                if name in mapped_parameters:
                    mapped_shells.add(shell_index)
        prepared[name] = normalized

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

    connections = _selected_connections(graph, requested_shells, required_shells=mapped_shells)
    if connections and all(connection.flavor is None for connection in connections):
        model_graph = graph.with_bond_flavors(kitaev_honeycomb_bond_flavors())
        connections = [
            connection for connection in model_graph.connections if connection.bond_class.shell in requested_shells
        ]
    try:
        connection_flavors = [KitaevBondFlavor(connection.flavor) for connection in connections]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The Kitaev Hamiltonian requires X, Y, or Z flavor IDs for every requested geometric connection."
        ) from error
    if any(connection.site_i == connection.site_j for connection in connections):
        raise ValueError("Kitaev interactions cannot connect a site to its own periodic image.")

    def parameter_value(name: str, connection: NeighborConnection) -> float:
        shell = connection.bond_class.shell
        if shell not in prepared[name]:
            return 0.0
        value = prepared[name][shell]
        result = value if isinstance(value, float) else value[connection.site_i, connection.site_j]
        if name not in mapped_parameters:
            result *= connection.weight
        return float(result)

    def nearest_neighbor_value(
        values: dict[KitaevBondFlavor, np.ndarray | float],
        connection: NeighborConnection,
        flavor: KitaevBondFlavor,
    ) -> float:
        if connection.bond_class.shell != 1:
            return 0.0
        value = values[flavor]
        result = value if isinstance(value, float) else value[connection.site_i, connection.site_j]
        return float(result * connection.weight)

    exchange_by_pair: dict[tuple[int, int], np.ndarray] = {}
    for connection, flavor in zip(connections, connection_flavors, strict=True):
        flavor_index = int(flavor)
        other_indices = tuple(index for index in range(3) if index != flavor_index)
        exchange = np.eye(3) * parameter_value("j", connection)
        exchange[flavor_index, flavor_index] += parameter_value("k" + flavor.name.lower(), connection)
        gamma_value = nearest_neighbor_value(gamma_parameters, connection, flavor)
        gamma_prime_value = nearest_neighbor_value(gamma_prime_parameters, connection, flavor)
        exchange[other_indices[0], other_indices[1]] = gamma_value
        exchange[other_indices[1], other_indices[0]] = gamma_value
        for other_index in other_indices:
            exchange[flavor_index, other_index] = gamma_prime_value
            exchange[other_index, flavor_index] = gamma_prime_value
        transformed = transform @ exchange @ transform.T / 4.0
        scale = np.max(np.abs(transformed))
        if scale != 0.0:
            transformed[np.abs(transformed) < 100 * np.finfo(float).eps * scale] = 0.0
        pair = (connection.site_i, connection.site_j)
        exchange_by_pair.setdefault(pair, np.zeros((3, 3)))
        exchange_by_pair[pair] += transformed

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
        sparse_output=grouped and bool(mapped_parameters),
    )


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
    ``{m: coupling}`` instead filters already-selected graph edges by geometric
    shell, independently of adjacency weights. Select active mapped shells with
    :meth:`~qdk_chemistry.data.LatticeGraph.from_geometry` first; no edges are added
    by the model. Periodic shell mappings remain unsupported.

    Args:
        graph: Lattice graph defining the connectivity.
        j: ZZ coupling as a scalar, ``(n, n)`` array, or ``{m: coupling}`` geometric-shell mapping.
        h: Transverse field strength (x direction). Scalar or length-n array.  Defaults to 0.
        include_term_groups: When ``True`` (default), attach a geometry-coloring term partition to the result.

    Returns:
        QubitOperator: The Ising model as a qubit Hamiltonian.

    """
    return create_heisenberg_hamiltonian(graph, jx=0.0, jy=0.0, jz=j, hx=h, include_term_groups=include_term_groups)
