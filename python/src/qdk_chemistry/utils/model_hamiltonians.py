"""Model Hamiltonian utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Iterable, Mapping
from enum import IntEnum
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
from qdk_chemistry.data import BondFlavorDefinition, LatticeGraph, LayeredPartition, QubitOperator
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


def _build_geometry_grouped_hamiltonian(
    graph: LatticeGraph,
    *,
    couplings: list[tuple[str, np.ndarray | float]],
    fields: list[tuple[str, np.ndarray | float]],
    coloring: dict[tuple[int, int], int] | None = None,
    apply_edge_weights: bool = True,
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
        apply_edge_weights: Multiply coupling matrices by adjacency weights when ``True``. Defaults to ``True``.

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
            edge_weight = graph.weight(i, j) if apply_edge_weights else 1.0
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

    Scalars and arrays retain the nearest-neighbor behavior and adjacency weighting. A mapping ``{m: coupling}``
    applies ``kx``, ``ky``, ``kz``, or ``j`` to geometric shell ``m`` independently of adjacency weights.
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
        graph: Lattice graph with complete X, Y, and Z flavor metadata for every requested geometric connection.
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
        include_term_groups: Attach an adjacency coloring partition for nearest-neighbor inputs. Defaults to ``True``.

    Returns:
        QubitOperator: The flavored spin model represented in the requested spin basis.

    Raises:
        ValueError: If the graph lacks required flavors, a shell index is invalid, or the basis transform is invalid.

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

    def pair_parameter(value: np.ndarray | float, name: str) -> np.ndarray | float:
        if isinstance(value, int | float | np.integer | np.floating):
            return float(value)
        return to_pair_param(value, graph, name)

    parameters = {"j": j, "kx": kx, "ky": ky, "kz": kz}
    mapped_parameters = {name for name, value in parameters.items() if isinstance(value, Mapping)}
    prepared: dict[str, dict[int, np.ndarray | float]] = {}
    requested_shells: set[int] = set()
    for name, parameter in parameters.items():
        shell_values = parameter.items() if isinstance(parameter, Mapping) else [(1, parameter)]
        normalized: dict[int, np.ndarray | float] = {}
        for shell, value in shell_values:
            if isinstance(shell, bool) or not isinstance(shell, Integral) or shell < 1:
                raise ValueError(f"{name} shell indices must be positive integers; got {shell!r}.")
            shell_index = int(shell)
            normalized[shell_index] = pair_parameter(value, f"{name}[{shell_index}]")
            if np.any(normalized[shell_index] != 0.0):
                requested_shells.add(shell_index)
        prepared[name] = normalized

    gamma_parameters = {
        KitaevBondFlavor.X: pair_parameter(gamma if gamma_x is None else gamma_x, "gamma_x"),
        KitaevBondFlavor.Y: pair_parameter(gamma if gamma_y is None else gamma_y, "gamma_y"),
        KitaevBondFlavor.Z: pair_parameter(gamma if gamma_z is None else gamma_z, "gamma_z"),
    }
    gamma_prime_parameters = {
        KitaevBondFlavor.X: pair_parameter(gamma_prime if gamma_prime_x is None else gamma_prime_x, "gamma_prime_x"),
        KitaevBondFlavor.Y: pair_parameter(gamma_prime if gamma_prime_y is None else gamma_prime_y, "gamma_prime_y"),
        KitaevBondFlavor.Z: pair_parameter(gamma_prime if gamma_prime_z is None else gamma_prime_z, "gamma_prime_z"),
    }
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

    model_graph = (
        graph.with_bond_flavors(kitaev_honeycomb_bond_flavors())
        if requested_shells and graph.positions is not None and not graph.bond_flavor_definitions
        else graph
    )
    connections = model_graph.neighbor_connections(sorted(requested_shells))
    try:
        connection_flavors = [KitaevBondFlavor(connection.flavor) for connection in connections]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The Kitaev Hamiltonian requires X, Y, or Z flavor IDs for every requested geometric connection."
        ) from error
    if any(connection.site_i == connection.site_j for connection in connections):
        raise ValueError("Kitaev interactions cannot connect a site to its own periodic image.")

    shell_one_multiplicity: dict[tuple[int, int], int] = {}
    for connection in connections:
        if connection.bond_class.shell == 1:
            pair = (connection.site_i, connection.site_j)
            shell_one_multiplicity[pair] = shell_one_multiplicity.get(pair, 0) + 1

    def parameter_value(name: str, connection) -> float:
        shell = connection.bond_class.shell
        if shell not in prepared[name]:
            return 0.0
        value = prepared[name][shell]
        result = value if isinstance(value, float) else value[connection.site_i, connection.site_j]
        if name not in mapped_parameters:
            pair = (connection.site_i, connection.site_j)
            result *= graph.weight(*pair) / shell_one_multiplicity[pair]
        return float(result)

    def nearest_neighbor_value(
        values: dict[KitaevBondFlavor, np.ndarray | float], connection, flavor: KitaevBondFlavor
    ) -> float:
        if connection.bond_class.shell != 1:
            return 0.0
        value = values[flavor]
        result = value if isinstance(value, float) else value[connection.site_i, connection.site_j]
        pair = (connection.site_i, connection.site_j)
        return float(result * graph.weight(*pair) / shell_one_multiplicity[pair])

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
    coupling_matrices = {
        first + second: np.zeros((graph.num_sites, graph.num_sites))
        for first in pauli_components
        for second in pauli_components
    }
    for (site_i, site_j), exchange in exchange_by_pair.items():
        for first_index, first in enumerate(pauli_components):
            for second_index, second in enumerate(pauli_components):
                coupling_matrices[first + second][site_i, site_j] = exchange[first_index, second_index]

    couplings = list(coupling_matrices.items())
    if include_term_groups and not mapped_parameters and graph.edge_coloring is not None:
        return _build_geometry_grouped_hamiltonian(
            graph,
            couplings=couplings,
            fields=list(zip(pauli_components, output_field, strict=True)),
            apply_edge_weights=False,
        )
    if include_term_groups and mapped_parameters:
        Logger.debug("Geometric shell couplings use ungrouped Hamiltonian construction.")

    pauli_strings: list[str] = []
    coefficients: list[complex] = []
    for (site_i, site_j), exchange in sorted(exchange_by_pair.items()):
        for first_index, first in enumerate(pauli_components):
            for second_index, second in enumerate(pauli_components):
                coefficient = exchange[first_index, second_index]
                if coefficient == 0.0:
                    continue
                pauli = ["I"] * graph.num_sites
                pauli[site_i] = first
                pauli[site_j] = second
                pauli_strings.append("".join(pauli[::-1]))
                coefficients.append(complex(coefficient))
    for site in range(graph.num_sites):
        for component, coefficient in zip(pauli_components, output_field, strict=True):
            if coefficient == 0.0:
                continue
            pauli = ["I"] * graph.num_sites
            pauli[site] = component
            pauli_strings.append("".join(pauli[::-1]))
            coefficients.append(complex(coefficient))
    if not pauli_strings:
        pauli_strings = ["I" * graph.num_sites]
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
