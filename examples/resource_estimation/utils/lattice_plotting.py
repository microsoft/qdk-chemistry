"""Utility functions for lattice graph visualization."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray
    from qdk_chemistry.data import LatticeGraph


def plot_lattice_graph(
    graph: LatticeGraph,
    positions: ArrayLike,
    *,
    flavor_labels: Mapping[int, str] | None = None,
    title: str | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Draw one panel per shell without cropping, reclassifying bonds, or modifying the graph.

    All indexed sites are shown. Each flavor ID, and unflavored bonds, get their own
    color from the default color cycle, consistent across panels. Existing shell-1
    bonds form a faint scaffold in higher-shell panels. As in the model Hamiltonian
    builders, every edge of a graph without edge labels is an unflavored shell-1 bond.
    Each bond is drawn straight between its site positions, so periodic wrap-around
    bonds cross the drawing.

    Args:
        graph: Graph to draw, such as one from LatticeGraph.from_geometry; supply a small graph for a preview.
        positions: Two-dimensional site positions with one row per graph site, such as LatticeGeometry.positions.
        flavor_labels: Optional legend labels by flavor ID; IDs themselves are used otherwise.
        title: Optional figure title.

    Returns:
        Figure and a one-dimensional array of axes; the caller handles display or saving.

    Raises:
        ValueError: If positions are not one two-dimensional row per site.

    """
    site_positions = np.asarray(positions, dtype=float)
    if site_positions.shape != (graph.num_sites, 2):
        raise ValueError("positions must have one two-dimensional row per graph site.")
    edges: dict[tuple[int, int], tuple[int, int | None]] = {
        pair: (label.shell, label.flavor) for pair, label in graph.edge_labels.items()
    }
    if not edges:
        adjacency = graph.sparse_adjacency_matrix().tocoo()
        edges = {
            (min(i, j), max(i, j)): (1, None)
            for i, j, weight in zip(
                adjacency.row.tolist(),
                adjacency.col.tolist(),
                adjacency.data.tolist(),
                strict=True,
            )
            if i != j and weight != 0.0
        }
    shells = sorted({shell for shell, _ in edges.values()})
    panels: list[int | None] = [*shells] or [None]
    segments: dict[tuple[int, int | None], list[np.ndarray]] = {}
    for (site_i, site_j), key in edges.items():
        segments.setdefault(key, []).append(site_positions[[site_i, site_j]])

    flavors = sorted({flavor for _, flavor in segments if flavor is not None})
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors: dict[int | None, str] = {
        flavor: palette[i % len(palette)] for i, flavor in enumerate(flavors)
    }
    if any(flavor is None for _, flavor in segments):
        colors[None] = palette[len(colors) % len(palette)]
    scaffold = [
        segment
        for (shell, _), bonds in segments.items()
        if shell == 1
        for segment in bonds
    ]
    styles = ("solid", "dashed", "dotted", "dashdot")
    figure, axes_grid = plt.subplots(
        1,
        len(panels),
        figsize=(5.5 * len(panels), 5.6),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes = axes_grid.ravel()
    for panel_index, (shell, axis) in enumerate(zip(panels, axes, strict=True)):
        ax: Axes = axis
        if shell != 1 and scaffold:
            ax.add_collection(
                LineCollection(scaffold, colors="#CBD5E1", linewidths=1.0, zorder=1)
            )
        bond_count = 0
        for (bond_shell, flavor), bonds in segments.items():
            if bond_shell == shell:
                ax.add_collection(
                    LineCollection(
                        bonds,
                        colors=colors[flavor],
                        linestyles=styles[panel_index % len(styles)],
                        linewidths=1.9,
                        zorder=2,
                    )
                )
                bond_count += len(bonds)
        ax.scatter(*site_positions.T, s=14, color="#334155", zorder=3)
        ax.autoscale_view()
        ax.margins(0.12)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(
            f"Shell {shell} | {bond_count} bonds" if shell is not None else "Sites only"
        )
    legend = [
        Line2D(
            [],
            [],
            color=color,
            lw=2.5,
            label="Unflavored bonds"
            if flavor is None
            else (flavor_labels or {}).get(flavor, f"Flavor {flavor}"),
        )
        for flavor, color in colors.items()
    ]
    if scaffold and any(shell != 1 for shell in panels):
        legend.append(
            Line2D([], [], color="#CBD5E1", lw=1.5, label="Nearest-neighbor scaffold")
        )
    if legend:
        figure.legend(
            handles=legend, loc="outside lower center", ncols=len(legend), frameon=False
        )
    figure.suptitle(title or f"Lattice graph | {graph.num_sites} sites")
    return figure, axes
