"""Shared geometric graph plotting for lattice-model examples."""

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
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from qdk_chemistry.data import LatticeGraph


def plot_lattice_graph(
    graph: LatticeGraph,
    *,
    shells: Sequence[int] | None = None,
    flavor_labels: Mapping[int, str] | None = None,
    flavor_colors: Mapping[int, str] | None = None,
    rotation_degrees: float = 0.0,
    vectors: Mapping[str, Sequence[float]] | None = None,
    vector_origin: Sequence[float] = (0.0, 0.0),
    title: str | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Draw one panel per shell without cropping, reclassifying bonds, or modifying the graph.

    All indexed sites are shown. Unflavored bonds use a neutral color; arbitrary
    flavor IDs get consistent colors across panels. Existing shell-1 bonds form
    a faint scaffold in higher-shell panels. Periodic connections end at their
    physical image coordinates, which may lie outside the fundamental cell.

    Args:
        graph: Graph carrying a two-dimensional geometry and resolved connections; supply a small graph for a preview.
        shells: Shells to display, in panel order; defaults to the graph's selected shells, or a sites-only panel.
        flavor_labels: Optional legend labels by flavor ID; IDs themselves are used otherwise.
        flavor_colors: Optional Matplotlib color strings by flavor ID; unspecified IDs use the default palette.
        rotation_degrees: Counterclockwise display rotation of positions, bonds, and annotation vectors only.
        vectors: Optional labels and Cartesian arrow vectors, such as scaled primitive lattice vectors.
        vector_origin: Common arrow origin in the original Cartesian frame.
        title: Optional figure title.

    Returns:
        Figure and a one-dimensional array of axes; the caller handles display or saving.

    Raises:
        ValueError: If the graph has no geometry or the display rotation is not finite.

    """
    geometry = graph.geometry
    if geometry is None:
        raise ValueError("Plotting requires a LatticeGraph with geometry.")
    if not np.isfinite(rotation_degrees):
        raise ValueError("rotation_degrees must be finite.")
    angle = np.deg2rad(rotation_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    original_positions = geometry.positions
    positions = original_positions @ rotation.T
    connections = graph.connections
    selected = list(dict.fromkeys(graph.selected_shells if shells is None else shells))
    panels = selected or [None]
    segments: dict[tuple[int, int | None], list[np.ndarray]] = {}
    for bond in connections:
        shell = bond.bond_class.shell
        if shell in selected or shell == 1:
            start = original_positions[bond.site_i]
            # Do not collapse distinct periodic images to the finite-cell endpoint.
            segment = np.array([start, start + bond.displacement]) @ rotation.T
            segments.setdefault((shell, bond.flavor), []).append(segment)

    flavors = sorted({flavor for shell, flavor in segments if shell in selected and flavor is not None})
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {flavor: palette[i % len(palette)] for i, flavor in enumerate(flavors)}
    colors.update(flavor_colors or {})
    scaffold = [segment for (shell, _), bonds in segments.items() if shell == 1 for segment in bonds]
    styles = ("solid", "dashed", "dotted", "dashdot")
    figure, axes_grid = plt.subplots(
        1, len(panels), figsize=(5.5 * len(panels), 5.6), squeeze=False, sharex=True, sharey=True, layout="constrained"
    )
    axes = axes_grid.ravel()
    for panel_index, (shell, axis) in enumerate(zip(panels, axes, strict=True)):
        ax: Axes = axis
        if shell is not None and shell != 1 and scaffold:
            ax.add_collection(LineCollection(scaffold, colors="#CBD5E1", linewidths=1.0, zorder=1))
        bond_count = 0
        for (bond_shell, flavor), bonds in segments.items():
            if bond_shell == shell:
                ax.add_collection(
                    LineCollection(
                        bonds,
                        colors=colors.get(flavor, "#1687B1"),
                        linestyles=styles[panel_index % len(styles)],
                        linewidths=1.9,
                        zorder=2,
                    )
                )
                bond_count += len(bonds)
        ax.scatter(*positions.T, s=14, color="#334155", zorder=3)
        origin = np.asarray(vector_origin) @ rotation.T
        for label, vector in (vectors or {}).items():
            endpoint = origin + np.asarray(vector) @ rotation.T
            ax.annotate(
                "",
                xy=endpoint,
                xytext=origin,
                arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.2},
                zorder=4,
            )
            ax.annotate(label, xy=endpoint, xytext=(3, 4), textcoords="offset points", fontsize=13)
            ax.update_datalim(np.array([origin, endpoint]))
        ax.autoscale_view()
        ax.margins(0.12)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"Shell {shell} | {bond_count} bonds" if shell is not None else "Sites (no shells selected)")
    legend = [
        Line2D([], [], color=colors[flavor], lw=2.5, label=(flavor_labels or {}).get(flavor, f"Flavor {flavor}"))
        for flavor in flavors
    ]
    if any(shell in selected and flavor is None for shell, flavor in segments):
        legend.append(Line2D([], [], color="#1687B1", lw=2.5, label="Unflavored bonds"))
    if scaffold and any(shell is not None and shell != 1 for shell in panels):
        legend.append(Line2D([], [], color="#CBD5E1", lw=1.5, label="Nearest-neighbor scaffold"))
    if legend:
        figure.legend(handles=legend, loc="outside lower center", ncols=len(legend), frameon=False)
    figure.suptitle(title or f"Lattice graph | {graph.num_sites} sites")
    return figure, axes
