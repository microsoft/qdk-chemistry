"""Orbital entanglement chord diagram.

Visualises single-orbital entropies and mutual information from a
``Wavefunction`` object as a chord diagram.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MplPath

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import matplotlib.axes
    import matplotlib.figure

    from qdk_chemistry.data import Wavefunction

__all__ = ["plot_orbital_entanglement"]


def _deg2xy(deg: float, r: float) -> tuple[float, float]:
    """Convert polar (degrees, radius) to Cartesian (x, y)."""
    rad = np.radians(deg)
    return r * np.cos(rad), r * np.sin(rad)


def _draw_arc(
    ax: matplotlib.axes.Axes,
    start_deg: float,
    end_deg: float,
    inner_r: float,
    outer_r: float,
    color: tuple,
    **kw: object,
) -> None:
    """Draw a filled annular arc on *ax*."""
    theta = np.linspace(np.radians(start_deg), np.radians(end_deg), 80)
    xs_out = outer_r * np.cos(theta)
    ys_out = outer_r * np.sin(theta)
    xs_in = inner_r * np.cos(theta[::-1])
    ys_in = inner_r * np.sin(theta[::-1])
    ax.fill(
        np.concatenate([xs_out, xs_in]),
        np.concatenate([ys_out, ys_in]),
        color=color,
        **kw,
    )


def _draw_arc_outline(
    ax: matplotlib.axes.Axes,
    start_deg: float,
    end_deg: float,
    inner_r: float,
    outer_r: float,
    edgecolor: str = "#222222",
    linewidth: float = 2.5,
    **kw: object,
) -> None:
    """Draw an outline (no fill) around an annular arc on *ax*."""
    theta = np.linspace(np.radians(start_deg), np.radians(end_deg), 80)
    xs_out = outer_r * np.cos(theta)
    ys_out = outer_r * np.sin(theta)
    xs_in = inner_r * np.cos(theta[::-1])
    ys_in = inner_r * np.sin(theta[::-1])
    ax.fill(
        np.concatenate([xs_out, xs_in]),
        np.concatenate([ys_out, ys_in]),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kw,
    )


def _draw_chord_line(
    ax: matplotlib.axes.Axes,
    angle_a: float,
    angle_b: float,
    color: tuple,
    linewidth: float,
    radius: float,
    arc_width: float,
) -> None:
    """Draw a cubic Bezier chord line between two points on the inner rim."""
    inner = radius - arc_width
    ctrl_r = inner * 0.55

    p0 = _deg2xy(angle_a, inner)
    c0 = _deg2xy(angle_a, ctrl_r)
    c1 = _deg2xy(angle_b, ctrl_r)
    p1 = _deg2xy(angle_b, inner)

    verts = [p0, c0, c1, p1]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]

    patch = mpatches.PathPatch(
        MplPath(verts, codes),
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        capstyle="round",
    )
    ax.add_patch(patch)


def plot_orbital_entanglement(
    wavefunction: Wavefunction,
    *,
    labels: Sequence[str] | None = None,
    gap_deg: float = 3.0,
    radius: float = 1.0,
    arc_width: float = 0.08,
    line_scale: float | None = None,
    title: str | None = "Orbital Entanglement",
    figsize: tuple[float, float] = (10, 11),
    save_path: str | Path | None = None,
    dpi: int = 150,
    mi_threshold: float = 0.0,
    s1_vmax: float | None = None,
    mi_vmax: float | None = None,
    selected_indices: Sequence[int] | None = None,
    selection_color: str = "#222222",
    selection_linewidth: float = 2.5,
    transparent_background: bool = False,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a chord diagram of orbital entropies and mutual information.

    The diagram encodes:
      * **Arc length** - proportional to the single-orbital entropy
        (sum of mutual information per orbital).
      * **Chord thickness** - proportional to the pairwise mutual
        information between two orbitals.

    Parameters
    ----------
    wavefunction:
        A ``Wavefunction`` that provides single-orbital entropies and
        mutual information (via ``get_single_orbital_entropies()`` and
        ``get_mutual_information()``).
    labels:
        Optional orbital labels.  When *None*, labels default to the
        true orbital indices (active-space indices when available,
        otherwise ``0, 1, 2, …``).
    gap_deg:
        Gap between arcs on the circle (degrees).
    radius:
        Outer radius of the diagram.
    arc_width:
        Width of each outer arc band.
    line_scale:
        Multiplier applied to mutual-information values to set chord
        line widths.  When *None* (default), a value is chosen
        automatically based on the number of orbitals so that chords
        remain readable for both small and large systems.
    title:
        Plot title.  Pass *None* to suppress.
    figsize:
        Figure size in inches ``(width, height)``.
    save_path:
        If given, the figure is saved to this path.
    dpi:
        Resolution used when saving.
    mi_threshold:
        Mutual-information values below this threshold are not drawn.
    s1_vmax:
        Fixed upper bound for the single-orbital entropy colour scale.
        When *None* the data maximum is used.  Setting a constant value
        across plots makes colours directly comparable.
    mi_vmax:
        Fixed upper bound for the mutual-information colour scale.
        When *None* the data maximum is used.
    selected_indices:
        Optional sequence of **orbital indices** (matching the label
        values, i.e. the active-space indices) whose arcs should be
        highlighted with an outline.  Useful for showing e.g. the
        AutoCAS-selected subset.
    selection_color:
        Edge colour of the selection outline.
    selection_linewidth:
        Stroke width of the selection outline.
    transparent_background:
        When *True*, the figure and axes backgrounds are set to
        transparent.  Useful for embedding the diagram in documents
        or web pages with a non-white background.
    ax:
        An existing ``Axes`` to draw into.  When *None* a new figure
        and axes are created.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib ``Figure`` and ``Axes`` objects so the caller
        can further customise the plot.

    Raises
    ------
    RuntimeError
        If the wavefunction does not contain the required entropy data.

    Examples
    --------
    >>> from qdk_chemistry.utils.visualization import plot_orbital_entanglement
    >>> fig, ax = plot_orbital_entanglement(wavefunction)
    >>> fig.savefig("entanglement.png")

    """
    # 1. Extract data from wavefunction
    if not wavefunction.has_single_orbital_entropies():
        raise RuntimeError(
            "The wavefunction does not contain single-orbital entropies. "
            "Make sure the wavefunction was computed with entropy support."
        )
    if not wavefunction.has_mutual_information():
        raise RuntimeError(
            "The wavefunction does not contain mutual information. "
            "Make sure the wavefunction was computed with entropy support."
        )

    s1 = np.asarray(wavefunction.get_single_orbital_entropies())
    mi_matrix = np.asarray(wavefunction.get_mutual_information())
    n = len(s1)

    # Auto-scale line thickness.  The maximum rendered linewidth is
    # capped so that even the strongest MI value produces a line no
    # wider than *max_lw* points.  A sqrt transform compresses the
    # dynamic range so weak connections remain visible next to strong
    # ones.  The cap shrinks with orbital count to keep large diagrams
    # clean.
    max_lw = max(12.0 * (20.0 / max(n, 1)) ** 0.5, 2.0)  # pt
    if line_scale is None:
        mi_peak = mi_matrix.max() if mi_matrix.max() > 0 else 1.0
        # Scale so that sqrt(mi_peak) * line_scale == max_lw
        line_scale = max_lw / np.sqrt(mi_peak)

    # 2. Build labels
    if labels is None:
        try:
            orbitals = wavefunction.get_orbitals()
            if orbitals.has_active_space():
                active_indices = orbitals.get_active_space_indices()[0]
                labels = [str(idx) for idx in active_indices]
            else:
                labels = [str(i) for i in range(n)]
        except (AttributeError, TypeError, IndexError):
            labels = [str(i) for i in range(n)]

    if len(labels) != n:
        raise ValueError(f"Number of labels ({len(labels)}) does not match the number of orbitals ({n}).")

    # 3. Colour scales
    #  Both scales run from black (0) through a saturated colour at the
    #  low-to-mid range out to light grey at the theoretical maximum.
    #  This gives more colour resolution where values typically cluster.
    arc_cmap = LinearSegmentedColormap.from_list(
        "grey_red_black",
        ["#d8d8d8", "#c82020", "#1a1a1a"],
    )
    chord_cmap = LinearSegmentedColormap.from_list(
        "grey_blue_black",
        ["#d8d8d8", "#2060b0", "#1a1a1a"],
    )

    s1_max = s1_vmax if s1_vmax is not None else np.log(4.0)
    arc_colours = [arc_cmap(v / s1_max) for v in s1]

    mi_max = mi_vmax if mi_vmax is not None else np.log(16.0)

    # 4. Arc geometry
    #  Arc length ∝ single-orbital entropy (acts as node importance).
    totals = s1.copy()
    # Guard against all-zero entropies (give equal arcs).
    grand = totals.sum()
    if grand == 0.0:
        totals = np.ones(n)
        grand = float(n)

    gap_total = gap_deg * n
    arc_degs = (360.0 - gap_total) * totals / grand

    starts = np.zeros(n)
    for i in range(1, n):
        starts[i] = starts[i - 1] + arc_degs[i - 1] + gap_deg

    # 5. Create figure / axes
    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=figsize)
        # Centre column (2/3 width) holds the colorbars; side columns
        # are padding so bars align under the circle, not the full figure.
        gs = GridSpec(
            3,
            3,
            height_ratios=[1, 0.025, 0.025],
            width_ratios=[1, 4, 1],
            hspace=0.08,
            figure=fig,
        )
        ax = fig.add_subplot(gs[0, :])
        cax_arc = fig.add_subplot(gs[1, 1])
        cax_chord = fig.add_subplot(gs[2, 1])
    else:
        assert ax is not None  # guaranteed by the caller
        fig = ax.figure
        cax_arc = None
        cax_chord = None

    assert ax is not None
    ax.set_aspect("equal")
    label_margin = radius + 0.25
    ax.set_xlim(-label_margin, label_margin)
    ax.set_ylim(-label_margin, label_margin)
    ax.axis("off")
    if created_fig:
        if transparent_background:
            fig.patch.set_alpha(0.0)
        else:
            fig.patch.set_facecolor("white")
    if transparent_background:
        ax.set_facecolor("none")

    # 6. Draw outer arcs and labels
    #  Strategy: keep labels at a legible font size (≥7pt), and when
    #  adjacent labels would overlap, push alternating labels to outer
    #  radial tiers with a thin connecting line back to the arc.

    # Font size: constant 9pt for <=20 orbitals, then 7pt.
    label_fontsize = 9 if n <= 20 else 7

    # Estimate the angular width a label occupies (in degrees).
    # A rough heuristic: each character at the current font size
    # subtends ~(fontsize * 0.012 / radius) radians ~ degrees * 0.7.
    max_label_len = max(len(lbl) for lbl in labels)
    char_deg = label_fontsize * 0.7 * max_label_len / max(radius, 0.5)
    # Minimum angular separation needed to avoid overlap.
    min_sep_deg = char_deg * 0.8

    # Compute arc mid-angles and decide radial tier per label.
    arc_mids = [starts[i] + arc_degs[i] / 2.0 for i in range(n)]
    base_offset = 0.07  # tier 0 distance from rim
    tier_step = 0.09  # additional offset per tier
    max_tiers = 4  # how many stagger levels

    # Greedy assignment: walk labels in angular order and bump the tier
    # whenever the previous placed label is too close.
    index_order = sorted(range(n), key=lambda i: arc_mids[i])
    tier = [0] * n  # tier per orbital index
    prev_angle = -999.0
    prev_tier = -1
    for idx in index_order:
        ang = arc_mids[idx]
        if ang - prev_angle < min_sep_deg:
            # Too close - bump to next tier (cycling)
            tier[idx] = (prev_tier + 1) % max_tiers
        else:
            tier[idx] = 0
        prev_angle = ang
        prev_tier = tier[idx]

    # Also check wrap-around (last label vs first).
    first_idx = index_order[0]
    last_idx = index_order[-1]
    wrap_gap = (arc_mids[first_idx] + 360.0) - arc_mids[last_idx]
    if wrap_gap < min_sep_deg and tier[first_idx] == tier[last_idx]:
        tier[first_idx] = (tier[last_idx] + 1) % max_tiers

    for i in range(n):
        _draw_arc(
            ax,
            starts[i],
            starts[i] + arc_degs[i],
            radius - arc_width,
            radius,
            arc_colours[i],
            zorder=2,
        )

        mid = arc_mids[i]
        t = tier[i]
        offset = base_offset + t * tier_step
        lx, ly = _deg2xy(mid, radius + offset)
        angle = mid % 360
        ha = "left" if 90 < angle < 270 else "right"
        rot = angle - 180 if 90 < angle < 270 else angle
        ax.text(
            lx,
            ly,
            labels[i],
            ha=ha,
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            rotation=rot,
            rotation_mode="anchor",
        )
        # Draw a thin tick line from arc rim to label when pushed out.
        if t > 0:
            rx, ry = _deg2xy(mid, radius + 0.01)
            ax.plot(
                [rx, lx],
                [ry, ly],
                color="#aaaaaa",
                linewidth=0.5,
                zorder=1,
            )

    # Update axes limits to accommodate outermost labels.
    max_offset = base_offset + (max(tier) if n > 0 else 0) * tier_step + 0.15
    lim = radius + max_offset
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # 6b. Highlight selected arcs
    if selected_indices is not None:
        selected_set = {str(idx) for idx in selected_indices}
        for i in range(n):
            if labels[i] in selected_set:
                _draw_arc_outline(
                    ax,
                    starts[i],
                    starts[i] + arc_degs[i],
                    radius - arc_width,
                    radius,
                    edgecolor=selection_color,
                    linewidth=selection_linewidth,
                    zorder=3,
                )

    # 7. Draw chord lines
    # For each node, sort its connections by angular proximity going
    # counter-clockwise around the circle.  This keeps nearby chords
    # adjacent on the arc and avoids unnecessary crossings.
    #
    # Draw lightest (smallest MI) to darkest so strong connections
    # paint on top.
    mi_row_sums = mi_matrix.sum(axis=1)
    arc_mids = [starts[i] + arc_degs[i] / 2.0 for i in range(n)]

    # Build per-node sorted connection lists.
    node_connections: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = mi_matrix[i, j]
            if val <= mi_threshold:
                continue
            node_connections[i].append((j, val))

        # Sort by counter-clockwise angular distance from node i.
        def _sort_key(jv: tuple[int, float], _mid: float = arc_mids[i]) -> float:
            return (_mid - arc_mids[jv[0]]) % 360

        node_connections[i].sort(key=_sort_key)

    # Allocate sub-segment midpoints per node in sorted order.
    cursor = starts.copy()
    allocated: dict[tuple[int, int], float] = {}

    for i in range(n):
        for j, val in node_connections[i]:
            span = arc_degs[i] * val / mi_row_sums[i] if mi_row_sums[i] > 0 else 0.0
            allocated[(i, j)] = cursor[i] + span / 2.0
            cursor[i] += span

    # Collect chords using the pre-allocated positions.
    chords = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in allocated:
                continue
            val = mi_matrix[i, j]
            chords.append((val, allocated[(i, j)], allocated[(j, i)]))

    # Sort lightest first so darkest (strongest) draws on top.
    chords.sort(key=lambda t: t[0])

    for val, angle_i, angle_j in chords:
        c = chord_cmap(val / mi_max)
        lw = min(np.sqrt(val) * line_scale, max_lw)

        _draw_chord_line(
            ax,
            angle_i,
            angle_j,
            color=c,
            linewidth=lw,
            radius=radius,
            arc_width=arc_width,
        )

    # 8. Colour-bar legends (horizontal, beneath the plot)
    sm_arc = plt.cm.ScalarMappable(
        cmap=arc_cmap,
        norm=plt.Normalize(vmin=0, vmax=s1_max),
    )
    sm_arc.set_array([])

    sm_chord = plt.cm.ScalarMappable(
        cmap=chord_cmap,
        norm=plt.Normalize(vmin=0, vmax=mi_max),
    )
    sm_chord.set_array([])

    if cax_arc is not None and cax_chord is not None:
        # Dedicated equal-width axes beneath the plot
        fig.colorbar(sm_arc, cax=cax_arc, orientation="horizontal")
        cax_arc.set_title("Single-orbital entropy", fontsize=10, pad=4)
        cax_arc.xaxis.set_ticks_position("top")
        cax_arc.xaxis.set_label_position("top")

        cbar_chord = fig.colorbar(sm_chord, cax=cax_chord, orientation="horizontal")
        cbar_chord.set_label("Mutual information", fontsize=10)
    else:
        # Fallback when drawn into a user-supplied axes
        fig.colorbar(sm_arc, ax=ax, fraction=0.04, pad=0.08, orientation="horizontal", location="bottom").set_label(
            "Single-orbital entropy",
            fontsize=10,
        )
        fig.colorbar(sm_chord, ax=ax, fraction=0.04, pad=0.12, orientation="horizontal", location="bottom").set_label(
            "Mutual information",
            fontsize=10,
        )

    # 9. Title and save
    if title is not None:
        ax.set_title(title, fontsize=14, pad=20)

    if save_path is not None:
        fig.savefig(
            str(save_path),
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent_background,
        )

    return fig, ax
