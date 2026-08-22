"""Generate the phase-wrapping figure and energy-grid table for the QPE tutorial."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REFERENCE_ENERGY_HARTREE = -9.653276065987
TARGET_OFFSET_HARTREE = 1e-3
NUM_PHASE_BITS = 6
SELECTED_GRID_INDEX = 16


def phase_to_energy(phase_fraction: float, evolution_time: float) -> float:
    """Convert a phase fraction to the signed QDK/Chemistry energy branch."""
    signed_fraction = phase_fraction if phase_fraction <= 0.5 else phase_fraction - 1.0
    return -2 * np.pi * signed_fraction / evolution_time


def energy_grid_rows(
    evolution_time: float,
) -> list[tuple[int | None, str | None, float]]:
    """Calculate neighboring grid energies with the known reference inserted."""
    grid_size = 2**NUM_PHASE_BITS

    def grid_energy(index: int) -> float:
        return phase_to_energy(index / grid_size, evolution_time)

    return [
        (17, f"{17:0{NUM_PHASE_BITS}b}", grid_energy(17)),
        (None, None, REFERENCE_ENERGY_HARTREE),
        (16, f"{16:0{NUM_PHASE_BITS}b}", grid_energy(16)),
        (15, f"{15:0{NUM_PHASE_BITS}b}", grid_energy(15)),
    ]


def render_energy_grid_table(
    rows: list[tuple[int | None, str | None, float]],
) -> str:
    """Render calculated grid rows as a Sphinx list-table fragment."""
    lines = [
        (
            ".. list-table:: Six-bit active-energy grid near the reference. "
            "Here :math:`k` is the integer grid index, so "
            ":math:`\\varphi_k=k/2^6=k/64`; its six-bit binary representation "
            "is the measured bitstring."
        ),
        "   :header-rows: 1",
        "   :widths: 24 24 30",
        "   :align: center",
        "",
        "   * - Grid point",
        "     - Bitstring",
        "     - Active energy (:math:`E_{\\mathrm{h}}`)",
    ]
    for index, bitstring, energy in rows:
        if index is None:
            grid_label = "Known reference (not a grid point)"
            bitstring_label = "Not applicable"
        else:
            selected = " (selected)" if index == SELECTED_GRID_INDEX else ""
            grid_label = f":math:`k={index}`{selected}"
            bitstring_label = f"``{bitstring}``"
        lines.extend(
            [
                f"   * - {grid_label}",
                f"     - {bitstring_label}",
                f"     - :math:`{energy:.12f}`",
            ]
        )
    return "\n".join(lines) + "\n"


def generate_phase_wrapping_figure(evolution_time: float) -> plt.Figure:
    """Show phase wrapping, the signed-energy branch, and aliasing."""
    figure, axis = plt.subplots(figsize=(8.8, 4.4), layout="constrained")

    lower_phases = np.linspace(0.0, 0.5, 100)
    upper_phases = np.linspace(0.5, 1.0, 100)
    lower_energies = -2.0 * lower_phases
    upper_energies = 2.0 * (1.0 - upper_phases)

    axis.plot(lower_phases, lower_energies, color="#00796B", linewidth=3.0)
    axis.plot(upper_phases, upper_energies, color="#7B1FA2", linewidth=3.0)
    axis.scatter([0.5], [-1.0], color="#00796B", s=80, zorder=4)
    axis.scatter(
        [0.5],
        [1.0],
        facecolors="white",
        edgecolors="#7B1FA2",
        s=80,
        linewidth=2.0,
        zorder=4,
    )
    axis.plot([0.5, 0.5], [-1.0, 1.0], color="#78909C", linestyle="--", linewidth=1.3)

    axis.set_xlim(-0.02, 1.08)
    axis.set_ylim(-1.18, 1.18)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "1/4", "1/2", "3/4", "1 → 0"])
    axis.set_yticks(
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        [r"$-\pi/t$", r"$-\pi/(2t)$", "$0$", r"$+\pi/(2t)$", r"$+\pi/t$"],
    )
    axis.set_xlabel(r"Reported phase fraction $\varphi\in[0,1)$")
    axis.set_ylabel("Signed reconstructed energy")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#ECEFF1", linewidth=0.8)

    axis.text(
        0.48,
        -1.08,
        "included boundary",
        ha="right",
        va="top",
        color="#00695C",
        fontsize=9,
    )
    axis.text(
        0.52,
        1.08,
        "excluded boundary",
        ha="left",
        va="bottom",
        color="#6A1B9A",
        fontsize=9,
    )
    alias_bracket_x = 1.045
    axis.annotate(
        "",
        xy=(alias_bracket_x, 0.94),
        xytext=(alias_bracket_x, -0.94),
        arrowprops={"arrowstyle": "<->", "color": "#C62828", "linewidth": 1.3},
        annotation_clip=False,
    )
    return figure


def main() -> None:
    """Write committed visual and table artifacts from the Chapter 6 constants."""
    grid_size = 2**NUM_PHASE_BITS
    selected_phase = SELECTED_GRID_INDEX / grid_size
    selected_energy = REFERENCE_ENERGY_HARTREE + TARGET_OFFSET_HARTREE
    evolution_time = -2 * np.pi * selected_phase / selected_energy
    reference_phase = (-evolution_time * REFERENCE_ENERGY_HARTREE / (2 * np.pi)) % 1.0
    energy_spacing = 2 * np.pi / (evolution_time * grid_size)

    assert np.isclose(evolution_time, 0.162738437655, atol=5e-13)
    assert np.isclose(reference_phase, 0.250025901, atol=5e-10)
    assert np.isclose(energy_spacing, 0.603267254, atol=5e-10)
    assert np.isclose(
        phase_to_energy(selected_phase, evolution_time) - REFERENCE_ENERGY_HARTREE,
        TARGET_OFFSET_HARTREE,
        atol=1e-12,
    )

    rows = energy_grid_rows(evolution_time)
    assert [index for index, _, _ in rows] == [17, None, 16, 15]
    assert np.isclose(rows[2][2] - rows[1][2], TARGET_OFFSET_HARTREE, atol=1e-12)

    figures = {
        "tutorial_qpe_phase_wrapping.png": generate_phase_wrapping_figure(
            evolution_time
        )
    }
    for filename, figure in figures.items():
        output_path = Path(__file__).with_name(filename)
        figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Wrote {output_path}")

    table_path = Path(__file__).with_name("tutorial_qpe_phase_grid_table.rst")
    table_path.write_text(render_energy_grid_table(rows), encoding="utf-8")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
