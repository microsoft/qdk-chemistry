"""Generate accessible molecular-QPE SVG figures from committed DOT sources."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import subprocess
from pathlib import Path

from tutorial_qpe_svg import (
    DIAGRAMS_DIR,
    add_accessibility_metadata,
    figure_descriptions,
    source_sha256,
)

TITLES = {
    "tutorial_qpe_workflow": "Molecular QPE tutorial workflow",
    "tutorial_qpe_wavefunction_hierarchy": "Wavefunction representation hierarchy",
    "tutorial_qpe_orbital_partition": "Active-space orbital partition",
    "tutorial_qpe_jordan_wigner_parity": "Jordan-Wigner parity mapping",
    "tutorial_qpe_iqpe_iteration": "Iterative phase-estimation iteration",
}


def accessible_svg(dot_path: Path, title: str, description: str) -> str:
    """Render one DOT source and add stable accessibility metadata."""
    result = subprocess.run(
        ["dot", "-Tsvg", str(dot_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    svg_id = dot_path.stem.replace("_", "-")
    return add_accessibility_metadata(
        result.stdout,
        identifier=svg_id,
        title=title,
        description=description,
        source_hash=source_sha256(
            dot_path,
            Path(__file__),
            Path(__file__).with_name("tutorial_qpe_svg.py"),
        ),
    )


def main() -> None:
    """Regenerate all committed molecular-QPE Graphviz SVGs."""
    descriptions = figure_descriptions()
    expected = {f"{stem}.svg" for stem in TITLES}
    if not expected.issubset(descriptions):
        raise ValueError(
            "Expected one documented figure for each generated Graphviz SVG; "
            f"documented={sorted(descriptions)}, expected={sorted(expected)}"
        )

    for stem, title in TITLES.items():
        dot_path = DIAGRAMS_DIR / f"{stem}.dot"
        svg_path = DIAGRAMS_DIR / f"{stem}.svg"
        svg_path.write_text(
            accessible_svg(dot_path, title, descriptions[svg_path.name]),
            encoding="utf-8",
        )
        print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
