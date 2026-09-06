"""Generate the energy-accounting figure for the molecular QPE tutorial."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import subprocess
from pathlib import Path

from tutorial_qpe_svg import (
    add_accessibility_metadata,
    figure_descriptions,
    source_sha256,
)

DOT_FILENAME = "tutorial_qpe_energy_accounting.dot"
SVG_FILENAME = "tutorial_qpe_energy_accounting.svg"


def main() -> None:
    """Render the energy-accounting DOT source as an accessible SVG."""
    dot_path = Path(__file__).with_name(DOT_FILENAME)
    output_path = Path(__file__).with_name(SVG_FILENAME)
    result = subprocess.run(
        ["dot", "-Tsvg", str(dot_path)],
        check=True,
        capture_output=True,
        encoding="utf-8",
        text=True,
    )
    output_path.write_text(
        add_accessibility_metadata(
            result.stdout,
            identifier="tutorial-qpe-energy-accounting",
            title="Molecular-energy reconstruction and comparison",
            description=figure_descriptions()[SVG_FILENAME],
            source_hash=source_sha256(
                dot_path,
                Path(__file__),
                Path(__file__).with_name("tutorial_qpe_svg.py"),
            ),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
