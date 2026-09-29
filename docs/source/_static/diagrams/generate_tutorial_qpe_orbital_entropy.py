"""Generate the orbital-entropy selection figure for the molecular QPE tutorial."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys
from pathlib import Path

import matplotlib as mpl

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "python"
sys.path.insert(0, str(EXAMPLES_DIR))

from qdk_chemistry.utils import Logger  # noqa: E402
from tutorial_choose_active_space import (  # noqa: E402
    plot_orbital_entropy_selection,
    run_active_space_workflow,
)
from tutorial_qpe_svg import (  # noqa: E402
    add_accessibility_metadata,
    figure_descriptions,
    source_sha256,
)

PLOT_BACKGROUND = "#FFFFFF"
SVG_FILENAME = "tutorial_qpe_orbital_entropy.svg"


def main() -> None:
    """Run the shared workflow and write the committed entropy figure."""
    description = figure_descriptions()[SVG_FILENAME]
    Logger.set_global_level(Logger.LogLevel.off)
    result = run_active_space_workflow()
    with mpl.rc_context(
        {
            "svg.fonttype": "path",
            "svg.hashsalt": "qdk-chemistry-orbital-entropy",
        }
    ):
        figure = plot_orbital_entropy_selection(result)
        figure.patch.set_facecolor(PLOT_BACKGROUND)
        for axis in figure.axes:
            axis.set_facecolor(PLOT_BACKGROUND)
        output_path = Path(__file__).with_name(SVG_FILENAME)
        figure.savefig(
            output_path,
            format="svg",
            bbox_inches="tight",
            facecolor=PLOT_BACKGROUND,
            metadata={"Date": None},
        )
        output_path.write_text(
            add_accessibility_metadata(
                output_path.read_text(encoding="utf-8"),
                identifier="tutorial-qpe-orbital-entropy",
                title="Natural-orbital entropy selection",
                description=description,
                source_hash=source_sha256(
                    Path(__file__),
                    Path(__file__).with_name("tutorial_qpe_svg.py"),
                    EXAMPLES_DIR / "tutorial_choose_active_space.py",
                ),
            ),
            encoding="utf-8",
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
