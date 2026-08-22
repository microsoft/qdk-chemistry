"""Generate the orbital-entropy selection figure for the molecular QPE tutorial."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "python"
sys.path.insert(0, str(EXAMPLES_DIR))

from qdk_chemistry.utils import Logger  # noqa: E402
from tutorial_choose_active_space import (  # noqa: E402
    plot_orbital_entropy_selection,
    run_active_space_workflow,
)


def main() -> None:
    """Run the shared workflow and write the committed entropy figure."""
    Logger.set_global_level(Logger.LogLevel.off)
    result = run_active_space_workflow()
    figure = plot_orbital_entropy_selection(result)
    output_path = Path(__file__).with_name("tutorial_qpe_orbital_entropy.png")
    figure.savefig(output_path, dpi=200, bbox_inches="tight", transparent=True)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
