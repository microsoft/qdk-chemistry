"""Resolve the path to the shared example data files (``examples/data/`` in the repo root).

Import ``EXAMPLES_DATA_DIR`` in any doc example script to locate molecular
structure files without hard-coding a fragile relative path.

Usage in doc examples::

    from examples_data import EXAMPLES_DATA_DIR
    structure = Structure.from_xyz_file(EXAMPLES_DATA_DIR / "water.structure.xyz")

When running under pytest the test harness sets
``QDK_CHEMISTRY_EXAMPLES_DATA_DIR`` to the canonical ``examples/data/`` folder
in the repository root, so examples work regardless of working directory.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
from pathlib import Path

# The environment variable is set by the test harness (test_docs_examples.py)
# to point at the canonical `examples/data/` directory in the repo root.
# When not set (e.g. running a script directly from the docs layout), fall back
# to the sibling `../data/` directory next to the `python/` folder.
EXAMPLES_DATA_DIR: Path = Path(
    os.environ.get(
        "QDK_CHEMISTRY_EXAMPLES_DATA_DIR",
        str(Path(__file__).resolve().parent / ".." / "data"),
    )
).resolve()
