"""Run the installed stub generator without package-directory shadowing."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import subprocess
import sys
import tempfile
from pathlib import Path


def run_stub_generator(chemistry_dir: Path) -> None:
    """Generate extension stubs without selecting the generator from the target directory."""
    chemistry_dir = chemistry_dir.resolve()
    # The extension directory must not select the generator module.
    bootstrap = "import sys; from pybind11_stubgen import main; sys.path.insert(0, sys.argv.pop(1)); main()"
    with tempfile.TemporaryDirectory(prefix="qdk-stubgen-") as working_dir:
        subprocess.run(
            [sys.executable, "-c", bootstrap, str(chemistry_dir), "_core", "-o", str(chemistry_dir)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=working_dir,
        )
