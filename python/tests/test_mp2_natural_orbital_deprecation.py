"""Regression tests for MP2 natural-orbital localizer deprecation warnings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import subprocess
import sys
import warnings

DEPRECATION_MESSAGE = "MP2NaturalOrbitalLocalizer is deprecated"


def test_registry_stub_generation_does_not_warn_about_mp2_natural_orbital_localizer(tmp_path):
    """Registry stub generation does not report deprecated localizer use."""
    stub_file = tmp_path / "registry.pyi"
    stub_file.write_text("# placeholder\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "always",
            "-c",
            """\
import os
from pathlib import Path
import sys

os.environ["QDK_CHEMISTRY_DOCS"] = "1"

import qdk_chemistry
from qdk_chemistry.algorithms import registry
from qdk_chemistry.utils import Logger

stub_dir = Path(sys.argv[1])
registry.__file__ = str(stub_dir / "registry.py")
qdk_chemistry._STUBGEN_BLOCK_MARKER = stub_dir / ".no-stubgen"
Logger.set_global_level("warn")
qdk_chemistry._generate_registry_stubs()
""",
            str(tmp_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Literal['qdk_mp2_natural_orbitals']" in stub_file.read_text(encoding="utf-8")
    assert DEPRECATION_MESSAGE not in result.stdout
    assert DEPRECATION_MESSAGE not in result.stderr


def test_explicit_mp2_natural_orbital_localizer_creation_warns_once(capfd):
    """Explicit registry creation retains its user-facing deprecation warning."""
    from qdk_chemistry.algorithms import create  # noqa: PLC0415
    from qdk_chemistry.utils import Logger  # noqa: PLC0415

    previous_level = Logger.get_global_level()
    try:
        Logger.set_global_level("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            localizer = create("orbital_localizer", "qdk_mp2_natural_orbitals")
    finally:
        Logger.set_global_level(previous_level)

    matching_warnings = [warning for warning in caught if DEPRECATION_MESSAGE in str(warning.message)]
    captured = capfd.readouterr()

    assert localizer.name() == "qdk_mp2_natural_orbitals"
    assert len(matching_warnings) == 1
    assert (captured.out + captured.err).count(DEPRECATION_MESSAGE) == 1
