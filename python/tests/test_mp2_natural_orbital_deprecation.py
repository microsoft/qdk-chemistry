"""Regression tests for MP2 natural-orbital localizer deprecation warnings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import subprocess
import sys
import warnings

DEPRECATION_MESSAGE = "MP2NaturalOrbitalLocalizer is deprecated"


def test_package_import_does_not_warn_about_mp2_natural_orbital_localizer():
    """Importing the package does not report deprecated localizer use."""
    result = subprocess.run(
        [sys.executable, "-W", "always", "-c", "import qdk_chemistry"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
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
