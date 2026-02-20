#!/usr/bin/env python3
"""Check that all version strings in the codebase are aligned.

This script verifies that the VERSION file exists and is valid, and that
all version-dependent files are configured to read from it:
- VERSION (canonical source of truth)
- cpp/CMakeLists.txt (reads from VERSION via file(READ ...))
- python/CMakeLists.txt (reads from VERSION via file(READ ...))
- python/pyproject.toml (uses scikit-build-core metadata provider)
- python/src/qdk_chemistry/__init__.py (uses importlib.metadata with fallback)
- python/src/qdk_chemistry/utils/telemetry.py (uses importlib.metadata with fallback)
- docs/source/conf.py (uses Path.read_text())
- docs/source/changelog.rst (must have entry for current version)

Exit codes:
    0: All versions are aligned
    1: Versions are misaligned (with details printed)
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re
import sys
from pathlib import Path


def check_versions() -> int:
    """Check that all version strings are aligned.

    Returns:
        0 if all versions match, 1 if there are mismatches
    """
    # Define the root directory (repository root)
    repo_root = Path(__file__).parent.parent.parent

    # Read the canonical version from the VERSION file
    version_file = repo_root / "VERSION"
    if not version_file.exists():
        print("Version check failed: VERSION file not found", file=sys.stderr)
        print(f"      Expected at: {version_file}", file=sys.stderr)
        return 1

    canonical_version = version_file.read_text().strip()
    if not canonical_version:
        print("Version check failed: VERSION file is empty", file=sys.stderr)
        return 1

    # Validate version format: X.Y.Z or X.Y.Z.T (4-component CMake-compatible)
    # The optional .T suffix is a numeric build/tweak component for CMake compatibility
    if not re.match(r"^\d+\.\d+\.\d+(\.\d+)?$", canonical_version):
        print(
            f"Version check failed: Invalid version format '{canonical_version}'",
            file=sys.stderr,
        )
        print("      Expected format: X.Y.Z or X.Y.Z.T", file=sys.stderr)
        return 1

    errors = []

    # Check 1: CMakeLists.txt files read from VERSION (not hardcoded)
    cmake_files = [
        ("python/CMakeLists.txt", repo_root / "python/CMakeLists.txt"),
        ("cpp/CMakeLists.txt", repo_root / "cpp/CMakeLists.txt"),
    ]

    for cmake_label, cmake_file in cmake_files:
        if not cmake_file.exists():
            errors.append(f"{cmake_label}: file not found")
            continue

        content = cmake_file.read_text()
        # Accept either direct path or via a variable
        if "/../VERSION" not in content:
            errors.append(f"{cmake_label}: does not read from VERSION file")
        if "CMAKE_CONFIGURE_DEPENDS" not in content:
            errors.append(
                f"{cmake_label}: missing CMAKE_CONFIGURE_DEPENDS for VERSION file "
                "(CMake won't reconfigure when VERSION changes)"
            )

    # Check 2: pyproject.toml uses scikit-build-core metadata provider
    pyproject = repo_root / "python/pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        if (
            'metadata.version.provider = "scikit_build_core.metadata.regex"'
            not in content
        ):
            errors.append(
                "pyproject.toml: missing scikit-build-core metadata.version.provider"
            )
        if 'metadata.version.input = "../VERSION"' not in content:
            errors.append(
                "pyproject.toml: missing metadata.version.input pointing to VERSION"
            )
        if 'dynamic = ["version"]' not in content:
            errors.append('pyproject.toml: missing dynamic = ["version"]')
    else:
        errors.append("pyproject.toml: file not found")

    # Check 3: __init__.py uses importlib.metadata with fallback
    init_py = repo_root / "python/src/qdk_chemistry/__init__.py"
    if init_py.exists():
        content = init_py.read_text()
        if (
            "from importlib.metadata import" not in content
            or "PackageNotFoundError" not in content
        ):
            errors.append(
                "__init__.py: missing importlib.metadata with PackageNotFoundError fallback"
            )
        if '/ "VERSION"' not in content:
            errors.append("__init__.py: fallback does not read from VERSION file")
    else:
        errors.append("__init__.py: file not found")

    # Check 4: telemetry.py uses importlib.metadata with fallback
    telemetry_py = repo_root / "python/src/qdk_chemistry/utils/telemetry.py"
    if telemetry_py.exists():
        content = telemetry_py.read_text()
        if (
            "from importlib.metadata import" not in content
            or "PackageNotFoundError" not in content
        ):
            errors.append(
                "telemetry.py: missing importlib.metadata with PackageNotFoundError fallback"
            )
    else:
        errors.append("telemetry.py: file not found")

    # Check 5: docs/source/conf.py reads from VERSION
    conf_py = repo_root / "docs/source/conf.py"
    if conf_py.exists():
        content = conf_py.read_text()
        if '/ "VERSION"' not in content:
            errors.append("docs/source/conf.py: does not read from VERSION file")
    else:
        errors.append("docs/source/conf.py: file not found")

    # Check 6: docs/source/changelog.rst has an entry for the current version
    changelog_rst = repo_root / "docs/source/changelog.rst"
    if changelog_rst.exists():
        content = changelog_rst.read_text()
        # Look for "Version X.Y.Z" header in rst format
        version_header = f"Version {canonical_version}"
        if version_header not in content:
            errors.append(
                f"docs/source/changelog.rst: missing entry for '{version_header}'"
            )
    else:
        errors.append("docs/source/changelog.rst: file not found")

    # Report results
    if errors:
        print("FAIL: Version check failed:", file=sys.stderr)
        print(file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "All version references should read from the VERSION file.",
            file=sys.stderr,
        )
        return 1

    # All versions aligned
    print(f"OK: All version strings are aligned: {canonical_version}")
    print(f"  Source: {version_file}")
    print("  Verified:")
    print("    - cpp/CMakeLists.txt")
    print("    - python/CMakeLists.txt")
    print("    - python/pyproject.toml")
    print("    - python/src/qdk_chemistry/__init__.py")
    print("    - python/src/qdk_chemistry/utils/telemetry.py")
    print("    - docs/source/conf.py")
    print("    - docs/source/changelog.rst")
    return 0


if __name__ == "__main__":
    sys.exit(check_versions())
