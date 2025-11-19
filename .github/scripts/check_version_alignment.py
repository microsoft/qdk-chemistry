#!/usr/bin/env python3
"""Check that all version strings in the codebase are aligned.

This script verifies that version strings across multiple files are consistent:
- __version__ in python/src/qdk_chemistry/__init__.py
- VERSION in python/CMakeLists.txt
- VERSION in cpp/CMakeLists.txt
- release in docs/source/conf.py

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
from typing import NamedTuple, TypedDict


class VersionLocation(NamedTuple):
    """Location of a version string in the codebase."""

    file_path: Path
    pattern: str
    description: str
    is_cmake: bool = False  # Whether this uses CMake version format


class VersionInfo(TypedDict):
    """Information about a version found in the codebase."""

    version: str
    location: VersionLocation


def extract_version(file_path: Path, pattern: str) -> str | None:
    """Extract version string from a file using a regex pattern.

    Args:
        file_path: Path to the file to search
        pattern: Regex pattern with a capture group for the version

    Returns:
        The extracted version string, or None if not found
    """
    if not file_path.exists():
        return None

    try:
        content = file_path.read_text()
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)

    return None


def normalize_cmake_version(cmake_version: str) -> str:
    """Convert CMake version format to Python version format.

    CMake versions use 4 components where the 4th is the pre-release number:
    - 1.0.0.1 -> 1.0.0-rc1
    - 1.0.0.2 -> 1.0.0-rc2
    - 1.0.0.0 -> 1.0.0 (no pre-release)

    Args:
        cmake_version: Version string in CMake format (e.g., "1.0.0.1")

    Returns:
        Version string in Python format (e.g., "1.0.0-rc1")
    """
    parts = cmake_version.split(".")
    if len(parts) == 4:
        tweak = int(parts[3])
        if tweak > 0:
            return f"{'.'.join(parts[:3])}-rc{tweak}"
        else:
            return ".".join(parts[:3])
    return cmake_version


def python_to_cmake_version(python_version: str) -> str:
    """Convert Python version format to CMake version format.

    Python versions with pre-release use hyphen notation:
    - 1.0.0-rc1 -> 1.0.0.1
    - 1.0.0-rc2 -> 1.0.0.2
    - 1.0.0 -> 1.0.0.0

    Args:
        python_version: Version string in Python format (e.g., "1.0.0-rc1")

    Returns:
        Version string in CMake format (e.g., "1.0.0.1")
    """
    # Match version with optional pre-release suffix
    match = re.match(r"^(\d+\.\d+\.\d+)(?:-rc(\d+))?$", python_version)
    if match:
        base_version = match.group(1)
        rc_number = match.group(2)
        if rc_number:
            return f"{base_version}.{rc_number}"
        else:
            return f"{base_version}.0"
    return python_version


def check_versions() -> int:
    """Check that all version strings are aligned.

    Returns:
        0 if all versions match, 1 if there are mismatches
    """
    # Define the root directory (repository root)
    repo_root = Path(__file__).parent.parent.parent

    # Define version locations
    version_locations = [
        VersionLocation(
            file_path=repo_root / "python/src/qdk_chemistry/__init__.py",
            pattern=r'__version__\s*=\s*["\']([^"\']+)["\']',
            description="Python __version__",
            is_cmake=False,
        ),
        VersionLocation(
            file_path=repo_root / "python/CMakeLists.txt",
            pattern=r"project\(qdk_chemistry_python\s+VERSION\s+([^\s)]+)",
            description="Python CMakeLists.txt VERSION",
            is_cmake=True,
        ),
        VersionLocation(
            file_path=repo_root / "cpp/CMakeLists.txt",
            pattern=r"project\(qdk\s+VERSION\s+([^\s)]+)",
            description="C++ CMakeLists.txt VERSION",
            is_cmake=True,
        ),
        VersionLocation(
            file_path=repo_root / "docs/source/conf.py",
            pattern=r'release\s*=\s*["\']([^"\']+)["\']',
            description="Sphinx release",
            is_cmake=False,
        ),
    ]

    # Extract all versions
    versions: dict[str, VersionInfo] = {}
    missing: list[VersionLocation] = []

    for loc in version_locations:
        version = extract_version(loc.file_path, loc.pattern)
        if version is None:
            missing.append(loc)
        else:
            # Normalize CMake versions to Python format for comparison
            normalized = normalize_cmake_version(version) if loc.is_cmake else version
            versions[loc.description] = {
                "version": normalized,
                "location": loc,
            }

    # Report missing versions
    if missing:
        print("❌ Version check failed: Missing version strings", file=sys.stderr)
        print(file=sys.stderr)
        for loc in missing:
            print(f"  ⚠️  Missing: {loc.description}", file=sys.stderr)
            print(f"      File: {loc.file_path}", file=sys.stderr)
        return 1

    # Get the canonical version (from Python __init__.py)
    canonical = versions.get("Python __version__")
    if not canonical:
        print("❌ Version check failed: Canonical version not found", file=sys.stderr)
        return 1

    canonical_version = canonical["version"]

    # Check alignment
    misaligned = []
    for desc, ver_info in versions.items():
        if desc == "Python __version__":
            continue  # Skip the canonical version itself

        actual = ver_info["version"]

        if actual != canonical_version:
            misaligned.append(
                {
                    "description": desc,
                    "expected": canonical_version,
                    "actual": actual,
                    "file": ver_info["location"].file_path,
                }
            )

    # Report results
    if misaligned:
        print(
            "❌ Version check failed: Version strings are not aligned", file=sys.stderr
        )
        print(file=sys.stderr)
        print(f"  ✓  Canonical version: {canonical_version}", file=sys.stderr)
        print(
            f"      Location: {canonical['location'].file_path}",
            file=sys.stderr,
        )
        print(file=sys.stderr)

        for item in misaligned:
            print(f"  ✗  {item['description']}: {item['actual']}", file=sys.stderr)
            print(f"      Expected: {item['expected']}", file=sys.stderr)
            print(f"      File: {item['file']}", file=sys.stderr)
            print(file=sys.stderr)

        print(
            "Please update the version strings to match the canonical version.",
            file=sys.stderr,
        )
        return 1

    # All versions aligned
    print(f"✓ All version strings are aligned: {canonical_version}")
    return 0


if __name__ == "__main__":
    sys.exit(check_versions())
