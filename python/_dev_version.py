"""Scikit-build-core metadata provider: appends +local for from-source dev builds."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import re
import subprocess
from pathlib import Path

# Set by the release pipeline to force a clean, publishable version (no +local).
_RELEASE_BUILD_ENV = "QDK_CHEMISTRY_RELEASE_BUILD"
_FALSEY = frozenset({"", "0", "false", "no", "off"})


def _is_release_build() -> bool:
    """Whether to emit a clean version (no +local)."""
    return os.environ.get(_RELEASE_BUILD_ENV, "").strip().lower() not in _FALSEY


def dynamic_metadata(field, settings):
    """Return version from VERSION file, appending +local if HEAD is not release-tagged."""
    assert field == "version"
    version_file = Path(settings.get("input", "../VERSION"))
    version = version_file.read_text().strip()

    # If VERSION already carries a pre-release/dev suffix (e.g. CI set it), use as-is.
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        return version

    # Release builds publish the bare version; never tag it +local.
    if _is_release_build():
        return version

    repo_root = version_file.resolve().parent

    # Only consider appending +local inside a git checkout.
    if not (repo_root / ".git").exists():
        return version

    # Check whether HEAD sits on a tag that matches the VERSION content.
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        if result.returncode == 0:
            tag = result.stdout.strip()
            if tag in (version, f"v{version}"):
                return version
    except FileNotFoundError:
        # git not installed — cannot verify release tag, treat as local build
        pass

    return f"{version}+local"
