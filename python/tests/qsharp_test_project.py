"""Staging for the Q# sources that exist only to be driven from the Python tests.

The shipped ``qdk_chemistry.utils.qsharp`` project must not carry test-only Q#
callables: they would be compiled into every user context even though production
code never calls them. They live in ``tests/qsharp/`` instead, and this module
stages them into a throwaway Q# project together with the shipped sources.

They are staged into *one* project rather than declared as a dependent package
because Q# ``internal`` visibility is package-scoped, and the drivers reach for
``internal`` library callables (for example ``ApplySignedPowerSchedule``).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import atexit
import shutil
import tempfile
from functools import cache
from pathlib import Path

import qdk
from qdk import TargetProfile

import qdk_chemistry.utils.qsharp as qsharp_package

__all__ = ["TEST_SOURCE_ROOT", "create_test_qsharp_context"]

#: Directory holding the Q# sources that only the test layer compiles.
TEST_SOURCE_ROOT = Path(__file__).parent / "qsharp"

_PACKAGE_ROOT = Path(qsharp_package.__file__).parent


@cache
def _test_project_root() -> str:
    """Stage the shipped Q# sources plus the test-only ones as one Q# project."""
    root = Path(tempfile.mkdtemp(prefix="qdk-chemistry-qsharp-tests-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    shutil.copyfile(_PACKAGE_ROOT / "qsharp.json", root / "qsharp.json")
    source_dir = shutil.copytree(_PACKAGE_ROOT / "src", root / "src")
    for path in sorted(TEST_SOURCE_ROOT.glob("*.qs")):
        staged = Path(source_dir) / path.name
        if staged.exists():
            msg = f"test Q# source {path.name} would overwrite a shipped source of the same name"
            raise RuntimeError(msg)
        shutil.copyfile(path, staged)
    return str(root)


def create_test_qsharp_context(target_profile: TargetProfile = TargetProfile.Adaptive_RIF) -> qdk.Context:
    """Create an isolated context carrying both the shipped and the test-only Q# sources."""
    return qdk.Context(project_root=_test_project_root(), target_profile=target_profile)
