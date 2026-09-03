"""Loading for the Q# sources that exist only to be driven from the Python tests.

The shipped ``qdk_chemistry.utils.qsharp`` project must not carry test-only Q#
callables: they would be compiled into every context a user creates even though
production code never calls them. They live in ``tests/qsharp/`` instead, and
this module evaluates them into a *fresh* context on top of the shipped project.

They are evaluated into an existing context rather than declared as a dependent
Q# package because Q# ``internal`` visibility is package-scoped, and the drivers
reach for ``internal`` library callables (for example ``ApplySignedPowerSchedule``);
a dependency would put them in another package. ``qdk.Context`` does not resolve
local path dependencies anyway.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import qdk
from qdk import TargetProfile

from qdk_chemistry.utils.qsharp import create_qsharp_context

__all__ = ["TEST_SOURCE_ROOT", "create_test_qsharp_context"]

#: Directory holding the Q# sources that only the test layer compiles.
TEST_SOURCE_ROOT = Path(__file__).parent / "qsharp"


def create_test_qsharp_context(target_profile: TargetProfile = TargetProfile.Adaptive_RIF) -> qdk.Context:
    """Create an isolated context carrying the shipped Q# sources plus the test-only drivers.

    The context is always a new one, never the shared context, so evaluating the
    drivers into it cannot leak them into what production code paths resolve against.
    """
    context = create_qsharp_context(target_profile=target_profile)
    for path in sorted(TEST_SOURCE_ROOT.glob("*.qs")):
        context.eval(path.read_text(encoding="utf-8"))
    return context
