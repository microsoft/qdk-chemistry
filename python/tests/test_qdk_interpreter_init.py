"""Tests for QDK interpreter initialization."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import TargetProfile, init, qsharp

try:
    from qdk._interpreter import get_config
except ImportError:
    from qsharp._qsharp import get_config
import sys


def test_qdk_interpreter_init_with_target_profile():
    """Verify that QSHARP_UTILS re-initializes after user changes the Q# profile."""
    # User sets a different target profile before importing qdk_chemistry
    sys.modules.pop("qdk_chemistry.utils.qsharp", None)
    init(target_profile=TargetProfile.Base)
    assert get_config().get_target_profile() == "base"

    # Simulate fresh import of qdk_chemistry.utils.qsharp
    from qdk_chemistry.utils.qsharp import QSHARP_UTILS  # noqa: PLC0415

    # After access, the profile should be back to adaptive_rif
    assert get_config().get_target_profile() == "adaptive_rif"
    # QSHARP_UTILS should transparently re-initialize and still work
    assert getattr(QSHARP_UTILS, "StatePreparation", None) is not None


def test_qsharp_utils_survives_user_reinit():
    """Verify QSHARP_UTILS works after a user re-initializes the Q# context."""
    # Simulate fresh import
    from qdk_chemistry.utils.qsharp import QSHARP_UTILS  # noqa: PLC0415

    # User re-initializes with a different profile, disposing the context
    init(target_profile=TargetProfile.Base)
    assert get_config().get_target_profile() == "base"

    # QSHARP_UTILS should recover automatically
    assert getattr(QSHARP_UTILS, "BinaryEncoding", None) is not None
    assert getattr(QSHARP_UTILS, "StatePreparation", None) is not None
    assert get_config().get_target_profile() == "adaptive_rif"


def test_user_qsharp_eval_alongside_qdk_chemistry():
    """Verify that user Q# code works alongside qdk_chemistry Q# utilities."""
    # Simulate fresh import
    from qdk_chemistry.utils.qsharp import QSHARP_UTILS  # noqa: PLC0415

    # Access QSHARP_UTILS first to ensure chemistry Q# is loaded
    assert getattr(QSHARP_UTILS, "StatePreparation", None) is not None

    # User evaluates their own Q# code in the same context
    qsharp.eval("operation MyCustomOp() : Int { return 42; }")
    result = qsharp.eval("MyCustomOp()")
    assert result == 42

    # Chemistry utilities should still work in the same context
    assert getattr(QSHARP_UTILS, "CircuitComposition", None) is not None
