"""Tests for QDK interpreter initialization."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys

from qdk import TargetProfile, init

try:
    from qdk._interpreter import get_config
except ImportError:
    from qsharp._qsharp import get_config


def test_default_qdk_interpreter_init():
    init(target_profile=TargetProfile.Unrestricted)
    sys.modules.pop("qdk_chemistry", None)
    from qdk_chemistry import _QDK_INTERPRETER_PROFILE  # noqa: PLC0415

    assert _QDK_INTERPRETER_PROFILE == "base"


def test_qdk_interpreter_init_with_target_profile():
    sys.modules.pop("qdk_chemistry", None)
    init(target_profile=TargetProfile.Adaptive_RIF)
    user_profile = get_config().get_target_profile()
    assert user_profile == "adaptive_rif"

    from qdk_chemistry import _QDK_INTERPRETER_PROFILE  # noqa: PLC0415

    assert user_profile == _QDK_INTERPRETER_PROFILE

    init(target_profile=TargetProfile.Base)

    from qdk_chemistry.utils.qsharp import QSHARP_UTILS  # noqa: PLC0415

    assert getattr(QSHARP_UTILS, "StatePreparation", None) is not None


def test_qsharp_utils_switch_between_base_and_adaptive_profiles():
    """Load the Base-compatible subset, then restore the complete adaptive project."""
    from qdk_chemistry.utils.qsharp import get_qsharp_utils  # noqa: PLC0415

    base_utils = get_qsharp_utils(target_profile=TargetProfile.Base)
    assert getattr(base_utils, "StatePreparation", None) is not None

    adaptive_utils = get_qsharp_utils(target_profile=TargetProfile.Adaptive_RIF)
    assert getattr(adaptive_utils, "SOSSAWalk", None) is not None
