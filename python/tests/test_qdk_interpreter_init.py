"""Tests for QDK interpreter initialization."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys

from qdk import TargetProfile, init
from qsharp._qsharp import get_config


def test_default_qdk_interpreter_init():
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
