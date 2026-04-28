"""Tests for QDK interpreter initialization."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys

from qdk import TargetProfile, init
from qsharp._qsharp import get_config


def test_qdk_interpreter_init_with_target_profile():
    # Remove qsharp init module from cache to force re-execution of its __init__.py
    sys.modules.pop("qdk_chemistry.utils.qsharp", None)

    init(target_profile=TargetProfile.Base)
    user_profile = get_config().get_target_profile()
    assert user_profile == "base"

    from qdk_chemistry.utils.qsharp import QSHARP_UTILS  # noqa: PLC0415

    assert get_config().get_target_profile() == "adaptive_rif"
    assert getattr(QSHARP_UTILS, "StatePreparation", None) is not None
