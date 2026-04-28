"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from pathlib import Path

import qdk
from qdk import init as qdk_init
from qsharp._qsharp import get_config as get_qdk_profile_config

from qdk_chemistry.utils import Logger

__all__ = ["QSHARP_UTILS"]

_QS_DIR = Path(__file__).parent

# Ensure the Q# interpreter uses Adaptive_RIF
qdk_config = get_qdk_profile_config()
_target_profile = qdk.TargetProfile.Adaptive_RIF

if qdk_config.get_target_profile() != "adaptive_rif":
    Logger.debug(
        f"QDK interpreter profile set to '{_target_profile}'. "
        "If you imported Q# code before this module was loaded, please re-import it, "
        "or set your target profile before importing qdk_chemistry."
    )

qdk_init(project_root=_QS_DIR, target_profile=_target_profile)
QSHARP_UTILS = qdk.code.QDKChemistry.Utils
