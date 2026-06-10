"""QDK/Chemistry Q# Utilities Module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from pathlib import Path

import qdk
from qdk import init as qdk_init
from qdk._context import QSharpError

try:
    from qdk._interpreter import get_config as get_qdk_profile_config
except ImportError:
    from qsharp._qsharp import get_config as get_qdk_profile_config

from qdk_chemistry.utils import Logger

__all__ = ["QSHARP_UTILS"]

_QS_DIR = Path(__file__).parent
_target_profile = qdk.TargetProfile.Adaptive_RIF


def _init_qsharp():
    """Initialize the Q# interpreter with the chemistry project root."""
    qdk_config = get_qdk_profile_config()
    if qdk_config.get_target_profile() != "adaptive_rif":
        Logger.debug(
            f"QDK interpreter profile set to '{_target_profile}'. "
            "If you imported Q# code before this module was loaded, please re-import it, "
            "or set your target profile before importing qdk_chemistry."
        )
    qdk_init(project_root=_QS_DIR, target_profile=_target_profile)


class _QSharpUtilsProxy:
    """Lazy proxy that resolves Q# utilities on each access.

    If the Q# context is disposed and re-created (e.g. user calls ``qdk.init()``),
    this proxy will re-initialize the chemistry Q# project automatically.
    """

    def __getattr__(self, name: str):
        try:
            utils = qdk.code.QDKChemistry.Utils
            return getattr(utils, name)
        except (AttributeError, QSharpError):
            # Context was disposed or not yet initialized; re-initialize
            _init_qsharp()
            return getattr(qdk.code.QDKChemistry.Utils, name)


# Perform initial initialization
_init_qsharp()
QSHARP_UTILS = _QSharpUtilsProxy()
