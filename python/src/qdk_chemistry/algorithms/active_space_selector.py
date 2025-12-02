"""Public entry point for the Active Space Selector algorithm.

This module re-exports the core :class:`ActiveSpaceSelector` so that consumers can
import it directly from ``qdk_chemistry.algorithms`` without depending on
internal package paths.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry._core._algorithms import (
    ActiveSpaceSelector,  # noqa: F401 - re-export
    QdkAutocasActiveSpaceSelector,  # noqa: F401 - re-export
    QdkAutocasEosActiveSpaceSelector,  # noqa: F401 - re-export
    QdkOccupationActiveSpaceSelector,  # noqa: F401 - re-export
    QdkValenceActiveSpaceSelector,  # noqa: F401 - re-export
)
