"""QDK/Chemistry Discovery remote backend."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdk_chemistry.plugins.discovery.backend import DiscoveryBackend

__all__ = ["DiscoveryBackend"]

_loaded = False


def __getattr__(name: str) -> Any:
    """Load the Discovery backend only when explicitly requested."""
    if name == "DiscoveryBackend":
        from qdk_chemistry.plugins.discovery.backend import DiscoveryBackend  # noqa: PLC0415

        return DiscoveryBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load() -> None:
    """Register the Discovery backend when its SDK is installed."""
    global _loaded  # noqa: PLW0603
    if _loaded:
        return

    importlib.import_module("azure.ai.discovery")

    from qdk_chemistry.plugins import PluginRegistrar  # noqa: PLC0415
    from qdk_chemistry.plugins.discovery.backend import DiscoveryBackend  # noqa: PLC0415

    PluginRegistrar().register_remote_backend("discovery", DiscoveryBackend)
    _loaded = True
