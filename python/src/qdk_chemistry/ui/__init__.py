"""MCP Server and CLI for the QDK/Chemistry Toolkit."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib

__copyright__ = """"""

__version__ = "1.0.0"

__all__ = ["app", "cli"]


def __getattr__(name: str):
    """Lazy-load submodules to avoid import-time side effects."""
    if name == "cli":
        return importlib.import_module(".cli", __name__)
    if name == "app":
        return importlib.import_module(".tools", __name__).app
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
