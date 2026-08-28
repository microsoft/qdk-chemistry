"""MCP Server and CLI for the QDK/Chemistry Toolkit."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib

from qdk_chemistry import __version__

__copyright__ = """"""

__all__ = ["app", "cli"]

app: object
cli: object


def __getattr__(name: str):
    """Lazy-load submodules to avoid import-time side effects."""
    if name == "cli":
        module = importlib.import_module(".cli", __name__)
        globals()[name] = module
        return module
    if name == "app":
        application = importlib.import_module(".tools", __name__).app
        globals()[name] = application
        return application
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
