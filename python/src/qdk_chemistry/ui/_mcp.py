"""Optional MCP server integration for the QDK Chemistry UI."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
from typing import Any

from qdk_chemistry import __version__

MCP_INSTALL_MESSAGE = "QDK Chemistry MCP support is not installed. Install 'qdk-chemistry[mcp]' to use qcmcp."
MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None


class _InactiveMCPServer:
    """Preserve shared CLI functions without registering MCP tools."""

    def tool(self, *args: Any, **kwargs: Any):
        """Return an identity decorator matching ``MCPServer.tool``."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(function):
            return function

        return decorator


if MCP_AVAILABLE:
    from mcp.server.mcpserver import Context as MCPContext
    from mcp.server.mcpserver import MCPServer

    app: Any = MCPServer(
        "qdk-chemistry",
        version=__version__,
        dependencies=["qdk_chemistry"],
        instructions=(
            "Call bind_workspace before every other QDK Chemistry tool when it is available. "
            "Plugin-launched servers require one immutable absolute workspace binding. "
            "Tool descriptions are compact call contracts; before chaining tools or choosing methods and settings, "
            "load the qdk-chemistry-mcp skill when it is available."
        ),
    )
else:
    MCPContext = Any
    app = _InactiveMCPServer()


def require_mcp() -> None:
    """Raise an actionable error when MCP support is unavailable."""
    if not MCP_AVAILABLE:
        raise ModuleNotFoundError(MCP_INSTALL_MESSAGE)
