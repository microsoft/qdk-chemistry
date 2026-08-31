"""Bind a plugin MCP process to one explicit workspace."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from mcp.server.mcpserver import Context  # noqa: TC002
from mcp.shared.exceptions import MCPError, NoBackChannelError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp.server.context import HandlerResult, ServerRequestContext

_WORKSPACE_LOCK = threading.Lock()
_WORKSPACE_ROOT: Path | None = None
_REQUIRE_BINDING_ENV = "QDK_REQUIRE_WORKSPACE_BINDING"


def current_workspace_root() -> Path | None:
    """Return the immutable workspace binding for this MCP process."""
    return _WORKSPACE_ROOT


def _file_uri_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    authority = f"//{parsed.netloc}" if parsed.netloc and parsed.netloc != "localhost" else ""
    return Path(url2pathname(authority + unquote(parsed.path))).resolve()


async def _workspace_from_client(ctx: Context) -> tuple[Path | None, str | None]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = await ctx.session.list_roots()
    except (MCPError, NoBackChannelError) as exc:
        return None, f"the MCP client did not provide a workspace root: {exc}"
    roots = {_file_uri_path(str(root.uri)) for root in result.roots}
    roots.discard(None)
    if len(roots) != 1:
        return None, "the MCP client must provide exactly one file workspace root, or workspace_root must be explicit"
    return roots.pop(), None


def configure_workspace(path: Path) -> dict[str, object]:
    """Bind this process to one workspace for relative file resolution.

    Args:
        path: Absolute path to the workspace directory.

    Returns:
        A mapping that confirms the binding and gives the resolved workspace
        root.

    Raises:
        ValueError: If ``path`` is relative or does not identify a directory.
        RuntimeError: If this process is already bound to a different workspace.

    """
    if not path.is_absolute():
        raise ValueError("workspace_root must be an absolute path")
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"workspace_root {str(resolved)!r} is not a directory")

    global _WORKSPACE_ROOT  # noqa: PLW0603
    with _WORKSPACE_LOCK:
        if _WORKSPACE_ROOT is not None and resolved != _WORKSPACE_ROOT:
            raise RuntimeError(
                f"this QDK Chemistry MCP process is already bound to workspace {str(_WORKSPACE_ROOT)!r}; "
                "start a separate MCP process for another workspace"
            )
        os.environ["QDK_WORKSPACE_ROOT"] = str(resolved)
        os.chdir(resolved)
        _WORKSPACE_ROOT = resolved

    return {"bound": True, "workspace_root": str(resolved)}


async def bind_workspace(ctx: Context, workspace_root: str | None = None) -> dict[str, object]:
    """Bind this MCP process to the active workspace before other QDK tools.

    Args:
        ctx: MCP request context used to discover the client workspace when
            ``workspace_root`` is omitted.
        workspace_root: Absolute workspace directory. When omitted, the client
            must provide exactly one file URI workspace root.

    Returns:
        A mapping that reports whether binding succeeded. Successful results
        include the resolved workspace root and binding source; failed results
        include an error message.

    """
    source = "argument"
    if workspace_root is None:
        discovered, error = await _workspace_from_client(ctx)
        if discovered is None:
            return {"bound": False, "error": error or "workspace root is unavailable"}
        path = discovered
        source = "mcp_roots"
    else:
        path = Path(workspace_root)
    try:
        return {**configure_workspace(path), "source": source}
    except (OSError, RuntimeError, ValueError) as exc:
        return {"bound": False, "error": str(exc)}


async def workspace_binding_middleware(
    context: ServerRequestContext[Any, Any],
    call_next: Callable[[ServerRequestContext[Any, Any]], Awaitable[HandlerResult]],
) -> HandlerResult:
    """Require a workspace binding before plugin tools resolve relative paths.

    Args:
        context: Incoming MCP request context.
        call_next: Handler that receives the request after binding succeeds.

    Returns:
        The downstream handler result.

    Raises:
        RuntimeError: If binding is required but neither an existing binding nor
            a single client workspace root is available.

    """
    if os.environ.get(_REQUIRE_BINDING_ENV, "").lower() not in {"1", "true", "yes"}:
        return await call_next(context)
    tool_name = (context.params or {}).get("name") if context.method == "tools/call" else None
    if tool_name is None or tool_name == "bind_workspace" or _WORKSPACE_ROOT is not None:
        return await call_next(context)
    discovered, error = await _workspace_from_client(context)  # type: ignore[arg-type]
    if discovered is not None:
        configure_workspace(discovered)
        return await call_next(context)
    raise RuntimeError(
        "QDK Chemistry workspace is not bound. Call bind_workspace with an absolute workspace_root first. "
        f"{error or ''}".strip()
    )
