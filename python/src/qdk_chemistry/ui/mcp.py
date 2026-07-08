"""MCP server entry point for the QDK/Chemistry Toolkit.

Launch the MCP server with::

    qdk_chem_mcp                          # stdio (default)
    qdk_chem_mcp --transport sse          # Server-Sent Events
    qdk_chem_mcp --transport streamable-http --port 8081
    qdk_chem_mcp --transport streamable-http --stateless-http --json-response
        # sessionless JSON-RPC-over-HTTP (no Mcp-Session-Id required)
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
import logging
import os
import sys

from .tools import app

__all__ = ["main"]

_TRANSPORTS = ("stdio", "streamable-http", "sse")


def _install_output_schema_stripper() -> None:
    """Wrap the low-level ListToolsRequest handler to drop ``outputSchema`` and ``_meta``.

    Some MCP hosts pinned to the 2025-03-26 spec reject tools that include
    ``outputSchema`` (added in 2025-06-18). The ``_meta`` field (used here for
    MCP-Apps UI widget hints) is spec-compliant but unknown to such hosts and
    can also cause silent indexing failures. Enabling this stripper removes
    both fields so the ``tools/list`` payload is maximally portable.
    """
    from mcp.types import ListToolsRequest  # noqa: PLC0415

    server = app._mcp_server  # noqa: SLF001  # low-level mcp.server.lowlevel.server.Server
    inner = server.request_handlers.get(ListToolsRequest)
    if inner is None:  # pragma: no cover - defensive
        return

    async def _stripped(req):
        result = await inner(req)
        # result is a ServerResult wrapping a ListToolsResult
        tools_result = getattr(result, "root", result)
        for tool in getattr(tools_result, "tools", []):
            if hasattr(tool, "outputSchema"):
                tool.outputSchema = None
            # Pydantic model field is exposed as ``meta`` but serialises as ``_meta``.
            if hasattr(tool, "meta"):
                tool.meta = None
        return result

    server.request_handlers[ListToolsRequest] = _stripped


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse MCP server flags (--transport, --host, --port, --verbose)."""
    p = argparse.ArgumentParser(
        prog="qdk_chem_mcp",
        description="QDK/Chemistry MCP server",
    )
    p.add_argument(
        "--transport",
        choices=_TRANSPORTS,
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for HTTP transports (default: 127.0.0.1)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port for HTTP transports (default: 8081)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging",
    )
    p.add_argument(
        "--strip-output-schema",
        action="store_true",
        default=os.environ.get("QDK_CHEM_MCP_STRIP_OUTPUT_SCHEMA", "").lower() in ("1", "true", "yes"),
        help=(
            "Omit 'outputSchema' and '_meta' from tools/list responses for "
            "hosts pinned to MCP spec 2025-03-26 (also via "
            "QDK_CHEM_MCP_STRIP_OUTPUT_SCHEMA=1)"
        ),
    )
    p.add_argument(
        "--stateless-http",
        action="store_true",
        default=os.environ.get("QDK_CHEM_MCP_STATELESS_HTTP", "").lower() in ("1", "true", "yes"),
        help=(
            "Run streamable-http transport in sessionless mode: each request "
            "is independent and no Mcp-Session-Id header is required (also via "
            "QDK_CHEM_MCP_STATELESS_HTTP=1). Ignored for stdio/sse transports."
        ),
    )
    p.add_argument(
        "--json-response",
        action="store_true",
        default=os.environ.get("QDK_CHEM_MCP_JSON_RESPONSE", "").lower() in ("1", "true", "yes"),
        help=(
            "Return plain application/json responses instead of SSE-framed "
            "text/event-stream for streamable-http (also via "
            "QDK_CHEM_MCP_JSON_RESPONSE=1). Useful with --stateless-http for "
            "simple JSON-RPC-over-HTTP clients. Ignored for stdio/sse transports."
        ),
    )
    p.add_argument(
        "--relax-accept-header",
        action="store_true",
        default=os.environ.get("QDK_CHEM_MCP_RELAX_ACCEPT", "").lower() in ("1", "true", "yes"),
        help=(
            "Inject 'Accept: application/json, text/event-stream' on incoming "
            "streamable-http requests that don't already advertise both media "
            "types. Lets hosts that only send 'Accept: application/json' work "
            "with the MCP spec (also via QDK_CHEM_MCP_RELAX_ACCEPT=1). "
            "Ignored for stdio/sse transports."
        ),
    )
    return p.parse_args(argv)


def _accept_relaxer_middleware(inner_app):
    """Return an ASGI app that ensures Accept advertises both required types.

    The MCP streamable-http server returns 406 unless the request's Accept
    header lists both ``application/json`` and ``text/event-stream``. Some
    hosts only send ``application/json`` (or ``*/*``). This middleware
    rewrites the Accept header on HTTP requests so the inner MCP app
    accepts them.
    """
    required = b"application/json, text/event-stream"

    async def app_with_relaxed_accept(scope, receive, send):
        if scope.get("type") != "http":
            await inner_app(scope, receive, send)
            return
        headers = list(scope.get("headers", []))
        accept_val = b""
        accept_idx = -1
        for i, (k, v) in enumerate(headers):
            if k.lower() == b"accept":
                accept_val = v
                accept_idx = i
                break
        lower = accept_val.lower()
        has_json = b"application/json" in lower or b"*/*" in lower
        has_sse = b"text/event-stream" in lower or b"*/*" in lower
        if not (has_json and has_sse):
            if accept_idx >= 0:
                headers[accept_idx] = (b"accept", required)
            else:
                headers.append((b"accept", required))
            scope = dict(scope)
            scope["headers"] = headers
        await inner_app(scope, receive, send)

    return app_with_relaxed_accept


def main() -> None:
    """Run the QDK/Chemistry MCP server."""
    args = _parse_args(sys.argv[1:])

    # Suppress noisy INFO/DEBUG logs unless --verbose is given.
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)
    for name in ("qdk_chemistry", "qsharp", "pyscf", "mcp", "mcp.server"):
        logging.getLogger(name).setLevel(level)

    if args.strip_output_schema:
        _install_output_schema_stripper()
        logging.getLogger("qdk_chemistry").info(
            "tools/list: outputSchema and _meta will be stripped from responses",
        )

    if args.transport == "stdio":
        app.run(transport="stdio")
    else:
        # FastMCP.run() only accepts transport/mount_path; host, port, and
        # streamable-http options live on app.settings.
        app.settings.host = args.host
        app.settings.port = args.port
        if args.transport == "streamable-http":
            app.settings.stateless_http = args.stateless_http
            app.settings.json_response = args.json_response
        elif args.stateless_http or args.json_response:
            logging.getLogger("qdk_chemistry").warning(
                "--stateless-http / --json-response only apply to --transport streamable-http; ignoring for %s.",
                args.transport,
            )

        # --relax-accept-header requires wrapping the ASGI app, which means
        # bypassing FastMCP.run() and serving with uvicorn directly.
        if args.relax_accept_header and args.transport == "streamable-http":
            import uvicorn  # noqa: PLC0415

            inner = app.streamable_http_app()
            wrapped = _accept_relaxer_middleware(inner)
            logging.getLogger("qdk_chemistry").info(
                "streamable-http: Accept header will be relaxed to 'application/json, text/event-stream' when missing",
            )
            uvicorn.run(
                wrapped,
                host=args.host,
                port=args.port,
                log_level=app.settings.log_level.lower(),
            )
            return
        if args.relax_accept_header and args.transport == "sse":
            logging.getLogger("qdk_chemistry").warning(
                "--relax-accept-header only applies to --transport streamable-http; ignoring for sse.",
            )
        app.run(transport=args.transport)


if __name__ == "__main__":
    main()
