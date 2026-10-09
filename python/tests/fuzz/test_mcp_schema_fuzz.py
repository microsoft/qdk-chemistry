"""Property-based tests for MCP tool schema validation and dispatch."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import asyncio
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis_jsonschema import from_schema

pytest.importorskip("mcp", reason="MCP support is not installed")

from mcp.client import Client
from mcp.server.mcpserver import MCPServer

_SERVER = MCPServer("qdk-chemistry-fuzz-test")


@_SERVER.tool()
def echo_tool(
    name: str,
    count: int,
    enabled: bool = False,
    labels: list[str] | None = None,
    settings: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return schema-validated arguments without performing domain work."""
    return {
        "name": name,
        "count": count,
        "enabled": enabled,
        "labels": labels,
        "settings": settings,
    }


async def _tool_schema() -> dict[str, Any]:
    """Return the input schema published for the harmless echo tool."""
    tools = await _SERVER.list_tools()
    return next(tool.input_schema for tool in tools if tool.name == "echo_tool")


_SCHEMA = asyncio.run(_tool_schema())
_JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=20), children, max_size=5),
    max_leaves=20,
)


async def _call_echo(arguments: dict[str, Any]):
    """Call the echo tool through the public in-process MCP client."""
    async with Client(_SERVER) as client:
        return await client.call_tool("echo_tool", arguments)


@given(from_schema(_SCHEMA))
def test_schema_generated_tool_calls_round_trip(arguments: dict[str, Any]) -> None:
    """Every input generated from the published schema reaches the tool body."""
    result = asyncio.run(_call_echo(arguments))

    assert result.is_error is False
    assert result.structured_content == {
        "name": arguments["name"],
        "count": arguments["count"],
        "enabled": arguments.get("enabled", False),
        "labels": arguments.get("labels"),
        "settings": arguments.get("settings"),
    }


@given(st.dictionaries(st.text(max_size=20), _JSON_VALUES, max_size=10))
def test_arbitrary_tool_arguments_return_an_mcp_result(arguments: dict[str, Any]) -> None:
    """Arbitrary JSON objects are either schema-valid or reported as tool errors."""
    result = asyncio.run(_call_echo(arguments))

    assert isinstance(result.is_error, bool)
    assert result.content or result.structured_content is not None
