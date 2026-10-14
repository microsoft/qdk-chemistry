"""Malformed-input recovery tests for the MCP stdio transport."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import os
import subprocess
import sys

import pytest

from qdk_chemistry.ui._mcp import MCP_AVAILABLE

pytestmark = pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP support is not installed")


def test_stdio_recovers_from_malformed_messages() -> None:
    """Malformed records do not prevent a subsequent valid MCP request."""
    malformed_corpus = [
        b"not-json\n",
        b"\xff\xfe\n",
        b'{"jsonrpc":"2.0"\n',
        b"{}\n",
        b"[]\n",
        b"null\n",
        b'"string"\n',
        (b"[" * 80) + (b"]" * 80) + b"\n",
        b'{"jsonrpc":"2.0","id":99,"method":"unknown/method"}\n',
    ]
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "qdk-chemistry-fuzz-test", "version": "1"},
        },
    }
    process = subprocess.run(
        [sys.executable, "-m", "qdk_chemistry.ui.mcp"],
        input=b"".join(malformed_corpus) + json.dumps(initialize).encode() + b"\n",
        capture_output=True,
        timeout=10,
        check=False,
        env=os.environ.copy(),
    )
    assert process.returncode == 0, process.stderr.decode(errors="replace")
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert any(response.get("id") == 99 and "error" in response for response in responses)
    assert any(response.get("id") == 1 and "result" in response for response in responses)
