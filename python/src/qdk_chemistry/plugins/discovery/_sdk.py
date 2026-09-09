"""Microsoft Discovery SDK client helpers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Any
from urllib.parse import unquote, urlsplit

from azure.core.polling import NoPolling

DISCOVERY_SCOPE = "https://discovery.azure.com/.default"


def create_credential(credential_mode: str) -> Any:
    """Create the configured Azure credential."""
    from azure.identity import AzureCliCredential, DefaultAzureCredential  # noqa: PLC0415

    if credential_mode == "azure-cli":
        return AzureCliCredential()
    if credential_mode == "default":
        return DefaultAzureCredential()
    raise ValueError(f"unsupported Microsoft Discovery credential mode {credential_mode!r}")


def create_workspace_client(endpoint: str, credential: Any) -> Any:
    """Create a Microsoft Discovery workspace client."""
    from azure.ai.discovery import WorkspaceClient  # noqa: PLC0415

    return WorkspaceClient(
        endpoint=endpoint,
        credential=credential,
        credential_scopes=[DISCOVERY_SCOPE],
    )


class _OperationIdPolling(NoPolling):
    """Capture a submitted run ID without polling the operation."""

    operation_id: str | None = None

    def initialize(self, client: Any, initial_response: Any, deserialization_callback: Any) -> None:
        """Capture the operation ID from the required LRO response header."""
        super().initialize(client, initial_response, deserialization_callback)
        operation_location = initial_response.http_response.headers.get("Operation-Location")
        if not operation_location:
            raise RuntimeError("Microsoft Discovery run submission returned no Operation-Location header")
        operation_path = urlsplit(str(operation_location)).path.rstrip("/")
        self.operation_id = unquote(operation_path.rsplit("/", 1)[-1])
        if not self.operation_id:
            raise RuntimeError("Microsoft Discovery run submission returned an invalid Operation-Location header")


def response_mapping(value: Any) -> dict[str, Any]:
    """Convert an Azure SDK response to a plain mapping."""
    if isinstance(value, dict):
        return dict(value)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        mapped = as_dict()
        if isinstance(mapped, dict):
            return mapped
    return {}
