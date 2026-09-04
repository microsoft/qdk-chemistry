"""Shared Azure credential helpers for QDK/Chemistry plugins."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

AUTH_MODES = ("azure-cli", "default")


def create_credential(auth_mode: str) -> Any:
    """Create an Azure credential for the requested authentication mode.

    Args:
        auth_mode: Either ``"azure-cli"`` (``AzureCliCredential``) or ``"default"`` (``DefaultAzureCredential``).

    Returns:
        Any: An ``azure.identity`` credential instance.

    Raises:
        ValueError: If *auth_mode* is not ``"azure-cli"`` or ``"default"``.

    """
    from azure.identity import AzureCliCredential, DefaultAzureCredential  # noqa: PLC0415

    if auth_mode == "azure-cli":
        return AzureCliCredential()
    if auth_mode == "default":
        return DefaultAzureCredential()
    raise ValueError(f"unsupported Azure credential mode {auth_mode!r}; expected one of: {', '.join(AUTH_MODES)}")
