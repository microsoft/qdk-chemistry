"""Module description."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from . import _core as _core


def main():
    print(f"qdk-chemistry { _core.version() } ({ _core.platform() })")
