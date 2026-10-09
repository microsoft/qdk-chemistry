"""Configuration for bounded property-based fuzz tests."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os

from hypothesis import settings

settings.register_profile(
    "qdk-fuzz",
    deadline=None,
    max_examples=int(os.environ.get("QDK_CHEMISTRY_FUZZ_EXAMPLES", "100")),
)
settings.load_profile("qdk-fuzz")
