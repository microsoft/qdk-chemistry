"""Utility enums for QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from enum import StrEnum

__all__: list[str] = ["CaseInsensitiveStrEnum"]


class CaseInsensitiveStrEnum(StrEnum):
    """StrEnum that allows case-insensitive lookup of values."""

    @classmethod
    def _missing_(cls, value):  # make input case-insensitive
        if isinstance(value, str):
            for member in cls:
                if member.value.upper() == value.upper():
                    return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")
