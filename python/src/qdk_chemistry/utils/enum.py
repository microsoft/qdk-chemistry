"""Utility enums for QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # Backport for Python < 3.11
    class StrEnum(str, Enum):
        """Enum where members are also (and must be) strings.

        Backport for Python < 3.11.
        """

        def __new__(cls, value):
            if not isinstance(value, str):
                raise TypeError(f"{value!r} is not a string")
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj

        def __str__(self):
            return self.value

        @staticmethod
        def _generate_next_value_(name, _start, _count, _last_values):
            """Return the lower-cased version of the member name."""
            return name.lower()


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
