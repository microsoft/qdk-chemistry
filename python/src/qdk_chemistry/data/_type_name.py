"""Wire-format identity helpers for QDK Chemistry data classes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdk_chemistry._core.data import DataClass as CoreDataClass

__all__: list[str] = []


def _validated_data_type_name(type_name: object, owner: str) -> str:
    """Validate a wire-format identifier.

    Args:
        type_name: Value returned by a data type naming method.
        owner: Qualified name used to identify the provider in error messages.

    Returns:
        The validated non-empty identifier.

    Raises:
        TypeError: If ``type_name`` is not a non-empty string.

    """
    if not isinstance(type_name, str) or not type_name:
        raise TypeError(f"{owner} must provide a non-empty data_type_name()")
    if any(token in type_name for token in ("/", "\\", "..", "*", "?", "[", "]")):
        raise TypeError(f"{owner} returned an unsafe data_type_name(): {type_name!r}")
    return type_name


def class_data_type_name(dataclass_type: type[CoreDataClass]) -> str:
    """Return a loader's static wire-format identifier.

    Args:
        dataclass_type: Data class loader that provides ``data_type_name()``.

    Returns:
        The loader's validated wire-format identifier.

    Raises:
        TypeError: If the loader does not provide ``data_type_name()`` or the
            method does not return a non-empty string.

    """
    method = getattr(dataclass_type, "data_type_name", None)
    if not callable(method):
        raise TypeError(f"{dataclass_type.__module__}.{dataclass_type.__qualname__} must provide data_type_name()")
    return _validated_data_type_name(method(), f"{dataclass_type.__module__}.{dataclass_type.__qualname__}")


def declares_data_type_name(dataclass_type: type[CoreDataClass]) -> bool:
    """Return whether a data class declares its own static identifier.

    Inherited identifiers describe members of a serialization family. Only a
    class that declares the method is a canonical loader for registry
    discovery.

    Args:
        dataclass_type: Data class to inspect.

    Returns:
        ``True`` when ``data_type_name`` is present in the class's own
        namespace; otherwise, ``False``.

    """
    return "data_type_name" in vars(dataclass_type)


def instance_data_type_name(value: CoreDataClass) -> str:
    """Return a data-class instance's wire-format identifier.

    Args:
        value: Data class instance whose identifier is requested.

    Returns:
        The instance's validated wire-format identifier.

    Raises:
        TypeError: If either naming method does not return a non-empty string,
            or if the class and instance identifiers disagree.

    """
    instance_name = _validated_data_type_name(
        value.get_data_type_name(),
        f"{type(value).__module__}.{type(value).__qualname__}",
    )
    class_name = class_data_type_name(type(value))
    if instance_name != class_name:
        raise TypeError(
            f"{type(value).__module__}.{type(value).__qualname__} returned inconsistent wire-format identifiers: "
            f"data_type_name()={class_name!r}, get_data_type_name()={instance_name!r}"
        )
    return instance_name
