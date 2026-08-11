"""Registry for serializable QDK/Chemistry data classes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from threading import RLock

from qdk_chemistry._core import DuplicateRegistrationError as _DuplicateRegistrationError
from qdk_chemistry._core.data import DataClass as _CoreDataClass
from qdk_chemistry.data._type_name import class_data_type_name, declares_data_type_name
from qdk_chemistry.data.base import DataClass as _PythonDataClass

_DATACLASS_REGISTRY: dict[str, type[_CoreDataClass]] = {}
_REGISTRY_LOCK = RLock()
_DISCOVERY_COMPLETE = False


def _declared_type_name(dataclass_type: type[_CoreDataClass]) -> str:
    """Return the wire type name declared directly by a DataClass loader."""
    if not declares_data_type_name(dataclass_type):
        raise TypeError(
            f"{dataclass_type.__module__}.{dataclass_type.__qualname__} is a DataClass subclass but does not declare "
            "a static data_type_name() method; add this method to register the DataClass in a plugin"
        )

    return class_data_type_name(dataclass_type)


def register_dataclass(dataclass_type: type[_CoreDataClass]) -> type[_CoreDataClass]:
    """Register a DataClass subclass for file deserialization.

    The loader must declare its own static ``data_type_name()`` method rather
    than inheriting another loader's wire-format identifier.

    Args:
        dataclass_type: DataClass subclass to register.

    Returns:
        The registered class, allowing this function to be used as a decorator.

    Raises:
        TypeError: If the value is not a DataClass subclass or has no declared type name.
        DuplicateRegistrationError: If another class already owns the declared type name.

    """
    if not isinstance(dataclass_type, type) or not issubclass(dataclass_type, _CoreDataClass):
        raise TypeError("registered data classes must derive from qdk_chemistry.data.DataClass")

    _discover_imported_dataclasses(excluded_type=dataclass_type)
    return _register_dataclass(dataclass_type)


def _register_dataclass(dataclass_type: type[_CoreDataClass]) -> type[_CoreDataClass]:
    """Register a validated DataClass without triggering discovery."""
    type_name = _declared_type_name(dataclass_type)
    with _REGISTRY_LOCK:
        registered_type = _DATACLASS_REGISTRY.get(type_name)
        if registered_type is not None and registered_type is not dataclass_type:
            raise _DuplicateRegistrationError(
                f"DataClass type name {type_name!r} is already registered by "
                f"{registered_type.__module__}.{registered_type.__qualname__}"
            )
        _DATACLASS_REGISTRY[type_name] = dataclass_type
    return dataclass_type


def _discover_imported_dataclasses(*, excluded_type: type[_CoreDataClass] | None = None) -> None:
    """Register canonical DataClass types that are already imported."""
    global _DISCOVERY_COMPLETE  # noqa: PLW0603

    with _REGISTRY_LOCK:
        if _DISCOVERY_COMPLETE:
            return

        import qdk_chemistry.data  # noqa: PLC0415
        import qdk_chemistry.data.symmetry  # noqa: F401, PLC0415

        stack = list(_PythonDataClass.__subclasses__()) + list(_CoreDataClass.__subclasses__())
        seen: set[int] = set()
        while stack:
            dataclass_type = stack.pop()
            if id(dataclass_type) in seen:
                continue
            seen.add(id(dataclass_type))
            if (
                dataclass_type is not _PythonDataClass
                and dataclass_type is not excluded_type
                and dataclass_type.__module__.startswith("qdk_chemistry.")
                and declares_data_type_name(dataclass_type)
            ):
                _register_dataclass(dataclass_type)
            stack.extend(dataclass_type.__subclasses__())
        _DISCOVERY_COMPLETE = True


def get_dataclass_type(type_name: str) -> type[_CoreDataClass] | None:
    """Return the registered DataClass for a wire type name, if available."""
    with _REGISTRY_LOCK:
        dataclass_type = _DATACLASS_REGISTRY.get(type_name)
    if dataclass_type is not None:
        return dataclass_type

    _discover_imported_dataclasses()
    with _REGISTRY_LOCK:
        return _DATACLASS_REGISTRY.get(type_name)


def available_dataclasses() -> dict[str, type[_CoreDataClass]]:
    """Return all currently registered and discoverable DataClass types."""
    _discover_imported_dataclasses()
    with _REGISTRY_LOCK:
        return dict(_DATACLASS_REGISTRY)


__all__ = ["available_dataclasses", "get_dataclass_type", "register_dataclass"]
