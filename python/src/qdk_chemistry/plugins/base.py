"""Unified plugin registration API for QDK/Chemistry extensions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Type  # noqa: UP035

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from qdk_chemistry.algorithms.base import AlgorithmFactory


class PluginRegistrar:
    """Register plugin capabilities through the existing public registries."""

    def register_algorithm_factory(self, factory: AlgorithmFactory) -> None:
        """Register a factory that introduces an algorithm type.

        Args:
            factory: Factory that defines and creates the algorithm type.

        Raises:
            DuplicateRegistrationError: If the algorithm type is already registered.

        """
        from qdk_chemistry.algorithms.registry import register_factory  # noqa: PLC0415

        register_factory(factory)

    def register_algorithm(
        self,
        factory: Callable[[], Any],
        *,
        data_classes: Iterable[Type[Any]] = (),  # noqa: UP006
    ) -> None:
        """Register an algorithm factory and its DataClass input or output types.

        Args:
            factory: Factory that creates the algorithm implementation.
            data_classes: DataClass loaders used by the algorithm's inputs or outputs.

        Raises:
            DuplicateRegistrationError: If the algorithm name or a DataClass wire type is already registered.
            TypeError: If a DataClass loader is invalid.

        """
        from qdk_chemistry.algorithms import register  # noqa: PLC0415
        from qdk_chemistry.data.registry import _validate_dataclass_registrations  # noqa: PLC0415

        data_classes = _validate_dataclass_registrations(data_classes)
        register(factory)
        for dataclass_type in data_classes:
            self.register_dataclass(dataclass_type)

    def register_dataclass(self, dataclass_type: Type[Any]) -> None:  # noqa: UP006
        """Register a DataClass loader used in algorithm inputs or outputs.

        Args:
            dataclass_type: Canonical DataClass loader to register.

        Raises:
            DuplicateRegistrationError: If the DataClass wire type is already registered.

        """
        from qdk_chemistry.data import register_dataclass  # noqa: PLC0415

        register_dataclass(dataclass_type)

    def register_remote_backend(self, name: str, backend_type: Type[Any]) -> None:  # noqa: UP006
        """Register a remote execution backend.

        Backends may expose selected constructor parameters to untrusted MCP
        clients by declaring ``mcp_safe_config_options`` as a frozenset on the
        concrete class. The declaration is validated during registration; an
        omitted declaration exposes no parameters.

        Args:
            name: Registry name for the backend.
            backend_type: Backend class to register.

        Raises:
            DuplicateRegistrationError: If the backend name or class is already registered.

        """
        from qdk_chemistry.remote.backends import register_backend  # noqa: PLC0415

        register_backend(name)(backend_type)

    def register_cache_backend(self, name: str, backend_type: Type[Any]) -> None:  # noqa: UP006
        """Register a cache backend.

        Args:
            name: Registry name for the cache backend.
            backend_type: Cache backend class to register.

        Raises:
            DuplicateRegistrationError: If the cache name or class is already registered.

        """
        from qdk_chemistry.remote.cache import register_cache  # noqa: PLC0415

        register_cache(name)(backend_type)


class QdkChemistryPlugin(ABC):
    """Base class for plugins discovered through ``qdk_chemistry.plugins``."""

    api_version = 1

    @abstractmethod
    def register(self, registrar: PluginRegistrar) -> None:
        """Register the capabilities provided by this plugin."""


ChemistryPlugin = QdkChemistryPlugin


__all__ = ["ChemistryPlugin", "PluginRegistrar", "QdkChemistryPlugin"]
