"""Plugin registration and discovery for QDK/Chemistry extensions.

Plugins connect QDK/Chemistry to optional chemistry packages, quantum software,
and execution services while preserving the library's common interfaces. An
extension can contribute:

* implementations of existing algorithm types or entirely new algorithm types;
* data classes used by algorithm inputs and outputs;
* remote execution backends; and
* cache backends.

External plugins
----------------
An installed package exposes a :class:`QdkChemistryPlugin` subclass through the
``qdk_chemistry.plugins`` entry-point group:

.. code-block:: toml

    [project.entry-points."qdk_chemistry.plugins"]
    custom = "custom_package.plugin:CustomPlugin"

Importing :mod:`qdk_chemistry` discovers each entry point and calls its
:meth:`QdkChemistryPlugin.register` method with a :class:`PluginRegistrar`. The
registrar provides access to the standard QDK/Chemistry registries without
requiring application code to import the plugin package directly.

.. code-block:: python

    from qdk_chemistry.plugins import PluginRegistrar, QdkChemistryPlugin

    class CustomPlugin(QdkChemistryPlugin):
         def register(self, registrar: PluginRegistrar) -> None:
              registrar.register_algorithm(lambda: CustomAlgorithm())
              registrar.register_remote_backend("custom", CustomRemoteBackend)

Bundled integrations
--------------------
QDK/Chemistry also includes integrations for:

* :mod:`qdk_chemistry.plugins.pyscf` for electronic-structure algorithms;
* :mod:`qdk_chemistry.plugins.qiskit` for circuit construction, mapping, and execution;
* :mod:`qdk_chemistry.plugins.openfermion` for operator conversion and qubit mapping;
* :mod:`qdk_chemistry.plugins.networkx` for graph-coloring term grouping; and
* :mod:`qdk_chemistry.plugins.geometric` for molecular geometry optimization.

Bundled integrations are loaded automatically when their optional dependencies
are available. Their implementations can then be created through the standard
algorithm registry, for example ``create("scf_solver", "pyscf")``.
"""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import warnings

from qdk_chemistry._core import DuplicateRegistrationError
from qdk_chemistry.plugins.base import ChemistryPlugin, PluginRegistrar, QdkChemistryPlugin

DuplicateRegistrationError.__module__ = __name__


def _load_plugins() -> None:
    """Load plugins advertised through the unified entry-point group."""
    from importlib.metadata import entry_points  # noqa: PLC0415

    for entry_point in entry_points(group="qdk_chemistry.plugins"):
        try:
            plugin_type = entry_point.load()
            if not isinstance(plugin_type, type) or not issubclass(plugin_type, QdkChemistryPlugin):
                raise TypeError("entry point must resolve to a QdkChemistryPlugin subclass")
            plugin = plugin_type()
            if plugin.api_version != QdkChemistryPlugin.api_version:
                raise ValueError(
                    f"unsupported plugin API version {plugin.api_version}; expected {QdkChemistryPlugin.api_version}"
                )
            plugin.register(PluginRegistrar())
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Failed to load QDK/Chemistry plugin {entry_point.name!r}: {exc}",
                UserWarning,
                stacklevel=2,
            )


__all__ = ["ChemistryPlugin", "DuplicateRegistrationError", "PluginRegistrar", "QdkChemistryPlugin"]
