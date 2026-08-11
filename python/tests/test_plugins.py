"""Tests for unified QDK/Chemistry plugin discovery and registration."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from unittest.mock import MagicMock, call

import pytest

import qdk_chemistry
from qdk_chemistry import plugins
from qdk_chemistry._core import DuplicateRegistrationError as CoreDuplicateRegistrationError
from qdk_chemistry.plugins import ChemistryPlugin, DuplicateRegistrationError, PluginRegistrar, QdkChemistryPlugin

_BUNDLED_PLUGIN_AUTOLOAD_CASES = (
    ("pyscf", "QDK_CHEMISTRY_DISABLE_PYSCF_AUTOLOAD"),
    ("qiskit", "QDK_CHEMISTRY_DISABLE_QISKIT_AUTOLOAD"),
    ("openfermion", "QDK_CHEMISTRY_DISABLE_OPENFERMION_AUTOLOAD"),
    ("networkx", "QDK_CHEMISTRY_DISABLE_NETWORKX_AUTOLOAD"),
    ("geometric", "QDK_CHEMISTRY_DISABLE_GEOMETRIC_AUTOLOAD"),
)


class _EntryPoint:
    """Minimal importlib entry-point stub."""

    def __init__(self, name, target):
        """Store entry-point metadata and its load target."""
        self.name = name
        self._target = target

    def load(self):
        """Return the target or reproduce its import failure."""
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def test_duplicate_registration_error_is_public_value_error():
    """The plugin API exposes the native collision error as a ValueError subtype."""
    assert DuplicateRegistrationError is CoreDuplicateRegistrationError
    assert issubclass(DuplicateRegistrationError, ValueError)


def test_plugin_registrar_delegates_to_existing_registries(monkeypatch):
    """The facade preserves algorithm and DataClass registries as implementation APIs."""
    algorithm_factory = MagicMock()
    algorithm_type_factory = MagicMock()
    dataclass_type = type("PluginData", (), {})
    register_algorithm = MagicMock()
    register_algorithm_factory = MagicMock()
    register_dataclass = MagicMock()
    monkeypatch.setattr("qdk_chemistry.algorithms.register", register_algorithm)
    monkeypatch.setattr("qdk_chemistry.algorithms.registry.register_factory", register_algorithm_factory)
    monkeypatch.setattr("qdk_chemistry.data.register_dataclass", register_dataclass)

    registrar = PluginRegistrar()
    registrar.register_algorithm_factory(algorithm_type_factory)
    registrar.register_algorithm(algorithm_factory, data_classes=(dataclass_type,))
    registrar.register_dataclass(dataclass_type)

    register_algorithm_factory.assert_called_once_with(algorithm_type_factory)
    register_algorithm.assert_called_once_with(algorithm_factory)
    assert register_dataclass.call_args_list == [call(dataclass_type), call(dataclass_type)]


def test_legacy_plugin_base_is_alias():
    """The original plugin base name remains compatible."""
    assert ChemistryPlugin is QdkChemistryPlugin


def test_unified_plugin_entry_points_load_independently(monkeypatch):
    """A broken unified plugin does not prevent later plugins from loading."""
    loaded = []

    class HealthyPlugin(QdkChemistryPlugin):
        """Plugin used to record successful registration."""

        def register(self, registrar):
            """Record the registrar supplied by the loader."""
            loaded.append(registrar)

    entry_points = [
        _EntryPoint("broken", RuntimeError("cannot import")),
        _EntryPoint("healthy", HealthyPlugin),
    ]

    def find_entry_points(*, group):
        assert group == "qdk_chemistry.plugins"
        return entry_points

    monkeypatch.setattr("importlib.metadata.entry_points", find_entry_points)

    with pytest.warns(UserWarning, match="plugin 'broken'.*cannot import"):
        plugins._load_plugins()

    assert len(loaded) == 1
    assert isinstance(loaded[0], PluginRegistrar)


def test_unified_plugins_load_before_bundled_integrations(monkeypatch):
    """Unified entry points register before bundled integrations."""
    calls = []
    monkeypatch.setattr(plugins, "_load_plugins", lambda: calls.append("unified"))
    monkeypatch.setattr(qdk_chemistry, "_load_bundled_plugin", lambda *args: calls.append(args[0]))

    qdk_chemistry._import_plugins()

    assert calls == ["unified", *(plugin_name for plugin_name, _ in _BUNDLED_PLUGIN_AUTOLOAD_CASES)]


@pytest.mark.parametrize(("plugin_name", "disable_env_var"), _BUNDLED_PLUGIN_AUTOLOAD_CASES)
def test_bundled_plugin_autoload_can_be_disabled(monkeypatch, plugin_name, disable_env_var):
    """Each bundled plugin can be excluded before its module is imported."""
    import_module = MagicMock()
    monkeypatch.setenv(disable_env_var, "TrUe")
    monkeypatch.setattr(qdk_chemistry.importlib, "import_module", import_module)

    qdk_chemistry._load_bundled_plugin(plugin_name, disable_env_var)

    import_module.assert_not_called()


def test_bundled_plugin_duplicate_warns_with_disable_hint_and_propagates(monkeypatch):
    """A bundled plugin collision identifies its opt-out and propagates."""
    plugin = MagicMock()
    plugin.load.side_effect = DuplicateRegistrationError("name is already registered")
    monkeypatch.delenv("QDK_CHEMISTRY_DISABLE_QISKIT_AUTOLOAD", raising=False)
    monkeypatch.setattr(qdk_chemistry.importlib, "import_module", MagicMock(return_value=plugin))

    with (
        pytest.warns(
            UserWarning,
            match=r"duplicate registration.*QDK_CHEMISTRY_DISABLE_QISKIT_AUTOLOAD=1",
        ),
        pytest.raises(DuplicateRegistrationError, match="name is already registered"),
    ):
        qdk_chemistry._load_bundled_plugin("qiskit", "QDK_CHEMISTRY_DISABLE_QISKIT_AUTOLOAD")


def test_bundled_plugin_unexpected_error_propagates(monkeypatch):
    """Automatic loading does not hide errors unrelated to registration collisions."""
    plugin = MagicMock()
    plugin.load.side_effect = RuntimeError("unexpected failure")
    monkeypatch.setattr(qdk_chemistry.importlib, "import_module", MagicMock(return_value=plugin))

    with pytest.raises(RuntimeError, match="unexpected failure"):
        qdk_chemistry._load_bundled_plugin("qiskit", "QDK_CHEMISTRY_DISABLE_QISKIT_AUTOLOAD")


def test_plugin_entry_point_must_resolve_to_plugin_class(monkeypatch):
    """Entry points reject values that do not implement the plugin contract."""

    def find_entry_points(*, group):
        assert group == "qdk_chemistry.plugins"
        return [_EntryPoint("invalid", object())]

    monkeypatch.setattr("importlib.metadata.entry_points", find_entry_points)

    with pytest.warns(UserWarning, match="QdkChemistryPlugin subclass"):
        plugins._load_plugins()
