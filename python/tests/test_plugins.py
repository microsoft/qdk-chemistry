"""Tests for unified QDK/Chemistry plugin discovery and registration."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

import qdk_chemistry
from qdk_chemistry import plugins
from qdk_chemistry._core import DuplicateRegistrationError as CoreDuplicateRegistrationError
from qdk_chemistry.algorithms import ScfSolver, registry
from qdk_chemistry.data import DataClass, Structure, register_dataclass
from qdk_chemistry.data import registry as dataclass_registry
from qdk_chemistry.plugins import ChemistryPlugin, DuplicateRegistrationError, PluginRegistrar, QdkChemistryPlugin
from qdk_chemistry.remote.backends import RemoteBackend, get_backend
from qdk_chemistry.remote.backends import base as remote_backend_registry
from qdk_chemistry.remote.cache import FolderCache

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


def _write_test_plugin(project_dir: Path) -> None:
    """Create a complete installable plugin package for entry-point testing."""
    package_dir = project_dir / "src" / "qdk_chemistry_entry_point_test"
    package_dir.mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text(
        """\
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "qdk-chemistry-entry-point-test"
version = "0.0.0"
dependencies = ["qdk-chemistry"]

[project.entry-points."qdk_chemistry.plugins"]
pytest = "qdk_chemistry_entry_point_test.plugin:PipEntryPointPlugin"

[tool.setuptools.packages.find]
where = ["src"]
""",
        encoding="utf-8",
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        """\
from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.plugins import PluginRegistrar, QdkChemistryPlugin


class PipEntryPointAlgorithm(Algorithm):
    def type_name(self):
        return "pytest_entry_point"

    def name(self):
        return "pip"

    def _run_impl(self):
        return "loaded through pip entry point"


class PipEntryPointAlgorithmFactory(AlgorithmFactory):
    def algorithm_type_name(self):
        return "pytest_entry_point"

    def default_algorithm_name(self):
        return "pip"


class PipEntryPointPlugin(QdkChemistryPlugin):
    def register(self, registrar: PluginRegistrar):
        registrar.register_algorithm_factory(PipEntryPointAlgorithmFactory())
        registrar.register_algorithm(PipEntryPointAlgorithm)
""",
        encoding="utf-8",
    )


def test_pip_installed_plugin_entry_point_loads_complete_module(tmp_path):
    """An independently installed plugin is discovered and usable on import."""
    project_dir = tmp_path / "plugin"
    target_dir = tmp_path / "site-packages"
    _write_test_plugin(project_dir)

    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--no-index",
            "--target",
            str(target_dir),
            str(project_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    package_root = Path(qdk_chemistry.__file__).resolve().parent.parent
    env = os.environ.copy()
    python_paths = [str(target_dir), str(package_root)]
    if existing_python_path := env.get("PYTHONPATH"):
        python_paths.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    for _, disable_env_var in _BUNDLED_PLUGIN_AUTOLOAD_CASES:
        env[disable_env_var] = "1"

    verify = subprocess.run(
        [
            sys.executable,
            "-c",
            """\
import importlib.metadata
from pathlib import Path
import sys

target_dir = Path(sys.argv[1]).resolve()
matching_entry_points = [
    entry_point
    for entry_point in importlib.metadata.entry_points(group="qdk_chemistry.plugins")
    if entry_point.name == "pytest"
]
assert len(matching_entry_points) == 1
assert matching_entry_points[0].value == (
    "qdk_chemistry_entry_point_test.plugin:PipEntryPointPlugin"
)

from qdk_chemistry.algorithms import available, create
import qdk_chemistry_entry_point_test

assert Path(qdk_chemistry_entry_point_test.__file__).resolve().is_relative_to(target_dir)
assert "pip" in available("pytest_entry_point")
algorithm = create("pytest_entry_point", "pip")
assert algorithm.__class__.__module__ == "qdk_chemistry_entry_point_test.plugin"
assert algorithm.run() == "loaded through pip entry point"
""",
            str(target_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_duplicate_registration_error_is_public_value_error():
    """The plugin API exposes the native collision error as a ValueError subtype."""
    assert DuplicateRegistrationError is CoreDuplicateRegistrationError
    assert issubclass(DuplicateRegistrationError, ValueError)


def test_plugin_registrar_delegates_to_existing_registries(monkeypatch):
    """The facade preserves the existing registries as implementation APIs."""
    algorithm_factory = MagicMock()
    algorithm_type_factory = MagicMock()
    dataclass_type = type("PluginData", (), {})
    remote_backend_type = type("PluginRemote", (), {})
    cache_backend_type = type("PluginCache", (), {})
    register_algorithm = MagicMock()
    register_algorithm_factory = MagicMock()
    register_dataclass = MagicMock()
    validate_dataclasses = MagicMock(side_effect=tuple)
    remote_decorator = MagicMock()
    register_remote_backend = MagicMock(return_value=remote_decorator)
    cache_decorator = MagicMock()
    register_cache_backend = MagicMock(return_value=cache_decorator)
    monkeypatch.setattr("qdk_chemistry.algorithms.register", register_algorithm)
    monkeypatch.setattr("qdk_chemistry.algorithms.registry.register_factory", register_algorithm_factory)
    monkeypatch.setattr("qdk_chemistry.data.register_dataclass", register_dataclass)
    monkeypatch.setattr("qdk_chemistry.data.registry._validate_dataclass_registrations", validate_dataclasses)
    monkeypatch.setattr("qdk_chemistry.remote.backends.register_backend", register_remote_backend)
    monkeypatch.setattr("qdk_chemistry.remote.cache.register_cache", register_cache_backend)

    registrar = PluginRegistrar()
    registrar.register_algorithm_factory(algorithm_type_factory)
    registrar.register_algorithm(algorithm_factory, data_classes=(dataclass_type,))
    registrar.register_dataclass(dataclass_type)
    registrar.register_remote_backend("plugin-remote", remote_backend_type)
    registrar.register_cache_backend("plugin-cache", cache_backend_type)

    register_algorithm_factory.assert_called_once_with(algorithm_type_factory)
    validate_dataclasses.assert_called_once_with((dataclass_type,))
    register_algorithm.assert_called_once_with(algorithm_factory)
    assert register_dataclass.call_args_list == [call(dataclass_type), call(dataclass_type)]
    register_remote_backend.assert_called_once_with("plugin-remote")
    remote_decorator.assert_called_once_with(remote_backend_type)
    register_cache_backend.assert_called_once_with("plugin-cache")
    cache_decorator.assert_called_once_with(cache_backend_type)


def test_plugin_registrar_validates_dataclasses_before_registering_algorithm(monkeypatch):
    """A DataClass collision cannot leave the plugin algorithm registered."""
    monkeypatch.setattr(dataclass_registry, "_DATACLASS_REGISTRY", {})
    monkeypatch.setattr(dataclass_registry, "_DISCOVERY_COMPLETE", True)

    class PluginAlgorithm(ScfSolver):
        """Algorithm that must not survive the failed plugin registration."""

        def name(self):
            """Return the unique test registry name."""
            return "failed_plugin_algorithm"

        def _run_impl(self, structure, charge, spin_multiplicity):
            """Provide the minimal implementation required by the base class."""

    class FirstPluginData(DataClass):
        """Existing owner of the shared test wire type."""

        @staticmethod
        def data_type_name():
            """Return the shared test wire type."""
            return "test.plugin_transaction"

    class ConflictingPluginData(DataClass):
        """Plugin DataClass colliding with an existing owner."""

        @staticmethod
        def data_type_name():
            """Return the shared test wire type."""
            return "test.plugin_transaction"

    register_dataclass(FirstPluginData)

    try:
        with pytest.raises(DuplicateRegistrationError, match="already registered"):
            PluginRegistrar().register_algorithm(PluginAlgorithm, data_classes=(ConflictingPluginData,))

        assert "failed_plugin_algorithm" not in registry.available("scf_solver")
        assert dataclass_registry.get_dataclass_type("test.plugin_transaction") is FirstPluginData
    finally:
        registry.unregister("scf_solver", "failed_plugin_algorithm")


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


def test_failed_plugin_registration_does_not_latch_filtered_dataclass_discovery(monkeypatch, tmp_path):
    """A plugin failure after validation does not prevent cache loader discovery."""

    class FailingPlugin(QdkChemistryPlugin):
        """Plugin whose algorithm construction fails after DataClass validation."""

        def register(self, registrar):
            """Validate Structure before reproducing an algorithm registration failure."""

            def fail_algorithm():
                raise RuntimeError("algorithm registration failed")

            registrar.register_algorithm(fail_algorithm, data_classes=[Structure])

    def find_entry_points(*, group):
        assert group == "qdk_chemistry.plugins"
        return [_EntryPoint("failed", FailingPlugin)]

    monkeypatch.setattr(dataclass_registry, "_DATACLASS_REGISTRY", {})
    monkeypatch.setattr(dataclass_registry, "_DISCOVERY_COMPLETE", False)
    monkeypatch.setattr("importlib.metadata.entry_points", find_entry_points)

    with pytest.warns(UserWarning, match="plugin 'failed'.*algorithm registration failed"):
        plugins._load_plugins()

    cache = FolderCache(tmp_path)
    structure = Structure([[0.0, 0.0, 0.0]], [1])
    cache.put_data("c7fa6171bf37a03c", structure)

    loaded = cache.get_data("c7fa6171bf37a03c")
    assert isinstance(loaded, Structure)
    assert loaded.get_num_atoms() == 1


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


def test_unified_plugin_registers_remote_backend(monkeypatch):
    """Unified discovery registers a remote backend without importing it explicitly."""

    class PluginRemoteBackend(RemoteBackend):
        """Remote backend supplied by the test plugin."""

        def connect(self):
            """Connect to the test backend."""

        def disconnect(self):
            """Disconnect from the test backend."""

        def upload(self, local_path, remote_path):
            """Accept an upload for the test backend."""

        def download(self, remote_path, local_path):
            """Accept a download for the test backend."""

    class RemotePlugin(QdkChemistryPlugin):
        """Plugin that contributes a remote backend."""

        def register(self, registrar: PluginRegistrar) -> None:
            """Register the remote backend."""
            registrar.register_remote_backend("plugin-remote", PluginRemoteBackend)

    def find_entry_points(*, group):
        """Return the remote plugin through the unified group."""
        assert group == "qdk_chemistry.plugins"
        return [_EntryPoint("remote", RemotePlugin)]

    monkeypatch.setattr("importlib.metadata.entry_points", find_entry_points)
    monkeypatch.setattr(remote_backend_registry, "_BACKENDS", {})

    plugins._load_plugins()

    backend = get_backend("plugin-remote", endpoint="compute.example.com")
    assert isinstance(backend, PluginRemoteBackend)
    assert backend._backend_args["endpoint"] == "compute.example.com"
