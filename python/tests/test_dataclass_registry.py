"""Tests for serializable DataClass discovery and registration."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

import qdk_chemistry.data as data_module
import qdk_chemistry.data.symmetry as symmetry_module
from qdk_chemistry._core.data import DataClass as CoreDataClass
from qdk_chemistry.data import DataClass, QpeResult, Settings, get_dataclass_type, register_dataclass
from qdk_chemistry.data import registry as dataclass_registry
from qdk_chemistry.data._type_name import class_data_type_name, instance_data_type_name
from qdk_chemistry.plugins import DuplicateRegistrationError
from qdk_chemistry.remote.serialization import deserialize_outputs, serialize_outputs


@pytest.mark.parametrize("public_module", [data_module, symmetry_module])
def test_all_public_dataclasses_have_canonical_loader(public_module):
    """Every public C++ and Python DataClass has a serialization loader."""
    registered_loaders = dataclass_registry.available_dataclasses()
    for public_name in public_module.__all__:
        dataclass_type = getattr(public_module, public_name)
        if (
            not isinstance(dataclass_type, type)
            or dataclass_type in (CoreDataClass, DataClass)
            or not issubclass(dataclass_type, CoreDataClass)
        ):
            continue

        type_name = class_data_type_name(dataclass_type)
        loader = registered_loaders.get(type_name)
        qualified_name = f"{public_module.__name__}.{public_name}"
        assert loader is not None, qualified_name
        assert get_dataclass_type(type_name) is loader, qualified_name


def test_static_and_instance_names_agree():
    """DataClass instances use the identifier declared by their loader."""
    settings = Settings()
    assert instance_data_type_name(settings) == Settings.data_type_name()


def test_qpe_result_output_round_trip(tmp_path):
    """An algorithm result absent from the former allowlist round-trips."""
    result = QpeResult.from_phase_fraction(
        method="test",
        phase_fraction=0.25,
        eigenvalue_from_phase=lambda phase: -phase,
    )

    serialize_outputs(tmp_path, result)
    restored = deserialize_outputs(tmp_path)

    assert isinstance(restored, QpeResult)
    assert restored.phase_fraction == pytest.approx(0.25)
    assert restored.raw_energy == pytest.approx(-0.25)


def test_empty_data_type_name_fails_before_writing(tmp_path, monkeypatch):
    """An invalid wire-format identifier does not create an output file."""
    result = QpeResult.from_phase_fraction(
        method="test",
        phase_fraction=0.25,
        eigenvalue_from_phase=lambda phase: -phase,
    )
    monkeypatch.setattr(QpeResult, "data_type_name", staticmethod(lambda: ""))

    with pytest.raises(TypeError, match="non-empty data_type_name"):
        serialize_outputs(tmp_path, result)

    assert not list(tmp_path.iterdir())


def test_serialization_rejects_mismatched_instance_name(tmp_path, monkeypatch):
    """Serialization rejects divergent class and instance identifiers."""
    result = QpeResult.from_phase_fraction(
        method="test",
        phase_fraction=0.25,
        eigenvalue_from_phase=lambda phase: -phase,
    )
    monkeypatch.setattr(QpeResult, "get_data_type_name", lambda _self: "other_result")

    with pytest.raises(TypeError, match="inconsistent wire-format identifiers"):
        serialize_outputs(tmp_path, result)

    assert not list(tmp_path.iterdir())


def test_registered_plugin_loader_round_trips_subclass(tmp_path, monkeypatch):
    """A plugin loader may restore an inherited concrete subtype."""
    monkeypatch.setattr(dataclass_registry, "_DATACLASS_REGISTRY", {})

    class PluginLoader(DataClass):
        """Test loader for plugin-defined data."""

        @staticmethod
        def data_type_name() -> str:
            """Return the test wire-format identifier."""
            return "test.plugin_data"

        def __init__(self, kind: str = "child") -> None:
            """Initialize a test family value."""
            self.kind = kind
            super().__init__()

        def get_summary(self) -> str:
            """Return a summary of the test value."""
            return f"PluginLoader(kind={self.kind!r})"

        def to_json(self) -> dict[str, str]:
            """Serialize the subtype discriminator to JSON."""
            return {"kind": self.kind}

        def to_hdf5(self, group) -> None:
            """Serialize the subtype discriminator to HDF5."""
            group.attrs["kind"] = self.kind

        @classmethod
        def from_json(cls, data: dict[str, str]):
            """Restore the encoded plugin subtype from JSON."""
            return PluginChild(data["kind"])

        @classmethod
        def from_hdf5(cls, group):
            """Restore the encoded plugin subtype from HDF5."""
            return PluginChild(str(group.attrs["kind"]))

        def _hash_update(self, hasher) -> None:
            """Add the subtype discriminator to a content hash."""
            hasher.update(self.kind.encode("utf-8"))

    class PluginChild(PluginLoader):
        """Concrete subtype restored by the test family loader."""

    register_dataclass(PluginLoader)
    original = PluginChild()

    serialize_outputs(tmp_path, original)
    restored = deserialize_outputs(tmp_path)

    assert type(restored) is PluginChild
    assert restored.kind == original.kind


def test_discovery_selects_class_that_declares_wire_type():
    """Inherited identifiers do not replace the canonical loader."""
    assert get_dataclass_type("settings") is Settings


def test_registration_explains_missing_plugin_type_name():
    """Plugin DataClasses without a static wire type receive actionable guidance."""

    class PluginDataWithoutTypeName(DataClass):
        pass

    with pytest.raises(TypeError, match=r"static data_type_name.*register the DataClass in a plugin"):
        register_dataclass(PluginDataWithoutTypeName)


def test_registration_rejects_builtin_wire_type_before_discovery(monkeypatch):
    """Plugin registration cannot claim a built-in loader's wire type."""
    monkeypatch.setattr(dataclass_registry, "_DATACLASS_REGISTRY", {})
    monkeypatch.setattr(dataclass_registry, "_DISCOVERY_COMPLETE", False)

    class PluginSettings(DataClass):
        """Plugin class claiming the Settings wire type."""

        @staticmethod
        def data_type_name():
            """Return the colliding wire type."""
            return Settings.data_type_name()

    with pytest.raises(DuplicateRegistrationError, match="already registered"):
        register_dataclass(PluginSettings)

    assert get_dataclass_type("settings") is Settings


def test_registration_discovery_does_not_recurse(monkeypatch):
    """Discovery uses the internal registration path without re-entering itself."""
    monkeypatch.setattr(dataclass_registry, "_DATACLASS_REGISTRY", {})
    monkeypatch.setattr(dataclass_registry, "_DISCOVERY_COMPLETE", False)
    original_discover = dataclass_registry._discover_imported_dataclasses
    discovery_active = False

    def tracked_discover(*, excluded_types=frozenset()):
        nonlocal discovery_active
        assert not discovery_active
        discovery_active = True
        try:
            return original_discover(excluded_types=excluded_types)
        finally:
            discovery_active = False

    monkeypatch.setattr(dataclass_registry, "_discover_imported_dataclasses", tracked_discover)

    register_dataclass(Settings)

    assert get_dataclass_type("settings") is Settings


def test_registration_rejects_duplicate_wire_type(monkeypatch):
    """Two plugin classes cannot silently claim the same serialized identifier."""
    monkeypatch.setattr(dataclass_registry, "_DATACLASS_REGISTRY", {})

    class FirstPluginData(DataClass):
        """First plugin class claiming the test wire type."""

        @staticmethod
        def data_type_name():
            """Return the shared test wire type."""
            return "test.plugin_data"

    class SecondPluginData(DataClass):
        """Second plugin class claiming the test wire type."""

        @staticmethod
        def data_type_name():
            """Return the shared test wire type."""
            return "test.plugin_data"

    register_dataclass(FirstPluginData)

    with pytest.raises(DuplicateRegistrationError, match="already registered"):
        register_dataclass(SecondPluginData)

    assert get_dataclass_type("test.plugin_data") is FirstPluginData
