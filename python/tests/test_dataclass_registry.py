"""Tests for serializable DataClass discovery and registration."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

import qdk_chemistry.data as data_module
import qdk_chemistry.data.symmetry as symmetry_module
from qdk_chemistry._core.data import DataClass as CoreDataClass
from qdk_chemistry.data import DataClass, Settings, get_dataclass_type, register_dataclass
from qdk_chemistry.data import registry as dataclass_registry
from qdk_chemistry.data._type_name import class_data_type_name, instance_data_type_name
from qdk_chemistry.plugins import DuplicateRegistrationError


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
