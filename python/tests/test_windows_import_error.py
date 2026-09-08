"""Tests for the Windows native-extension import diagnostic."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib
import importlib.util
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("system", ["win32", "linux", "darwin"])
def test_native_import_error_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
) -> None:
    """Report the bundled Windows runtime prerequisite and preserve non-Windows errors."""
    spec = importlib.util.find_spec("qdk_chemistry")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original = ImportError("localized native extension loader error")
    import_module = importlib.import_module

    def fail_import(name: str, package: str | None = None) -> ModuleType:
        if name != "qdk_chemistry._core":
            return import_module(name, package)
        monkeypatch.setattr(module, "_sys", SimpleNamespace(platform=system))
        raise original

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(ImportError) as caught:
        spec.loader.exec_module(module)

    if system == "win32":
        assert "Visual C++ v14" in str(caught.value)
        assert "includes ARM64" in str(caught.value)
        assert str(caught.value).endswith("https://aka.ms/vc14/vc_redist.x64.exe")
        assert caught.value.__cause__ is original
    else:
        assert caught.value is original
