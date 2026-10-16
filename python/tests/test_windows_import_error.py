"""Tests for the native-extension import diagnostics."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib
import importlib.util
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("system", ["win32", "linux", "darwin"])
@pytest.mark.parametrize("error_type", [ImportError, ModuleNotFoundError])
@pytest.mark.parametrize(
    "message",
    ["localized native extension loader error", "libgomp.so.1: cannot allocate memory in static TLS block"],
    ids=["loader_error", "static_tls"],
)
def test_native_import_error_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    error_type: type[ImportError],
    message: str,
) -> None:
    """Report actionable load failures without masking missing modules or other errors."""
    spec = importlib.util.find_spec("qdk_chemistry")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original = error_type(message, name="qdk_chemistry._core")
    import_module = importlib.import_module

    def fail_import(name: str, package: str | None = None) -> ModuleType:
        if name != "qdk_chemistry._core":
            return import_module(name, package)
        monkeypatch.setattr(module, "_sys", SimpleNamespace(platform=system))
        raise original

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(ImportError) as caught:
        spec.loader.exec_module(module)

    if system == "win32" and error_type is ImportError:
        assert "Visual C++ v14" in str(caught.value)
        assert "includes ARM64" in str(caught.value)
        assert "https://aka.ms/vc14/vc_redist.x64.exe" in str(caught.value)
        assert caught.value.__cause__ is original
    elif error_type is ImportError and "static TLS" in message:
        assert "GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536" in str(caught.value)
        assert caught.value.__cause__ is original
    else:
        assert caught.value is original
