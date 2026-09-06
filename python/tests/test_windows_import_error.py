"""Tests for the Windows native-extension import diagnostic."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from types import SimpleNamespace

import pytest

import qdk_chemistry


def test_windows_dll_load_error_identifies_redistributable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the machine prerequisite when Windows cannot load the extension."""
    original = ImportError("DLL load failed while importing _core: The specified module could not be found.")

    def fail_import(_: str) -> None:
        raise original

    monkeypatch.setattr(
        qdk_chemistry,
        "importlib",
        SimpleNamespace(import_module=fail_import),
    )
    monkeypatch.setattr(qdk_chemistry, "_sys", SimpleNamespace(platform="win32"))

    with pytest.raises(ImportError, match="Visual C\\+\\+ v14") as caught:
        qdk_chemistry._import_core()

    assert caught.value.__cause__ is original
