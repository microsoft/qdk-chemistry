"""Regression tests for native binding documentation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("sphinx")
pytest.importorskip("sphinx_autodoc_typehints")
pytest.importorskip("qdk_chemistry")


@pytest.fixture
def docs_conf(monkeypatch):
    """Load the real documentation hooks without starting a full docs build."""
    path = Path(__file__).parents[1] / "source" / "conf.py"
    spec = importlib.util.spec_from_file_location("qdk_docs_conf", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (
            "tuple[numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False], "
            "numpy.ndarray[dtype=float64, shape=(*, *), order='F', writable=False]]",
            "tuple[numpy.ndarray, numpy.ndarray]",
        ),
        (
            "tuple[float, numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]]",
            "tuple[float, numpy.ndarray]",
        ),
        ("list[tuple[int, str]] | None", "list[tuple[int, str]] | None"),
    ],
)
def test_native_array_annotations(docs_conf, annotation, expected):
    """Normalize native ndarray metadata without flattening enclosing types."""
    result = docs_conf.normalize_autodoc_signature(
        None, "method", "get_arrays", None, None, "()", annotation
    )
    assert (result[1] if result is not None else annotation) == expected


def test_native_functions_and_aliases_build_without_warnings(
    docs_conf, monkeypatch, tmp_path
):
    """Render native-like exports and overloads with resolvable public links."""
    from sphinx.application import Sphinx

    native_name = "qdk_chemistry._core.docs_probe"
    public_name = "qdk_chemistry.docs_probe"
    native = ModuleType(native_name)
    public = ModuleType(public_name)
    array = "numpy.ndarray[dtype=float64, shape=(*), order='C', writable=False]"

    class NativeFunction:
        __module__ = native_name
        __nb_signature__ = ()

        def __init__(self):
            self.__name__ = "arrays"
            self.__qualname__ = "arrays"
            self.__doc__ = (
                f"arrays() -> tuple[{array}, {array}]\n"
                f"arrays(count: int) -> tuple[float, {array}]\n\n"
                "Return arrays."
            )

        def __call__(self, *args, **kwargs):
            return args, kwargs

    class NativeMethods:
        __module__ = public_name
        __qualname__ = "NativeMethods"

        def get_arrays(self):
            """Return arrays."""

    NativeMethods.get_arrays.__doc__ = (
        f"get_arrays(self) -> tuple[{array}, {array}]\n"
        f"get_arrays(self, count: int) -> tuple[float, {array}]\n\n"
        "Return arrays."
    )
    NativeMethods.get_arrays.__module__ = public_name
    NativeMethods.get_arrays.__qualname__ = "NativeMethods.get_arrays"
    public.__dict__.update(
        arrays=NativeFunction(),
        NativeMethods=NativeMethods,
        __all__=["arrays", "NativeMethods"],
    )
    monkeypatch.setitem(sys.modules, native_name, native)
    monkeypatch.setitem(sys.modules, public_name, public)
    monkeypatch.setattr(
        docs_conf,
        "_MODULE_ALIAS_RULES",
        ((native_name, public_name), *docs_conf._MODULE_ALIAS_RULES),
    )
    docs_conf.normalize_public_export_modules()
    assert public.arrays.__module__ == public_name

    (tmp_path / "conf.py").write_text(
        "from qdk_docs_conf import setup, nitpick_ignore_regex, autodoc_default_options\n"
        "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', "
        "'sphinx_autodoc_typehints']\n"
        "nitpicky = True\n"
        "autodoc_typehints = 'description'\n"
    )
    (tmp_path / "index.rst").write_text(
        "Native bindings\n===============\n\n"
        ".. py:class:: numpy.ndarray\n\n"
        ".. py:class:: tuple\n\n"
        ".. py:class:: float\n\n"
        ".. py:class:: int\n\n"
        ".. py:class:: qdk_chemistry.data.qubit_operator.QubitOperator\n\n"
        f".. automodule:: {public_name}\n   :members:\n\n"
        f":func:`{public_name}.arrays`\n\n"
        ":class:`qdk_chemistry.data.qubit_operator._DeprecatedQubitHamiltonian`\n\n"
        ":class:`qdk_chemistry.data.SumPauliOperatorExpression`\n"
    )
    warnings = io.StringIO()
    app = Sphinx(
        str(tmp_path),
        str(tmp_path),
        str(tmp_path / "_build"),
        str(tmp_path / "_doctrees"),
        "html",
        status=io.StringIO(),
        warning=warnings,
        freshenv=True,
        warningiserror=True,
    )
    app.build()
    assert not warnings.getvalue(), warnings.getvalue()
    objects = app.env.get_domain("py").objects
    assert objects[f"{public_name}.arrays"].objtype == "function"
    assert objects[f"{public_name}.NativeMethods.get_arrays"].objtype == "method"
    html = (tmp_path / "_build" / "index.html").read_text()
    assert "writable=False" not in html
    assert "#qdk_chemistry.data.qubit_operator.QubitOperator" in html
