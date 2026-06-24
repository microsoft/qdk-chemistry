# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
"""Sphinx extension: inline ``.xyz`` files into example snippets at build time.

The example scripts under ``docs/source/_static/examples`` load molecular
geometries from ``.xyz`` files, e.g.::

    structure = Structure.from_xyz_file(Path(__file__).parent / "../data/h2.structure.xyz")

Keeping the geometry in standalone ``.xyz`` files gives us a single source of
truth (the files are also used when the examples are executed as part of the
test suite). However, when these snippets are rendered into the documentation
via ``literalinclude``, a reader cannot access ``../data/h2.structure.xyz``.

This extension rewrites ``from_xyz_file(<path-to-.xyz>)`` calls into an
equivalent ``from_xyz(<embedded-contents>)`` call *only in the rendered
documentation*, and only for calls that explicitly opt in (see below). The
files on disk are left untouched, so the examples remain runnable and the
``.xyz`` files remain the single source of truth.

Python snippets become::

    structure = Structure.from_xyz(\"\"\"\\
    2
    H2 molecule
    H    0.000000    0.000000    0.000000
    H    0.000000    0.000000    0.740848
    \"\"\")

C++ snippets become::

    auto structure = Structure::from_xyz(R"(2
    H2 molecule
    H    0.000000    0.000000    0.000000
    H    0.000000    0.000000    0.740848
    )");

The rewrite hooks into ``LiteralIncludeReader.read_file`` because the
``literalinclude`` directive (unlike ``include``) does not emit the
``include-read`` event.

Opt-in
------
Inlining is opt-in: by default the rendered docs show exactly what is in the
file. A call is inlined only when it carries the marker comment
``docs:inline-xyz`` anywhere within the call (so it also works when a formatter
wraps the call across multiple lines)::

    structure = Structure.from_xyz_file(path)  # docs:inline-xyz

The marker itself is stripped from the rendered output so readers never see it.
Calls without the marker (e.g. the ``Structure`` file-loading tutorial) are left
verbatim as ``from_xyz_file`` calls.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx
from sphinx.directives import code as _code
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Directory containing the example snippets that may reference ``.xyz`` files.
_EXAMPLES_DIR = (
    Path(__file__).resolve().parent.parent / "_static" / "examples"
).resolve()

# Source file types whose ``from_xyz_file`` calls should be inlined.
_PYTHON_SUFFIXES = {".py"}
_CPP_SUFFIXES = {".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h"}

# Matches the start of a ``from_xyz_file(`` call (Python or C++).
_CALL_RE = re.compile(r"from_xyz_file\s*\(")

# Matches a quoted string literal that ends in ``.xyz``.
_XYZ_PATH_RE = re.compile(r"""['"]([^'"]*\.xyz)['"]""")

# Marker comment that opts a call *into* inlining (rewritten to ``from_xyz``).
# The marker is stripped from the rendered output.
_INLINE_MARKER = "docs:inline-xyz"
_INLINE_MARKER_STRIP_RE = re.compile(
    r"[ \t]*(?:#|//)[ \t]*" + re.escape(_INLINE_MARKER) + r"[ \t]*$"
)


def _find_matching_paren(text: str, open_idx: int) -> int:
    """Return the index of the ``)`` matching the ``(`` at ``open_idx``.

    String literals are skipped so that parentheses inside quotes are ignored.
    Returns ``-1`` if no matching parenthesis is found.
    """
    depth = 0
    i = open_idx
    in_str: str | None = None
    while i < len(text):
        ch = text[i]
        if in_str is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _embed(contents: str, suffix: str) -> str:
    """Build the inline ``from_xyz(...)`` replacement for the given contents."""
    if not contents.endswith("\n"):
        contents += "\n"
    if suffix in _PYTHON_SUFFIXES:
        return 'from_xyz("""\\\n' + contents + '""")'
    # C/C++ raw string literal keeps the contents verbatim.
    return 'from_xyz(R"(' + contents + ')")'


def _strip_inline_markers(text: str) -> str:
    """Remove trailing ``docs:inline-xyz`` marker comments from each line."""
    return "\n".join(_INLINE_MARKER_STRIP_RE.sub("", line) for line in text.split("\n"))


def _inline_xyz_in_text(text: str, example_dir: Path, suffix: str) -> str:
    """Replace ``from_xyz_file(...)`` calls referencing ``.xyz`` files inline."""
    pieces: list[str] = []
    pos = 0
    for match in _CALL_RE.finditer(text):
        open_paren = match.end() - 1
        close_paren = _find_matching_paren(text, open_paren)
        if close_paren == -1:
            continue
        # Only inline calls that opt in via the marker anywhere within the
        # (possibly multi-line) call, including trailing comments.
        call_span_start = text.rfind("\n", 0, match.start()) + 1
        call_span_end = text.find("\n", close_paren)
        if call_span_end == -1:
            call_span_end = len(text)
        if _INLINE_MARKER not in text[call_span_start:call_span_end]:
            continue
        inner = text[open_paren + 1 : close_paren]
        path_match = _XYZ_PATH_RE.search(inner)
        if path_match is None:
            continue
        xyz_path = (example_dir / path_match.group(1)).resolve()
        if not xyz_path.is_file():
            continue
        contents = xyz_path.read_text(encoding="utf-8")
        pieces.append(text[pos : match.start()])
        pieces.append(_embed(contents, suffix))
        pos = close_paren + 1
    if not pieces:
        return text
    pieces.append(text[pos:])
    return "".join(pieces)


def _maybe_inline_xyz(filename: Path, lines: list[str]) -> list[str]:
    """Inline ``.xyz`` contents for example snippets; otherwise return as-is."""
    resolved = filename.resolve()
    if _EXAMPLES_DIR not in resolved.parents:
        return lines
    suffix = resolved.suffix.lower()
    if suffix not in _PYTHON_SUFFIXES and suffix not in _CPP_SUFFIXES:
        return lines

    text = "".join(lines)
    if _INLINE_MARKER not in text:
        return lines

    new_text = _inline_xyz_in_text(text, resolved.parent, suffix)
    new_text = _strip_inline_markers(new_text)
    if new_text == text:
        return lines
    return new_text.splitlines(keepends=True)


_orig_read_file = _code.LiteralIncludeReader.read_file


def _patched_read_file(self, filename, location=None):  # type: ignore[no-untyped-def]
    lines = _orig_read_file(self, filename, location=location)
    try:
        return _maybe_inline_xyz(Path(filename), lines)
    except Exception as exc:  # pragma: no cover - never break the build
        logger.warning("inline_xyz: failed to inline %s: %s", filename, exc)
        return lines


def setup(app: Sphinx) -> dict[str, Any]:
    _code.LiteralIncludeReader.read_file = _patched_read_file
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
