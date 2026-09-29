"""Shared helpers for accessible, reproducible molecular-QPE SVG assets."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import hashlib
import re
from pathlib import Path
from xml.sax.saxutils import escape

DIAGRAMS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = (
    DIAGRAMS_DIR.parents[1] / "tutorials" / "ground_state_molecular_energies_with_qpe"
)

FIGURE_PATTERN = re.compile(
    r"^\.\. figure:: /_static/diagrams/(?P<name>tutorial_qpe_\S+\.svg)\n"
    r"(?P<options>(?:   :[^\n]+\n)*)",
    re.MULTILINE,
)
ALT_PATTERN = re.compile(r"^   :alt: (?P<alt>.+)$", re.MULTILINE)
SVG_PATTERN = re.compile(r"<svg\b[^>]*>", re.DOTALL)


def figure_descriptions() -> dict[str, str]:
    """Return SVG names mapped to their authored RST alternative text."""
    descriptions = {}
    for rst_path in TUTORIAL_DIR.glob("*.rst"):
        source = rst_path.read_text(encoding="utf-8")
        for figure_match in FIGURE_PATTERN.finditer(source):
            alt_match = ALT_PATTERN.search(figure_match.group("options"))
            if alt_match is None:
                raise ValueError(
                    f"{rst_path}: missing alt text for {figure_match.group('name')}"
                )
            descriptions[figure_match.group("name")] = alt_match.group("alt")
    return descriptions


def source_sha256(*paths: Path) -> str:
    """Hash named source files in argument order."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def add_accessibility_metadata(
    svg: str,
    *,
    identifier: str,
    title: str,
    description: str,
    source_hash: str,
) -> str:
    """Add accessible labels and a source hash to an SVG document."""
    svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    root_match = SVG_PATTERN.search(svg)
    if root_match is None:
        raise ValueError("Generated content does not contain an SVG root")
    root = root_match.group(0)[:-1]
    accessible_root = (
        f'{root} role="img" aria-labelledby="{identifier}-title {identifier}-desc" '
        f'data-source-sha256="{source_hash}">'
    )
    metadata = (
        f'\n<title id="{identifier}-title">{escape(title)}</title>'
        f'\n<desc id="{identifier}-desc">{escape(description)}</desc>'
    )
    return (
        svg[: root_match.start()] + accessible_root + metadata + svg[root_match.end() :]
    )
