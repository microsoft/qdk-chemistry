"""Generate accessible molecular-QPE SVG figures from committed DOT sources."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import hashlib
import re
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

DIAGRAMS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = (
    DIAGRAMS_DIR.parents[1] / "tutorials" / "ground_state_molecular_energies_with_qpe"
)

TITLES = {
    "tutorial_qpe_workflow": "Molecular QPE tutorial workflow",
    "tutorial_qpe_wavefunction_hierarchy": "Wavefunction representation hierarchy",
    "tutorial_qpe_orbital_partition": "Active-space orbital partition",
    "tutorial_qpe_jordan_wigner_parity": "Jordan-Wigner parity mapping",
    "tutorial_qpe_iqpe_iteration": "Iterative phase-estimation iteration",
}
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


def accessible_svg(dot_path: Path, title: str, description: str) -> str:
    """Render one DOT source and add stable accessibility metadata."""
    source = dot_path.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    result = subprocess.run(
        ["dot", "-Tsvg", str(dot_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    svg_id = dot_path.stem.replace("_", "-")
    root_match = SVG_PATTERN.search(result.stdout)
    if root_match is None:
        raise ValueError(f"Graphviz did not produce an SVG root for {dot_path}")
    root = root_match.group(0)[:-1]
    accessible_root = (
        f'{root} role="img" aria-labelledby="{svg_id}-title {svg_id}-desc" '
        f'data-source-sha256="{source_hash}">'
    )
    metadata = (
        f'\n<title id="{svg_id}-title">{escape(title)}</title>'
        f'\n<desc id="{svg_id}-desc">{escape(description)}</desc>'
    )
    return (
        result.stdout[: root_match.start()]
        + accessible_root
        + metadata
        + result.stdout[root_match.end() :]
    )


def main() -> None:
    """Regenerate all committed molecular-QPE Graphviz SVGs."""
    descriptions = figure_descriptions()
    expected = {f"{stem}.svg" for stem in TITLES}
    if descriptions.keys() != expected:
        raise ValueError(
            "Expected one documented figure for each generated SVG; "
            f"documented={sorted(descriptions)}, expected={sorted(expected)}"
        )

    for stem, title in TITLES.items():
        dot_path = DIAGRAMS_DIR / f"{stem}.dot"
        svg_path = DIAGRAMS_DIR / f"{stem}.svg"
        svg_path.write_text(
            accessible_svg(dot_path, title, descriptions[svg_path.name]),
            encoding="utf-8",
        )
        print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
