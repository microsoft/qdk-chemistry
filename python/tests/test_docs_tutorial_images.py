"""Validate figure assets used by the molecular-QPE tutorial."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util
import re
import sys
import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).parent.parent.parent
DIAGRAMS_DIR = REPOSITORY_ROOT / "docs" / "source" / "_static" / "diagrams"
TUTORIAL_DIR = REPOSITORY_ROOT / "docs" / "source" / "tutorials" / "ground_state_molecular_energies_with_qpe"

TRANSPARENT_PNG_FIGURES = {
    "tutorial_qpe_atomic_basis_functions.png",
    "tutorial_qpe_example_molecular_orbitals.png",
}
OPAQUE_PNG_FIGURES = {
    "tutorial_qpe_power_one_circuit_overview.png",
    "tutorial_qpe_state_preparation_comparison.png",
}
PNG_FIGURES = TRANSPARENT_PNG_FIGURES | OPAQUE_PNG_FIGURES
GRAPHVIZ_SVG_FIGURES = {
    "tutorial_qpe_iqpe_iteration.svg",
    "tutorial_qpe_jordan_wigner_parity.svg",
    "tutorial_qpe_orbital_partition.svg",
    "tutorial_qpe_wavefunction_hierarchy.svg",
    "tutorial_qpe_workflow.svg",
}
PLOT_SVG_FIGURES = {
    "tutorial_qpe_orbital_entropy.svg",
    "tutorial_qpe_phase_wrapping.svg",
}
SVG_FIGURES = GRAPHVIZ_SVG_FIGURES | PLOT_SVG_FIGURES
FIGURE_PATTERN = re.compile(r"^\.\. figure:: /_static/diagrams/(tutorial_qpe_\S+)$", re.MULTILINE)
SVG_FIGURE_PATTERN = re.compile(
    r"^\.\. figure:: /_static/diagrams/(?P<source>tutorial_qpe_\S+\.svg)\n"
    r"(?P<options>(?:   :[^\n]+\n)*)",
    re.MULTILINE,
)
ALT_PATTERN = re.compile(r"^   :alt: (?P<alt>.+)$", re.MULTILINE)
SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def source_sha256(*paths: Path) -> str:
    """Hash named source files in argument order."""
    digest = sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_tutorial_figure_references_are_complete():
    """Keep all tutorial figure directives aligned with committed assets."""
    references = {
        match
        for path in TUTORIAL_DIR.glob("*.rst")
        for match in FIGURE_PATTERN.findall(path.read_text(encoding="utf-8"))
    }

    assert references == PNG_FIGURES | SVG_FIGURES
    assert all((DIAGRAMS_DIR / name).is_file() for name in references)


def test_tutorial_png_figures_have_real_transparency():
    """Require orbital PNGs to contain both transparent and opaque pixels."""
    for name in TRANSPARENT_PNG_FIGURES:
        with Image.open(DIAGRAMS_DIR / name) as image:
            assert image.mode == "RGBA", name
            assert image.getchannel("A").getextrema() == (0, 255), name


def test_opaque_png_figures_use_white_backgrounds():
    """Keep every fixed-color PNG on the shared opaque white canvas."""
    for name in OPAQUE_PNG_FIGURES:
        with Image.open(DIAGRAMS_DIR / name) as image:
            rgba = image.convert("RGBA")
            assert rgba.getchannel("A").getextrema() == (255, 255), name
            assert rgba.getpixel((0, 0)) == (255, 255, 255, 255), name


def test_circuit_png_figures_meet_non_text_contrast():
    """Keep circuit structure above the WCAG non-text contrast threshold."""
    background = np.array([255, 255, 255], dtype=np.uint8)
    structure = np.array([143, 143, 143], dtype=np.uint8)

    def relative_luminance(color: np.ndarray) -> float:
        channels = color.astype(np.float64) / 255.0
        linear = np.where(
            channels <= 0.04045,
            channels / 12.92,
            ((channels + 0.055) / 1.055) ** 2.4,
        )
        return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])

    lighter = relative_luminance(background)
    darker = relative_luminance(structure)
    assert (lighter + 0.05) / (darker + 0.05) >= 3.0

    for name in (
        "tutorial_qpe_power_one_circuit_overview.png",
        "tutorial_qpe_state_preparation_comparison.png",
    ):
        with Image.open(DIAGRAMS_DIR / name) as image:
            rgb = np.asarray(image.convert("RGB"))
        assert np.any(np.all(rgb == structure, axis=2)), name


def test_graphviz_sources_generate_accessible_svg_figures():
    """Keep committed SVGs aligned with DOT sources and authored descriptions."""
    directives = [
        (path, match)
        for path in TUTORIAL_DIR.glob("*.rst")
        for match in SVG_FIGURE_PATTERN.finditer(path.read_text(encoding="utf-8"))
        if match.group("source") in GRAPHVIZ_SVG_FIGURES
    ]
    references = {match.group("source") for _, match in directives}
    dot_files = {path.name for path in DIAGRAMS_DIR.glob("tutorial_qpe_*.dot")}
    svg_files = {path.name for path in DIAGRAMS_DIR.glob("tutorial_qpe_*.svg") if path.name in GRAPHVIZ_SVG_FIGURES}

    assert references == {name.replace(".dot", ".svg") for name in dot_files}
    assert references == svg_files
    assert len(directives) == 5
    for rst_path, match in directives:
        svg_name = match.group("source")
        dot_name = svg_name.replace(".svg", ".dot")
        dot_path = DIAGRAMS_DIR / dot_name
        dot_source = dot_path.read_text(encoding="utf-8")
        assert 'bgcolor="#FFFFFF"' in dot_source
        if dot_name != "tutorial_qpe_wavefunction_hierarchy.dot":
            assert "<SUB>" not in dot_source
        alt_match = ALT_PATTERN.search(match.group("options"))
        assert alt_match is not None, f"{rst_path}: {svg_name}"

        svg_path = DIAGRAMS_DIR / svg_name
        root = ET.parse(svg_path).getroot()
        svg_id = dot_path.stem.replace("_", "-")
        assert root.attrib["role"] == "img"
        assert root.attrib["aria-labelledby"] == f"{svg_id}-title {svg_id}-desc"
        assert root.attrib["data-source-sha256"] == source_sha256(
            dot_path,
            DIAGRAMS_DIR / "generate_tutorial_qpe_graphviz.py",
            DIAGRAMS_DIR / "tutorial_qpe_svg.py",
        )
        title = root.find(f"{{{SVG_NAMESPACE}}}title")
        description = root.find(f"{{{SVG_NAMESPACE}}}desc")
        assert title is not None
        assert title.text
        assert description is not None
        assert description.text == alt_match.group("alt")
        assert "Arial Bold" not in dot_source
        svg_text = svg_path.read_text(encoding="utf-8")
        assert 'fill="#ffffff"' in svg_text
        assert 'font-weight="bold"' in svg_text


def test_matplotlib_sources_generate_accessible_svg_figures():
    """Keep committed plot SVGs aligned with generators and authored descriptions."""
    directives = {
        match.group("source"): (path, match)
        for path in TUTORIAL_DIR.glob("*.rst")
        for match in SVG_FIGURE_PATTERN.finditer(path.read_text(encoding="utf-8"))
        if match.group("source") in PLOT_SVG_FIGURES
    }
    source_paths = {
        "tutorial_qpe_phase_wrapping.svg": (
            DIAGRAMS_DIR / "generate_tutorial_qpe_phase_grid.py",
            DIAGRAMS_DIR / "tutorial_qpe_svg.py",
        ),
        "tutorial_qpe_orbital_entropy.svg": (
            DIAGRAMS_DIR / "generate_tutorial_qpe_orbital_entropy.py",
            DIAGRAMS_DIR / "tutorial_qpe_svg.py",
            DIAGRAMS_DIR.parent / "examples" / "python" / "tutorial_choose_active_space.py",
        ),
    }

    assert directives.keys() == PLOT_SVG_FIGURES
    for svg_name, paths in source_paths.items():
        rst_path, match = directives[svg_name]
        alt_match = ALT_PATTERN.search(match.group("options"))
        assert alt_match is not None, f"{rst_path}: {svg_name}"
        svg_path = DIAGRAMS_DIR / svg_name
        root = ET.parse(svg_path).getroot()
        svg_id = svg_path.stem.replace("_", "-")
        assert root.attrib["role"] == "img"
        assert root.attrib["aria-labelledby"] == f"{svg_id}-title {svg_id}-desc"
        assert root.attrib["data-source-sha256"] == source_sha256(*paths)
        title = root.find(f"{{{SVG_NAMESPACE}}}title")
        description = root.find(f"{{{SVG_NAMESPACE}}}desc")
        assert title is not None
        assert title.text
        assert description is not None
        assert description.text == alt_match.group("alt")
        svg_text = svg_path.read_text(encoding="utf-8")
        assert "fill: #ffffff" in svg_text
        assert "<image" not in svg_text


def test_screenshot_derived_images_match_local_sources():
    """Recreate each screenshot-derived output from local source assets."""
    source_dir = REPOSITORY_ROOT / "docs" / "figure_sources" / "ground_state_qpe"
    script_path = DIAGRAMS_DIR / "generate_tutorial_qpe_screenshot_images.py"
    sys.path.insert(0, str(DIAGRAMS_DIR))
    try:
        module_spec = importlib.util.spec_from_file_location(
            "generate_tutorial_qpe_screenshot_images",
            script_path,
        )
        assert module_spec is not None
        assert module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    for name in module.ORBITAL_SCREENSHOTS:
        with Image.open(source_dir / name) as source_image:
            expected = module.white_to_alpha(source_image)
        with Image.open(DIAGRAMS_DIR / name) as published:
            assert np.array_equal(
                np.asarray(expected),
                np.asarray(published.convert("RGBA")),
            )

    for name in module.CIRCUIT_SCREENSHOTS:
        with Image.open(source_dir / name) as source_image:
            expected = module.replace_circuit_background(source_image)
        with Image.open(DIAGRAMS_DIR / name) as published:
            assert np.array_equal(
                np.asarray(expected),
                np.asarray(published.convert("RGB")),
            )


def test_white_to_alpha_preserves_white_composite():
    """Color-to-alpha conversion must not change the source appearance on white."""
    script_path = DIAGRAMS_DIR / "convert_white_to_alpha.py"
    module_spec = importlib.util.spec_from_file_location("convert_white_to_alpha", script_path)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    source = Image.new("RGB", (3, 2))
    source.putdata(
        [
            (255, 255, 255),
            (0, 0, 0),
            (200, 220, 255),
            (12, 90, 180),
            (254, 254, 254),
            (255, 128, 0),
        ]
    )
    converted = module.white_to_alpha(source)
    white_background = Image.new("RGBA", converted.size, "white")
    white_background.alpha_composite(converted)

    assert converted.getchannel("A").getextrema() == (0, 255)
    assert np.array_equal(np.asarray(white_background.convert("RGB")), np.asarray(source))
