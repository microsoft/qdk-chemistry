"""Validate transparent figure assets used by the molecular-QPE tutorial."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import ast
import importlib.util
import re
import sys
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
    "tutorial_qpe_orbital_entropy.png",
    "tutorial_qpe_phase_wrapping.png",
    "tutorial_qpe_power_one_circuit_overview.png",
    "tutorial_qpe_state_preparation_comparison.png",
}
PNG_FIGURES = TRANSPARENT_PNG_FIGURES | OPAQUE_PNG_FIGURES
FIGURE_PATTERN = re.compile(r"^\.\. figure:: /_static/diagrams/(tutorial_qpe_\S+)$", re.MULTILINE)
GRAPHVIZ_PATTERN = re.compile(r"^\.\. graphviz:: /_static/diagrams/(tutorial_qpe_\S+\.dot)$", re.MULTILINE)


def test_tutorial_figure_references_are_complete():
    """Keep the six tutorial figure directives aligned with committed assets."""
    references = {
        match
        for path in TUTORIAL_DIR.glob("*.rst")
        for match in FIGURE_PATTERN.findall(path.read_text(encoding="utf-8"))
    }

    assert references == PNG_FIGURES
    assert all((DIAGRAMS_DIR / name).is_file() for name in references)


def test_tutorial_png_figures_have_real_transparency():
    """Require orbital PNGs to contain both transparent and opaque pixels."""
    for name in TRANSPARENT_PNG_FIGURES:
        with Image.open(DIAGRAMS_DIR / name) as image:
            assert image.mode == "RGBA", name
            assert image.getchannel("A").getextrema() == (0, 255), name


def test_opaque_png_figures_use_light_gray_backgrounds():
    """Keep fixed-color plots and circuits readable against dark themes."""
    for name in OPAQUE_PNG_FIGURES:
        with Image.open(DIAGRAMS_DIR / name) as image:
            rgba = image.convert("RGBA")
            assert rgba.getchannel("A").getextrema() == (255, 255), name
            assert rgba.getpixel((0, 0)) == (242, 242, 242, 255), name


def test_tutorial_graphviz_figures_render_as_transparent_svg():
    """Require SVG output and transparent canvases for every tutorial DOT file."""
    references = {
        match
        for path in TUTORIAL_DIR.glob("*.rst")
        for match in GRAPHVIZ_PATTERN.findall(path.read_text(encoding="utf-8"))
    }
    dot_files = {path.name for path in DIAGRAMS_DIR.glob("tutorial_qpe_*.dot")}

    assert references == dot_files
    for name in references:
        dot_source = (DIAGRAMS_DIR / name).read_text(encoding="utf-8")
        assert 'bgcolor="transparent"' in dot_source
        if name != "tutorial_qpe_wavefunction_hierarchy.dot":
            assert "<SUB>" not in dot_source

    hierarchy_source = (DIAGRAMS_DIR / "tutorial_qpe_wavefunction_hierarchy.dot").read_text(encoding="utf-8")
    assert "χ[μ]" not in hierarchy_source
    assert "Φ(HF)" not in hierarchy_source
    assert 'χ&#160;<SUB><FONT POINT-SIZE="9">μ</FONT></SUB>' in hierarchy_source
    assert 'Φ&#160;<SUB><FONT POINT-SIZE="9">HF</FONT></SUB>' in hierarchy_source
    unapproved_subscripts = re.sub(
        r'&#160;<SUB><FONT POINT-SIZE="9">(?:μ|HF)</FONT></SUB>',
        "",
        hierarchy_source,
    )
    assert "<SUB>" not in unapproved_subscripts

    configuration = ast.parse((REPOSITORY_ROOT / "docs" / "source" / "conf.py").read_text(encoding="utf-8"))
    output_formats = [
        node.value.value
        for node in configuration.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "graphviz_output_format" for target in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert output_formats == ["svg"]


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
