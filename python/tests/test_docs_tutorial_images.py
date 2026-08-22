"""Validate transparent figure assets used by the molecular-QPE tutorial."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).parent.parent.parent
DIAGRAMS_DIR = REPOSITORY_ROOT / "docs" / "source" / "_static" / "diagrams"
TUTORIAL_DIR = REPOSITORY_ROOT / "docs" / "source" / "tutorials" / "ground_state_molecular_energies_with_qpe"

PNG_FIGURES = {
    "tutorial_qpe_atomic_basis_functions.png",
    "tutorial_qpe_example_molecular_orbitals.png",
    "tutorial_qpe_orbital_entropy.png",
    "tutorial_qpe_phase_wrapping.png",
}
SVG_FIGURES = {
    "tutorial_qpe_power_one_circuit_overview.svg",
    "tutorial_qpe_state_preparation_comparison.svg",
}
FIGURE_PATTERN = re.compile(r"^\.\. figure:: /_static/diagrams/(tutorial_qpe_\S+)$", re.MULTILINE)


def test_tutorial_figure_references_are_complete():
    """Keep the six tutorial figure directives aligned with committed assets."""
    references = {
        match
        for path in TUTORIAL_DIR.glob("*.rst")
        for match in FIGURE_PATTERN.findall(path.read_text(encoding="utf-8"))
    }

    assert references == PNG_FIGURES | SVG_FIGURES
    assert all((DIAGRAMS_DIR / name).is_file() for name in references)


def test_tutorial_png_figures_have_real_transparency():
    """Require every PNG figure to contain both transparent and opaque pixels."""
    for name in PNG_FIGURES:
        with Image.open(DIAGRAMS_DIR / name) as image:
            assert image.mode == "RGBA", name
            assert image.getchannel("A").getextrema() == (0, 255), name


def test_circuit_svgs_preserve_transparent_structure():
    """Check transparent styling and the wires and labels used by the tutorial."""
    for name in SVG_FIGURES:
        path = DIAGRAMS_DIR / name
        root = ET.parse(path).getroot()
        text = path.read_text(encoding="utf-8")

        assert root.tag.endswith("svg"), name
        assert "--circuit-bg: transparent" in text, name
        assert "currentColor" in text, name
        assert ".dropzone-layer { display: none; }" in text, name

    iqpe_path = DIAGRAMS_DIR / "tutorial_qpe_power_one_circuit_overview.svg"
    iqpe_root = ET.parse(iqpe_path).getroot()
    iqpe_text = "".join(iqpe_root.itertext())
    wire_indices = {int(element.attrib["data-wire"]) for element in iqpe_root.iter() if "data-wire" in element.attrib}

    assert wire_indices == set(range(13))
    assert "MakeIQPECircuit" in iqpe_text
    assert "RunIQPE" in iqpe_text
    assert "RepControlledPauliExp" in iqpe_text


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
