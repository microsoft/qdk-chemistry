"""Generate published molecular-QPE images from the original screenshots."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from convert_white_to_alpha import white_to_alpha

DIAGRAMS_DIR = Path(__file__).resolve().parent
SOURCE_DIR = Path(__file__).resolve().parents[3] / "figure_sources" / "ground_state_qpe"

ORBITAL_SCREENSHOTS = (
    "tutorial_qpe_atomic_basis_functions.png",
    "tutorial_qpe_example_molecular_orbitals.png",
)
CIRCUIT_SCREENSHOTS = (
    "tutorial_qpe_state_preparation_comparison.png",
    "tutorial_qpe_power_one_circuit_overview.png",
)

CIRCUIT_BACKGROUND = 248
MIN_NEUTRAL_VALUE = 200
MAX_NEUTRAL_SPREAD = 5
MIN_BACKGROUND_COMPONENT_AREA = 1000


def replace_circuit_background(image: Image.Image) -> Image.Image:
    """Replace large white screenshot regions with a light-gray background."""
    rgb = np.asarray(image.convert("RGB")).copy()
    neutral = (rgb.min(axis=2) >= MIN_NEUTRAL_VALUE) & (
        (rgb.max(axis=2) - rgb.min(axis=2)) <= MAX_NEUTRAL_SPREAD
    )
    components, num_components = ndimage.label(
        neutral,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    areas = np.bincount(components.ravel())
    border_components = (
        set(components[0])
        | set(components[-1])
        | set(components[:, 0])
        | set(components[:, -1])
    )
    background_components = {
        index
        for index in range(1, num_components + 1)
        if index in border_components or areas[index] >= MIN_BACKGROUND_COMPONENT_AREA
    }
    background = np.isin(components, list(background_components))
    offset = 255 - CIRCUIT_BACKGROUND
    rgb[background] = np.maximum(
        rgb[background].astype(np.int16) - offset,
        0,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def main() -> None:
    """Regenerate all published screenshot-derived tutorial images."""
    for filename in ORBITAL_SCREENSHOTS:
        with Image.open(SOURCE_DIR / filename) as image:
            output = white_to_alpha(image)
        output_path = DIAGRAMS_DIR / filename
        output.save(output_path)
        print(f"Wrote {output_path}")

    for filename in CIRCUIT_SCREENSHOTS:
        with Image.open(SOURCE_DIR / filename) as image:
            output = replace_circuit_background(image)
        output_path = DIAGRAMS_DIR / filename
        output.save(output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
