"""Convert a white-backed PNG to transparency without changing its white composite."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def white_to_alpha(image: Image.Image) -> Image.Image:
    """Return an RGBA image that reproduces ``image`` exactly over white."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    alpha = (255 - rgb.min(axis=2)).astype(np.uint8)
    alpha_fraction = alpha.astype(np.float32) / 255.0
    foreground = np.full_like(rgb, 255)

    for channel in range(3):
        values = rgb[:, :, channel].astype(np.float32)
        unmixed = 255.0 + (values - 255.0) / np.maximum(alpha_fraction, 1.0 / 255.0)
        foreground[:, :, channel] = np.clip(np.rint(unmixed), 0, 255).astype(np.uint8)

    return Image.fromarray(np.dstack((foreground, alpha)), mode="RGBA")


def main() -> None:
    """Convert one opaque source PNG and write its transparent equivalent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Opaque source PNG")
    parser.add_argument("output", type=Path, help="Transparent output PNG")
    arguments = parser.parse_args()

    with Image.open(arguments.input) as image:
        converted = white_to_alpha(image)
    converted.save(arguments.output)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
