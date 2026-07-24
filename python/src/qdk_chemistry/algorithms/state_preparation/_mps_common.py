"""Shared helpers for MPS state preparation algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass

from qdk_chemistry.data import Configuration, MPSContainer


def validate_mps_physical_basis(container: MPSContainer) -> None:
    """Require the physical-slice order assumed by the Q# operations."""
    canonical_basis = [Configuration.from_spin_half_string(state) for state in ("0", "u", "d", "2")]
    if container.physical_basis != canonical_basis:
        raise ValueError("MPS state preparation requires physical basis ordering ('0', 'u', 'd', '2').")


@dataclass
class GivensLayerData:
    """Result of decomposing a unitary into Givens rotation layers."""

    layer_angles: list[list[float]]
    layer_shifted: list[bool]
    phases: list[bool]
