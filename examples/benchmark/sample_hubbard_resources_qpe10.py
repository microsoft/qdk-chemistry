"""Sample Hubbard resource estimates at a fixed 10-bit IQPE resolution.

This reuses the pipeline in :mod:`sample_hubbard_resources` and changes only how QPE is
sized: the resolution is pinned to 10 bits instead of being derived from the target
energy accuracy, and the plaquette Trotter step count is pinned explicitly rather than
resolved from a ``target_accuracy``.

Examples:
    Sample 120 x 120 and 130 x 130 lattices into separate CSVs::

        python sample_hubbard_resources_qpe10.py --size 120 130 -o qpe10_L{L}.csv

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import sample_hubbard_resources as base

#: Fixed IQPE resolution, replacing the value derived from the energy budget.
NUM_BITS = 10

#: Plaquette Trotter steps per evolution. The accuracy-driven value already resolves to
#: the builder's floor of 1 at every lattice size, so it is pinned here rather than halved.
NUM_DIVISIONS = 1


def qpe_parameters(
    one_norm: float,
    energy_budget: float,
) -> tuple[float, int, dict[str, float | int]]:
    """Size QPE at a fixed bit count and pin the plaquette builder's step count.

    Args:
        one_norm: Hamiltonian coefficient one-norm.
        energy_budget: Ground-state energy accuracy. Unused; kept for signature parity.

    Returns:
        Base evolution time, the fixed resolution bits, and builder settings.

    """
    base_time = math.pi / one_norm / 2
    # target_accuracy must be disabled, the builder otherwise takes max(manual, automatic).
    return base_time, NUM_BITS, {"num_divisions": NUM_DIVISIONS, "target_accuracy": 0.0}


# run_sampling resolves this name as a module global at call time.
base.qpe_parameters = qpe_parameters


if __name__ == "__main__":
    raise SystemExit(base.main())
