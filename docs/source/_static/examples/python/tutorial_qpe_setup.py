"""Verify the Python environment and implementations required by the tutorial.

The script checks package compatibility, imports notebook/widget dependencies,
and instantiates every QDK/Chemistry algorithm implementation used later. It does
not run a chemistry calculation.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os

import qdk_chemistry


def public_version(version: str) -> str:
    """Remove PEP 440 local build metadata from a package version.

    Args:
        version: Installed version such as ``2.0.0`` or ``2.0.0+local``.

    Returns:
        The public release component, such as ``2.0.0``.
    """
    return version.partition("+")[0]


# CI sets this variable to verify the tutorial's declared release compatibility.
# Downloaded copies leave it unset, so ordinary student execution reports the
# installed version without embedding test-harness policy in the example.
GROUND_STATE_TUTORIAL_VERSION = os.getenv("GROUND_STATE_TUTORIAL_VERSION")
if GROUND_STATE_TUTORIAL_VERSION is not None:
    installed_public_version = public_version(qdk_chemistry.__version__)
    assert installed_public_version == GROUND_STATE_TUTORIAL_VERSION, (
        f"Tutorial expects QDK/Chemistry {GROUND_STATE_TUTORIAL_VERSION}, "
        f"but {qdk_chemistry.__version__} is installed."
    )

# start-cell-verify
import platform
import sys

import ipykernel
from qdk.widgets import MoleculeViewer
from qdk_chemistry.algorithms import create

# create() resolves each factory key and instantiates its implementation without
# running a calculation. Successful construction proves the required algorithms
# are registered and their import-time dependencies are available. Retaining the
# objects in a tuple also lets the final diagnostic report how many were checked.
required_implementations = (
    create("scf_solver", "qdk"),
    create("active_space_selector", "qdk_autocas_eos"),
    create("hamiltonian_constructor", "qdk"),
    create("multi_configuration_calculator", "macis_cas"),
    create("qubit_mapper", "qdk"),
    create("state_prep", "sparse_isometry_gf2x"),
    create("phase_estimation", "qdk_iterative"),
    create("circuit_executor", "qdk_full_state_simulator"),
)

print(f"Python executable: {sys.executable}")
print(f"Python version: {platform.python_version()}")
print(f"QDK/Chemistry version: {qdk_chemistry.__version__}")
print(f"IPython kernel version: {ipykernel.__version__}")
print(f"Verified widget: {MoleculeViewer.__name__}")
print(f"Verified {len(required_implementations)} built-in implementations.")
# end-cell-verify
