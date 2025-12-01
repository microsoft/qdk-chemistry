"""QDK/Chemistry-Qiskit Bindings."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings

# Suppress deprecation warnings from Qiskit and Aer dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit_aer.*")


def load():
    """Load the Qiskit plugin into QDK/Chemistry."""
    import qdk_chemistry.plugins.qiskit.energy_estimator  # noqa: PLC0415
    import qdk_chemistry.plugins.qiskit.qubit_mapper  # noqa: PLC0415
    import qdk_chemistry.plugins.qiskit.regular_isometry  # noqa: PLC0415
