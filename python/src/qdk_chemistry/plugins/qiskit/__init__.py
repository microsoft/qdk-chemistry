"""QDK/Chemistry-Qiskit Bindings."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings

# Suppress deprecation warnings from Qiskit and Aer dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit_aer.*")

_loaded = False


def load():
    """Load the Qiskit plugin into QDK/Chemistry."""
    global _loaded  # noqa: PLW0603
    if _loaded:
        return
    _loaded = True

    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.qiskit.circuit_executor import QiskitAerSimulator  # noqa: PLC0415
    from qdk_chemistry.plugins.qiskit.energy_estimator import QiskitEnergyEstimator  # noqa: PLC0415
    from qdk_chemistry.plugins.qiskit.qubit_mapper import QiskitQubitMapper  # noqa: PLC0415
    from qdk_chemistry.plugins.qiskit.regular_isometry import RegularIsometryStatePreparation  # noqa: PLC0415
    from qdk_chemistry.plugins.qiskit.standard_phase_estimation import QiskitStandardPhaseEstimation  # noqa: PLC0415

    register(lambda: QiskitEnergyEstimator())
    register(lambda: QiskitQubitMapper())
    register(lambda: RegularIsometryStatePreparation())
    register(lambda: QiskitAerSimulator())
    register(lambda: QiskitStandardPhaseEstimation())
