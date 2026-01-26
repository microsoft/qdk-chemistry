"""QDK/Chemistry-Qiskit Bindings."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util
import warnings

from qdk_chemistry.utils import Logger

# Suppress deprecation warnings from Qiskit and Aer dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit_aer.*")

_loaded = False


def load():
    """Load the Qiskit related plugins into QDK/Chemistry."""
    Logger.trace_entering()
    global _loaded  # noqa: PLW0603
    if _loaded:
        return
    _loaded = True
    if importlib.util.find_spec("qiskit") is not None:
        qiskit_load()
    if importlib.util.find_spec("qiskit_nature") is not None:
        qiskit_nature_load()
    if importlib.util.find_spec("qiskit_aer") is not None:
        qiskit_aer_load()


def qiskit_load():
    """Load the Qiskit plugins into QDK/Chemistry."""
    Logger.trace_entering()

    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.qiskit.regular_isometry import RegularIsometryStatePreparation  # noqa: PLC0415
    from qdk_chemistry.plugins.qiskit.standard_phase_estimation import QiskitStandardPhaseEstimation  # noqa: PLC0415

    register(lambda: RegularIsometryStatePreparation())
    register(lambda: QiskitStandardPhaseEstimation())

    Logger.info(
        "Qiskit plugins loaded: "
        f"[{RegularIsometryStatePreparation().type_name()}: {RegularIsometryStatePreparation().name()}], "
        f"[{QiskitStandardPhaseEstimation().type_name()}: {QiskitStandardPhaseEstimation().name()}]."
    )


def qiskit_nature_load():
    """Load the Qiskit Nature plugin into QDK/Chemistry."""
    Logger.trace_entering()

    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.qiskit.qubit_mapper import QiskitQubitMapper  # noqa: PLC0415

    register(lambda: QiskitQubitMapper())
    Logger.info(f"Qiskit Nature plugin loaded: [{QiskitQubitMapper().type_name()}: {QiskitQubitMapper().name()}].")


def qiskit_aer_load():
    """Load the Qiskit Aer plugin into QDK/Chemistry."""
    Logger.trace_entering()

    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.qiskit.circuit_executor import QiskitAerSimulator  # noqa: PLC0415
    from qdk_chemistry.plugins.qiskit.energy_estimator import QiskitEnergyEstimator  # noqa: PLC0415

    register(lambda: QiskitEnergyEstimator())
    register(lambda: QiskitAerSimulator())
    Logger.info(
        f"Qiskit Aer plugins loaded: "
        f"[{QiskitAerSimulator().type_name()}: {QiskitAerSimulator().name()}], "
        f"[{QiskitEnergyEstimator().type_name()}: {QiskitEnergyEstimator().name()}]."
    )
