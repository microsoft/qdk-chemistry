"""QDK/Chemistry-Azure Quantum Bindings."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util

from qdk_chemistry.utils import Logger

_loaded = False
QDK_CHEMISTRY_HAS_AZURE_QUANTUM = False


def load():
    """Load the Azure Quantum plugins into QDK/Chemistry."""
    Logger.trace_entering()
    global _loaded  # noqa: PLW0603
    global QDK_CHEMISTRY_HAS_AZURE_QUANTUM  # noqa: PLW0603
    if _loaded:
        return
    _loaded = True

    if importlib.util.find_spec("azure.quantum") is not None and importlib.util.find_spec("azure.identity") is not None:
        QDK_CHEMISTRY_HAS_AZURE_QUANTUM = True

    if QDK_CHEMISTRY_HAS_AZURE_QUANTUM:
        azure_quantum_load()


def azure_quantum_load():
    """Load the Azure Quantum emulator circuit executor into QDK/Chemistry."""
    Logger.trace_entering()

    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.azure_quantum.circuit_executor import AzureQuantumEmulator  # noqa: PLC0415

    register(lambda: AzureQuantumEmulator())
    Logger.debug(
        f"Azure Quantum plugins loaded: [{AzureQuantumEmulator().type_name()}: {AzureQuantumEmulator().name()}]."
    )
