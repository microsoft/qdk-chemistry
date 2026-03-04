"""QDK/Chemistry-OpenFermion Bindings."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util

from qdk_chemistry.utils import Logger

_loaded = False
QDK_CHEMISTRY_HAS_OPENFERMION = False


def load():
    """Load the OpenFermion related plugins into QDK/Chemistry."""
    Logger.trace_entering()
    global _loaded  # noqa: PLW0603
    global QDK_CHEMISTRY_HAS_OPENFERMION  # noqa: PLW0603
    if _loaded:
        return
    _loaded = True
    if importlib.util.find_spec("openfermion") is not None:
        QDK_CHEMISTRY_HAS_OPENFERMION = True
        openfermion_load()


def openfermion_load():
    """Load the OpenFermion qubit mapper plugin into QDK/Chemistry."""
    Logger.trace_entering()

    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.openfermion.qubit_mapper import OpenFermionQubitMapper  # noqa: PLC0415

    register(lambda: OpenFermionQubitMapper())
    Logger.debug(
        f"OpenFermion plugin loaded: [{OpenFermionQubitMapper().type_name()}: {OpenFermionQubitMapper().name()}]."
    )
