"""NetworkX plugin for QDK/Chemistry.

Provides improved graph-coloring-based term groupers using networkx's
DSATUR (saturation-largest-first) strategy, which typically produces
fewer groups than the built-in greedy first-fit algorithm.

When loaded, this plugin registers two additional ``term_grouper``
algorithms:

- ``"nx_commuting"`` — full Pauli commutation grouping via DSATUR
- ``"nx_qubit_wise_commuting"`` — qubit-wise commutation grouping via DSATUR

These are drop-in alternatives to the built-in ``"commuting"`` and
``"qubit_wise_commuting"`` groupers.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util

from qdk_chemistry.utils import Logger

_loaded = False
QDK_CHEMISTRY_HAS_NETWORKX = False


def load():
    """Load the NetworkX plugin into QDK/Chemistry."""
    Logger.trace_entering()
    global _loaded, QDK_CHEMISTRY_HAS_NETWORKX  # noqa: PLW0603
    if _loaded:
        return
    _loaded = True

    if importlib.util.find_spec("networkx") is not None:
        QDK_CHEMISTRY_HAS_NETWORKX = True
        _register_algorithms()


def _register_algorithms():
    """Register NetworkX-backed term grouper algorithms."""
    Logger.trace_entering()
    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.networkx.term_grouper import (  # noqa: PLC0415
        NxFullCommutingTermGrouper,
        NxQubitWiseCommutingTermGrouper,
    )

    register(lambda: NxFullCommutingTermGrouper())
    register(lambda: NxQubitWiseCommutingTermGrouper())
    Logger.debug(
        f"NetworkX plugin loaded: "
        f"[{NxFullCommutingTermGrouper().type_name()}: {NxFullCommutingTermGrouper().name()}], "
        f"[{NxQubitWiseCommutingTermGrouper().type_name()}: {NxQubitWiseCommutingTermGrouper().name()}]."
    )
