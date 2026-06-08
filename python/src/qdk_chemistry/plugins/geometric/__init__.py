"""geomeTRIC plugin for QDK/Chemistry geometry optimization."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util

from qdk_chemistry.utils import Logger

_loaded = False
QDK_CHEMISTRY_HAS_GEOMETRIC = False


def load():
    """Load the geomeTRIC plugin into QDK/Chemistry."""
    Logger.trace_entering()
    global _loaded, QDK_CHEMISTRY_HAS_GEOMETRIC  # noqa: PLW0603
    if _loaded:
        return
    _loaded = True

    if importlib.util.find_spec("geometric") is not None:
        QDK_CHEMISTRY_HAS_GEOMETRIC = True
        _register_algorithms()


def _register_algorithms():
    """Register geomeTRIC-backed optimizer algorithms."""
    Logger.trace_entering()
    from qdk_chemistry.algorithms import register  # noqa: PLC0415
    from qdk_chemistry.plugins.geometric.geometry_optimizer import GeometricOptimizer  # noqa: PLC0415

    register(lambda: GeometricOptimizer())
    Logger.debug(f"geomeTRIC plugin loaded: [{GeometricOptimizer().type_name()}: {GeometricOptimizer().name()}].")
