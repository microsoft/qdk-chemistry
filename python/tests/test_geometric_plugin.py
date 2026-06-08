"""Tests for the optional geomeTRIC geometry optimizer plugin."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms, data
from qdk_chemistry.plugins.geometric import QDK_CHEMISTRY_HAS_GEOMETRIC

pytestmark = pytest.mark.skipif(not QDK_CHEMISTRY_HAS_GEOMETRIC, reason="geomeTRIC not available")


def _h2_structure():
    return data.Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1, 1])


def test_geometric_plugin_registration():
    """The geomeTRIC plugin registers a geometry optimizer when installed."""
    assert "geometric" in algorithms.available("geometry_optimizer")
    optimizer = algorithms.create("geometry_optimizer", "geometric")
    assert isinstance(optimizer, algorithms.GeometryOptimizer)
    assert optimizer.name() == "geometric"


def test_geometric_optimizer_settings():
    """The geomeTRIC optimizer exposes shared optimizer settings."""
    optimizer = algorithms.create("geometry_optimizer", "geometric")
    settings = optimizer.settings()

    assert settings.get("transition_state") is False
    assert settings.get("compute_hessian") is False
    assert settings.get("max_iterations") == 300


def test_geometric_optimizer_smoke_run():
    """Run a small geometry optimization through the optional geomeTRIC backend."""
    optimizer = algorithms.create("geometry_optimizer", "geometric")
    optimizer.settings().set("max_iterations", 2)
    derivative_ref = data.AlgorithmRef("nuclear_derivative_calculator", "finite_difference")
    derivative_ref.get_settings().set("finite_difference_step", 1.0e-2)
    optimizer.settings().set("derivative_calculator", derivative_ref)

    energy, structure, wavefunction, hessian = optimizer.run(_h2_structure(), 0, 1, "sto-3g")

    assert np.isfinite(energy)
    assert structure.get_num_atoms() == 2
    assert structure.get_coordinates().shape == (2, 3)
    assert wavefunction is not None
    assert hessian is None
