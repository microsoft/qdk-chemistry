"""Tests for the optional geomeTRIC geometry optimizer plugin."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import logging

import numpy as np
import pytest

pytest.importorskip("geometric", reason="geomeTRIC not available")

from qdk_chemistry import algorithms, data
from qdk_chemistry.plugins.geometric.geometry_optimizer import (
    GEOMETRIC_OPTIMIZER_ALGORITHMS,
    _close_geometric_log_handler,
)


def _h2_structure():
    """Create the H2 test structure."""
    return data.Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1, 1])


def test_geometric_plugin_registration():
    """The geomeTRIC plugin registers a geometry optimizer when installed."""
    available = algorithms.available("geometry_optimizer")
    assert "geometric" in available
    for algorithm in GEOMETRIC_OPTIMIZER_ALGORITHMS:
        assert f"geometric_geoopt_{algorithm}" in available
        assert f"geometric_tsopt_{algorithm}" in available

    default_optimizer = algorithms.create("geometry_optimizer", "geometric")
    assert isinstance(default_optimizer, algorithms.GeometryOptimizer)
    assert default_optimizer.name() == "geometric"
    assert default_optimizer.settings().get("coordinate_system") == "tric"

    optimizer = algorithms.create("geometry_optimizer", "geometric_geoopt_tric")
    assert isinstance(optimizer, algorithms.GeometryOptimizer)
    assert optimizer.name() == "geometric_geoopt_tric"
    assert optimizer.settings().get("coordinate_system") == "tric"

    ts_optimizer = algorithms.create("geometry_optimizer", "geometric_tsopt_dlc")
    assert isinstance(ts_optimizer, algorithms.GeometryOptimizer)
    assert ts_optimizer.name() == "geometric_tsopt_dlc"
    assert ts_optimizer.settings().get("coordinate_system") == "dlc"
    assert ts_optimizer.settings().get("transition_state") is True


def test_geometric_optimizer_settings():
    """The geomeTRIC optimizer exposes shared optimizer settings."""
    optimizer = algorithms.create("geometry_optimizer", "geometric_geoopt_cartesian")
    settings = optimizer.settings()

    derivative_ref = settings.get("derivative_calculator")
    assert derivative_ref.algorithm_type == "nuclear_derivative_calculator"
    assert derivative_ref.algorithm_name == "qdk_finite_difference"

    assert settings.get("transition_state") is False
    assert settings.get("coordinate_system") == "cart"
    assert settings.get("compute_hessian") is False
    assert settings.get("max_iterations") == 300
    assert settings.get("convergence_energy") == pytest.approx(1.0e-6)
    assert settings.get("convergence_gradient") == pytest.approx(3.0e-4)
    assert settings.get("convergence_displacement") == pytest.approx(1.2e-3)


def test_geometric_log_handler_is_closed_before_temp_directory_cleanup(tmp_path):
    """Close the temporary geomeTRIC log so Windows can delete it."""
    log_path = tmp_path / "qdk-chemistry.log"
    handler = logging.FileHandler(log_path)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        _close_geometric_log_handler(log_path)

        assert handler not in root_logger.handlers
        assert handler.stream is None
        log_path.unlink()
    finally:
        root_logger.removeHandler(handler)
        handler.close()


def test_geometric_optimizer_smoke_run():
    """Run a small geometry optimization through the optional geomeTRIC backend."""
    optimizer = algorithms.create("geometry_optimizer", "geometric_geoopt_tric")
    optimizer.settings().set("max_iterations", 20)
    derivative_ref = data.AlgorithmRef("nuclear_derivative_calculator", "qdk_finite_difference")
    derivative_ref.set("finite_difference_step", 1.0e-2)
    optimizer.settings().set("derivative_calculator", derivative_ref)

    energy, structure, wavefunction, hessian = optimizer.run(_h2_structure(), 0, 1, "sto-3g")

    assert np.isfinite(energy)
    assert structure.get_num_atoms() == 2
    assert structure.get_coordinates().shape == (2, 3)
    assert wavefunction is not None
    assert hessian is None
