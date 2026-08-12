"""Tests for the optional geomeTRIC geometry optimizer plugin."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import logging
import logging.config
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("geometric", reason="geomeTRIC not available")

from geometric.config import config_dir
from geometric.molecule import Molecule
from geometric.nifty import bohr2ang

from qdk_chemistry import algorithms, data
from qdk_chemistry.constants import BOHR_TO_ANGSTROM
from qdk_chemistry.plugins.geometric.geometry_optimizer import (
    _extract_coordinates,
    _preserve_root_logging,
    _QdkDerivativeEngine,
)


def _h2_structure():
    """Create the H2 test structure."""
    return data.Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1, 1])


def test_geometric_plugin_registration():
    """The geomeTRIC plugin registers a geometry optimizer when installed."""
    available = algorithms.available("geometry_optimizer")
    assert "geometric" in available
    assert not any(name.startswith("geometric_") for name in available)

    optimizer = algorithms.create("geometry_optimizer", "geometric")
    assert isinstance(optimizer, algorithms.GeometryOptimizer)
    assert optimizer.name() == "geometric"
    assert optimizer.settings().get("optimizer") == "tric"


def test_geometric_optimizer_settings():
    """The geomeTRIC optimizer separates shared and method-specific settings."""
    shared_settings = algorithms.GeometryOptimizerSettings()
    assert not shared_settings.has("convergence_energy")

    optimizer = algorithms.create("geometry_optimizer", "geometric")
    settings = optimizer.settings()

    assert isinstance(settings, algorithms.GeometryOptimizerSettings)
    derivative_ref = settings.get("derivative_calculator")
    assert derivative_ref.algorithm_type == "nuclear_derivative_calculator"
    assert derivative_ref.algorithm_name == "qdk_finite_difference"

    assert settings.get("transition_state") is False
    assert settings.get("optimizer") == "tric"
    assert not settings.has("algorithm")
    assert settings.get("compute_hessian") is False
    assert settings.get("max_iterations") == 300
    assert settings.get("convergence_energy") == pytest.approx(1.0e-6)
    assert settings.get("convergence_rms_gradient") == pytest.approx(3.0e-4)
    assert settings.get("convergence_max_gradient") == pytest.approx(4.5e-4)
    assert settings.get("convergence_rms_displacement") == pytest.approx(1.2e-3)
    assert settings.get("convergence_max_displacement") == pytest.approx(1.8e-3)

    with pytest.raises(ValueError, match="max_iterations.*out of allowed range"):
        settings.set("max_iterations", 0)
    with pytest.raises(ValueError, match="convergence_rms_gradient.*out of allowed range"):
        settings.set("convergence_rms_gradient", -1.0e-4)

    settings.set("optimizer", "dlc")
    settings.set("transition_state", True)
    settings.set("convergence_energy", 2.0e-6)
    settings.set("convergence_rms_gradient", 2.0e-4)
    settings.set("convergence_max_gradient", 3.0e-4)
    settings.set("convergence_rms_displacement", 8.0e-4)
    settings.set("convergence_max_displacement", 1.0e-3)
    options = optimizer._geometric_options()
    assert options["coordsys"] == "dlc"
    assert options["transition"] is True
    assert options["convergence_energy"] == pytest.approx(2.0e-6)
    assert options["convergence_grms"] == pytest.approx(2.0e-4)
    assert options["convergence_gmax"] == pytest.approx(3.0e-4)
    assert options["convergence_drms"] == pytest.approx(8.0e-4)
    assert options["convergence_dmax"] == pytest.approx(1.0e-3)
    assert optimizer.name() == "geometric"


def test_geometric_restores_host_logging_configuration(tmp_path):
    """Restore live host handlers and level after geomeTRIC configures logging."""
    host_log_path = tmp_path / "host.log"
    geometric_log_path = tmp_path / "geometric.log"
    host_handler = logging.FileHandler(host_log_path, mode="w")
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    root_logger.addHandler(host_handler)
    expected_handlers = [*original_handlers, host_handler]
    root_logger.setLevel(logging.ERROR)

    try:
        with _preserve_root_logging():
            logging.config.fileConfig(
                Path(config_dir) / "log.ini",
                defaults={"logfilename": str(geometric_log_path)},
                disable_existing_loggers=False,
            )

        assert root_logger.handlers == expected_handlers
        assert root_logger.level == logging.ERROR
        root_logger.error("host logging restored")
        host_handler.flush()
        assert host_log_path.read_text().splitlines() == ["host logging restored"]
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
        host_handler.close()


def test_geometric_engine_coordinates_are_bohr():
    """Keep geomeTRIC engine coordinates in the QDK structure's Bohr units."""
    structure = _h2_structure()
    molecule = Molecule()
    molecule.elem = ["H", "H"]
    molecule.xyzs = [np.zeros((2, 3))]
    engine = _QdkDerivativeEngine(structure, 0, 1, "sto-3g", 0, None, molecule)

    displaced_bohr = np.asarray(structure.get_coordinates()) + 0.25
    converted = engine._structure_from_bohr_coordinates(displaced_bohr)

    np.testing.assert_allclose(converted.get_coordinates(), displaced_bohr)
    np.testing.assert_allclose(engine.last_coordinates(), structure.get_coordinates())


def test_geometric_result_coordinates_match_engine_cache():
    """Preserve geomeTRIC's internal Bohr coordinates across its Angstrom result."""
    structure = _h2_structure()
    coordinates_bohr = np.asarray(structure.get_coordinates())
    molecule = Molecule()
    molecule.elem = ["H", "H"]
    molecule.xyzs = [coordinates_bohr * bohr2ang]
    engine = _QdkDerivativeEngine(structure, 0, 1, "sto-3g", 0, None, molecule)
    engine._last_energy = -1.0
    result = type("Result", (), {"xyzs": [coordinates_bohr * bohr2ang]})()

    converted = _extract_coordinates(result, engine)

    np.testing.assert_allclose(converted, coordinates_bohr, rtol=0.0, atol=1.0e-12)
    assert engine.cached_result(converted) == (-1.0, None)


def test_geometric_optimizer_smoke_run():
    """Optimize H2 to its STO-3G HF equilibrium bond length."""
    optimizer = algorithms.create("geometry_optimizer", "geometric")
    optimizer.settings().set("max_iterations", 20)
    derivative_ref = data.AlgorithmRef("nuclear_derivative_calculator", "qdk_finite_difference")
    derivative_ref.set("finite_difference_step", 1.0e-2)
    optimizer.settings().set("derivative_calculator", derivative_ref)

    energy, structure, hessian, wavefunction = optimizer.run(_h2_structure(), 0, 1, "sto-3g")

    assert np.isfinite(energy)
    assert structure.get_num_atoms() == 2
    assert structure.get_coordinates().shape == (2, 3)
    bond_length_bohr = np.linalg.norm(np.diff(structure.get_coordinates(), axis=0))
    bond_length_angstrom = bond_length_bohr * BOHR_TO_ANGSTROM
    assert bond_length_bohr == pytest.approx(1.346, abs=0.02)
    assert bond_length_angstrom == pytest.approx(0.712, abs=0.01)
    assert wavefunction is not None
    assert hessian is None
