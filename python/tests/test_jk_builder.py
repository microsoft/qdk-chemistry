"""Tests for the J/K builder algorithm type."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import JKBuilder, JKEngine
from qdk_chemistry.data import AlgorithmRef, Structure

from .reference_tolerances import float_comparison_absolute_tolerance, scf_energy_tolerance

WATER = """3

O    0.000000    0.000000    0.000000
H    0.758602    0.000000    0.504284
H   -0.758602    0.000000    0.504284
"""

O2 = """2

O    0.000000    0.000000    0.000000
O    0.000000    0.000000    1.208000
"""


def run_scf(xyz, multiplicity, jk_builder):
    """Run an HF calculation in sto-3g with a J/K builder."""
    solver = algorithms.create("scf_solver")
    solver.settings().set("jk_builder", AlgorithmRef("jk_builder", jk_builder))
    return solver.run(Structure.from_xyz(xyz), 0, multiplicity, "sto-3g")


class DelegatingJKEngine(JKEngine):
    """Engine building J and K with the engine of the qdk builder."""

    num_builds = 0

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def build(self, densities, coulomb, exchange, alpha, beta, omega):
        assert densities.ndim == 3
        assert not densities.flags.writeable
        self.engine.build(densities, coulomb, exchange, alpha, beta, omega)
        DelegatingJKEngine.num_builds += 1

    def build_gradients(self, densities, coulomb, exchange, alpha, beta, omega):
        assert coulomb.shape == exchange.shape == (3, 3)
        self.engine.build_gradients(densities, coulomb, exchange, alpha, beta, omega)


class DelegatingJKBuilder(JKBuilder):
    """Python J/K builder delegating to the qdk builder."""

    def name(self):
        return "test_python_delegating"

    def _run_impl(self, basis_set, eri_threshold):
        return DelegatingJKEngine(algorithms.create("jk_builder", "qdk").run(basis_set, eri_threshold))


def test_default_is_qdk():
    assert "qdk" in algorithms.available("jk_builder")
    assert algorithms.create("jk_builder").name() == "qdk"
    ref = algorithms.create("scf_solver").settings().get("jk_builder")
    assert ref.get("algorithm_name") == "qdk"


def water_engine():
    """Return the qdk engine and the HF/sto-3g density matrix of water."""
    _, wavefunction = run_scf(WATER, 1, "qdk")
    orbitals = wavefunction.get_orbitals()
    basis_set = orbitals.get_basis_set()
    occupations = np.zeros(orbitals.get_num_molecular_orbitals())
    occupations[:5] = 2.0
    density = np.ascontiguousarray(orbitals.calculate_ao_density_matrix(occupations))
    return algorithms.create("jk_builder").run(basis_set, 1e-12), density


def test_engine_builds():
    engine, density = water_engine()
    n = density.shape[0]

    coulomb, exchange = np.zeros((n, n)), np.zeros((n, n))
    engine.build(density, coulomb, exchange)
    assert np.allclose(coulomb, coulomb.T, atol=float_comparison_absolute_tolerance)
    assert 0.5 * np.sum(density * coulomb) > 0.0

    # Two density matrices: J and K are linear in the density.
    densities = np.stack([0.25 * density, 0.75 * density])
    coulombs, exchanges = np.zeros_like(densities), np.zeros_like(densities)
    engine.build(densities, coulombs, exchanges)
    assert np.allclose(coulombs.sum(axis=0), coulomb, atol=1e-10)
    assert np.allclose(exchanges.sum(axis=0), exchange, atol=1e-10)

    exchange_only = np.zeros((n, n))
    engine.build(density, None, exchange_only)
    assert np.allclose(exchange_only, exchange, atol=1e-12)

    with pytest.raises(ValueError, match="shape of densities"):
        engine.build(density, np.zeros((n, n), dtype=np.float32), None)
    with pytest.raises(ValueError, match="shape of densities"):
        engine.build(density, np.zeros((n, n)).T[:, : n - 1], None)
    with pytest.raises(ValueError, match="square"):
        engine.build(np.zeros((n, n + 1)), None, None)


def test_engine_gradients():
    engine, density = water_engine()
    coulomb, exchange = np.zeros((3, 3)), np.zeros((3, 3))
    engine.build_gradients(density, coulomb, exchange)
    # The energies are invariant under translations.
    assert np.abs(coulomb).max() > 0.1
    assert np.allclose(coulomb.sum(axis=1), 0.0, atol=1e-8)
    assert np.allclose(exchange.sum(axis=1), 0.0, atol=1e-8)

    exchange_only = np.zeros((3, 3))
    engine.build_gradients(density, None, exchange_only)
    assert np.allclose(exchange_only, exchange, atol=1e-12)

    with pytest.raises(ValueError, match="3 atoms"):
        engine.build_gradients(density, np.zeros((3, 2)), None)
    with pytest.raises(ValueError, match=r"\(3, num_atoms\)"):
        engine.build_gradients(density, np.zeros((3, 3)), np.zeros((3, 2)))

    class BuildOnlyEngine(JKEngine):
        def build(self, densities, coulomb, exchange, alpha, beta, omega):
            pass

    with pytest.raises(RuntimeError, match="does not build gradients"):
        BuildOnlyEngine().build_gradients(density, coulomb, None)


def test_python_builder_in_scf():
    algorithms.register(lambda: DelegatingJKBuilder())
    try:
        DelegatingJKEngine.num_builds = 0
        for xyz, multiplicity in ((WATER, 1), (O2, 3)):
            energy, _ = run_scf(xyz, multiplicity, "test_python_delegating")
            reference, _ = run_scf(xyz, multiplicity, "qdk")
            assert energy == pytest.approx(reference, abs=scf_energy_tolerance)
        assert DelegatingJKEngine.num_builds > 0
    finally:
        algorithms.unregister("jk_builder", "test_python_delegating")


def test_python_builder_gradients():
    def gradients(jk_builder):
        calculator = algorithms.create("nuclear_derivative_calculator", "qdk")
        scf_solver = AlgorithmRef("scf_solver", "qdk", jk_builder=AlgorithmRef("jk_builder", jk_builder))
        calculator.settings().set("energy_calculator", scf_solver)
        _, result, _, _ = calculator.run(Structure.from_xyz(WATER), 0, 1, "sto-3g")
        return result.get_values()

    algorithms.register(lambda: DelegatingJKBuilder())
    try:
        assert np.allclose(gradients("test_python_delegating"), gradients("qdk"), atol=1e-6)
    finally:
        algorithms.unregister("jk_builder", "test_python_delegating")
