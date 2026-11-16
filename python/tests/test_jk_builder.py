"""Tests for the J/K builder algorithm types."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import JKBuilder, JKGradientBuilder
from qdk_chemistry.data import AlgorithmRef, Structure
from qdk_chemistry.data.symmetry import SymmetryLabel, axes, make_spin_diagonal_rank2_sbt

from .reference_tolerances import scf_energy_tolerance

WATER = """3

O    0.000000    0.000000    0.000000
H    0.758602    0.000000    0.504284
H   -0.758602    0.000000    0.504284
"""

O2 = """2

O    0.000000    0.000000    0.000000
O    0.000000    0.000000    1.208000
"""

ALPHA = SymmetryLabel([axes.alpha()])
BETA = SymmetryLabel([axes.beta()])


def run_scf(xyz, multiplicity, jk_builder="qdk"):
    """Run an HF calculation in sto-3g with a J/K builder."""
    solver = algorithms.create("scf_solver")
    solver.settings().set("jk_builder", AlgorithmRef("jk_builder", jk_builder))
    return solver.run(Structure.from_xyz(xyz), 0, multiplicity, "sto-3g")


def qdk_settings(builder):
    """The settings of a J/K builder shared by the qdk builders."""
    return {key: builder.settings().get(key) for key in ("eri_threshold", "alpha", "beta", "omega")}


class DelegatingJKBuilder(JKBuilder):
    """Python J/K builder delegating, with its settings, to the qdk builder."""

    num_builds = 0

    def name(self):
        """Return the registered name of the builder."""
        return "test_python_delegating"

    def _run_impl(self, basis_set, densities):
        """Build J and K with the qdk builder."""
        DelegatingJKBuilder.num_builds += 1
        return algorithms.create("jk_builder", "qdk", **qdk_settings(self)).run(basis_set, densities)


class DelegatingJKGradientBuilder(JKGradientBuilder):
    """Python J/K gradient builder delegating, with its settings, to the qdk builder."""

    def name(self):
        """Return the registered name of the builder."""
        return "test_python_delegating"

    def _run_impl(self, basis_set, densities):
        """Build the gradients with the qdk builder."""
        return algorithms.create("jk_gradient_builder", "qdk", **qdk_settings(self)).run(basis_set, densities)


@pytest.fixture(scope="module")
def water():
    """Basis set and alpha density of HF/sto-3g water."""
    _, wavefunction = run_scf(WATER, 1)
    orbitals = wavefunction.get_orbitals()
    occupations = np.zeros(orbitals.get_num_molecular_orbitals())
    occupations[:5] = 1.0
    return orbitals.get_basis_set(), orbitals.calculate_ao_density_matrix(occupations)


def test_defaults():
    """The qdk builders are the defaults, also of the SCF solver."""
    assert algorithms.create("jk_builder").name() == "qdk"
    assert algorithms.create("jk_gradient_builder").name() == "qdk"
    settings = algorithms.create("scf_solver").settings()
    assert settings.get("jk_builder").get("algorithm_name") == "qdk"
    assert settings.get("jk_gradient_builder").get("algorithm_name") == "qdk"


def test_builds(water):
    """J and K are built block by block in the spin structure of the densities."""
    basis_set, density = water
    coulomb, exchange = algorithms.create("jk_builder").run(basis_set, make_spin_diagonal_rank2_sbt(density))
    j = coulomb.block([ALPHA, ALPHA])
    assert np.allclose(j, j.T)
    assert np.sum(density * j) > 0.0

    # Each spin block is built from its own density.
    densities = make_spin_diagonal_rank2_sbt(density, np.zeros_like(density))
    coulomb_u, exchange_u = algorithms.create("jk_builder").run(basis_set, densities)
    assert np.allclose(coulomb_u.block([ALPHA, ALPHA]), j, atol=1e-10)
    assert np.allclose(exchange_u.block([ALPHA, ALPHA]), exchange.block([ALPHA, ALPHA]), atol=1e-10)
    assert np.allclose(coulomb_u.block([BETA, BETA]), 0.0, atol=1e-12)

    with pytest.raises(ValueError, match="atomic orbitals"):
        algorithms.create("jk_builder").run(basis_set, make_spin_diagonal_rank2_sbt(np.eye(2)))


def test_gradients(water):
    """The gradients of the Coulomb and exchange energies are translation invariant."""
    basis_set, density = water
    coulomb, exchange = algorithms.create("jk_gradient_builder").run(basis_set, make_spin_diagonal_rank2_sbt(density))
    assert np.abs(coulomb.get_values()).max() > 0.1
    # The energies are invariant under translations.
    for gradient in (coulomb, exchange):
        assert np.allclose(np.reshape(gradient.get_values(), (-1, 3)).sum(axis=0), 0.0, atol=1e-8)


def test_python_builder_in_scf():
    """A Python J/K builder reproduces the SCF energies of the qdk builder."""
    algorithms.register(lambda: DelegatingJKBuilder())
    try:
        DelegatingJKBuilder.num_builds = 0
        for xyz, multiplicity in ((WATER, 1), (O2, 3)):
            energy, _ = run_scf(xyz, multiplicity, "test_python_delegating")
            reference, _ = run_scf(xyz, multiplicity)
            assert energy == pytest.approx(reference, abs=scf_energy_tolerance)
        assert DelegatingJKBuilder.num_builds > 0
    finally:
        algorithms.unregister("jk_builder", "test_python_delegating")


def test_python_gradient_builder():
    """A Python J/K gradient builder reproduces the analytic gradients of the qdk builder."""

    def gradients(jk_gradient_builder):
        """Return the analytic HF/sto-3g gradient of water."""
        calculator = algorithms.create("nuclear_derivative_calculator", "qdk")
        scf_solver = AlgorithmRef(
            "scf_solver", "qdk", jk_gradient_builder=AlgorithmRef("jk_gradient_builder", jk_gradient_builder)
        )
        calculator.settings().set("energy_calculator", scf_solver)
        _, result, _, _ = calculator.run(Structure.from_xyz(WATER), 0, 1, "sto-3g")
        return result.get_values()

    algorithms.register(lambda: DelegatingJKGradientBuilder())
    try:
        assert np.allclose(gradients("test_python_delegating"), gradients("qdk"), atol=1e-6)
    finally:
        algorithms.unregister("jk_gradient_builder", "test_python_delegating")
