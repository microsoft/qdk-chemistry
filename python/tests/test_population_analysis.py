"""Tests for population-analysis algorithm bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry import algorithms, data


def _h2_structure():
    return data.Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1, 1])


def _model_wavefunction():
    orbitals = data.ModelOrbitals(3)
    determinant = data.Configuration.from_bitstring("110")
    return data.Wavefunction(data.StateVectorContainer(determinant, orbitals))


def test_population_analyzer_factory_registered():
    """Create the default population analyzer from the factory."""
    analyzer = algorithms.create("population_analyzer")

    assert isinstance(analyzer, algorithms.PopulationAnalyzer)
    assert isinstance(analyzer, algorithms.QdkPopulationAnalyzer)
    assert analyzer.name() == "qdk"


def test_qdk_population_analyzer_structure_input():
    """The QDK analyzer rejects structures without an electronic state."""
    analyzer = algorithms.create("population_analyzer", "qdk")

    with np.testing.assert_raises(ValueError):
        analyzer.run(_h2_structure(), charge=1, spin_multiplicity=1)


def test_qdk_population_analyzer_model_wavefunction():
    """The QDK analyzer returns particle counts for model-system sites."""
    analyzer = algorithms.create("population_analyzer", "qdk")

    populations = analyzer.run(_model_wavefunction(), charge=0, spin_multiplicity=1)

    np.testing.assert_allclose(populations, [1.0, 1.0, 0.0])
