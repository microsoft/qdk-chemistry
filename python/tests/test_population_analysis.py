"""Tests for population-analysis algorithm bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry import algorithms, data


def _h2_structure():
    return data.Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1, 1])


def test_population_analyzer_factory_registered():
    """Create the default population analyzer from the factory."""
    analyzer = algorithms.create("population_analyzer")

    assert isinstance(analyzer, algorithms.PopulationAnalyzer)
    assert isinstance(analyzer, algorithms.QdkPopulationAnalyzer)
    assert analyzer.name() == "qdk"


def test_qdk_population_analyzer_structure_input():
    """The QDK analyzer accepts a structure input and returns one charge per atom."""
    analyzer = algorithms.create("population_analyzer", "qdk")
    analyzer.settings().set("charge", 1)

    charges = analyzer.run(_h2_structure())

    assert len(charges) == 2
    np.testing.assert_allclose(charges, [0.5, 0.5])
