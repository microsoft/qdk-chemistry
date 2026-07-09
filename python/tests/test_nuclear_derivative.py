"""Tests for nuclear derivative data and algorithm bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry import algorithms, data


def _h2_structure():
    """Create an H2 structure for derivative data tests."""
    return data.Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1, 1])


def test_nuclear_gradients_roundtrip_json():
    """Round-trip nuclear gradients through JSON."""
    structure = _h2_structure()
    values = np.arange(6, dtype=float)

    gradients = data.NuclearGradients(structure, values)

    assert gradients.get_structure().get_num_atoms() == 2
    np.testing.assert_allclose(gradients.get_values(), values)
    assert gradients.as_matrix().shape == (2, 3)
    np.testing.assert_allclose(gradients.get_atom_gradient(1), [3.0, 4.0, 5.0])

    loaded = data.NuclearGradients.from_json(gradients.to_json())
    np.testing.assert_allclose(loaded.get_values(), values)
    np.testing.assert_allclose(
        loaded.get_structure().get_coordinates(),
        structure.get_coordinates(),
    )


def test_nuclear_hessian_roundtrip_json():
    """Round-trip a nuclear Hessian through JSON."""
    structure = _h2_structure()
    matrix = np.arange(36, dtype=float).reshape(6, 6)

    hessian = data.NuclearHessian(structure, matrix)

    assert hessian.get_structure().get_num_atoms() == 2
    np.testing.assert_allclose(hessian.get_matrix(), matrix)
    np.testing.assert_allclose(hessian.get_atom_pair_block(0, 1), matrix[:3, 3:])

    loaded = data.NuclearHessian.from_json(hessian.to_json())
    np.testing.assert_allclose(loaded.get_matrix(), matrix)
    np.testing.assert_allclose(
        loaded.get_structure().get_coordinates(),
        structure.get_coordinates(),
    )


def test_nuclear_derivative_factory_registered():
    """Create the default nuclear derivative calculator from the factory."""
    calculator = algorithms.create("nuclear_derivative_calculator")

    assert isinstance(calculator, algorithms.NuclearDerivativeCalculator)
    assert calculator.name() == "qdk_finite_difference"


def test_qdk_nuclear_derivative_factory_registered():
    """Create the QDK nuclear derivative calculator from the factory."""
    calculator = algorithms.create("nuclear_derivative_calculator", "qdk")

    assert isinstance(calculator, algorithms.QdkNuclearDerivativeCalculator)
    assert calculator.name() == "qdk"
