"""Documented reference tolerances for various test comparisons."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# This is the SCF energy convergence tolerance used for the reference values.
# Consistency of the tested modules can only be expected to agree up to this tolerance.
scf_energy_tolerance = 1e-8

# This is the SCF orbital convergence tolerance used for the reference values.
# Consistency of the tested modules can only be expected to agree up to this tolerance.
scf_orbital_tolerance = 1e-6

# This is the tolerance for comparing outputs related to the plain text input.
# ASCII reference values are not expected to exceed this precision.
plain_text_tolerance = 1e-8

# This is the tolerance for comparing structures related to xyz file inputs.
# Consistency of the tested modules can only be expected to agree up to this tolerance.
xyz_file_structure_tolerance = 1e-6

# This is the coupled cluster convergence tolerance used for the reference values.
# Consistency of the tested modules can only be expected to agree up to this tolerance.
cc_tolerance = 1e-6

# This is the CI energy convergence tolerance used for the reference values.
# Consistency of the tested modules can only be expected to agree up to this tolerance.
ci_energy_tolerance = 1e-8

# This is the MCSCF energy convergence tolerance used for the reference values
# Consistency of the tested modules can only be expected to agree up to this tolerance.
mcscf_energy_tolerance = 1e-8

# Energy tolerance for estimating expectation values from a limited number of shots.
# The goal is to achieve energy estimates within chemical accuracy (~1e-3 Hartree) of the exact value.
estimator_energy_tolerance = 1e-3

# Relative tolerance parameter for comparing calculated floats in tests using np.isclose and np.allclose.
# Floats that are not calculated values should be compared using '=='.
float_comparison_relative_tolerance = 1e-12

# Absolute tolerance parameter for comparing calculated floats in tests using np.isclose and np.allclose.
# Floats that are not calculated values should be compared using '=='.
float_comparison_absolute_tolerance = 1e-12

# Tolerance for imaginary parts of Pauli operator coefficients in time evolution.
# Coefficients with imaginary parts below this threshold are considered real.
pauli_coefficient_imaginary_tolerance = 1e-8

# Tolerance for phase fraction comparisons in quantum phase estimation.
# Phase fractions from QPE algorithms are expected to agree within this tolerance.
qpe_phase_fraction_tolerance = 1e-6

# Tolerance for energy comparisons in quantum phase estimation.
# Energies derived from phase estimation are expected to agree within this tolerance.
qpe_energy_tolerance = 1e-6
