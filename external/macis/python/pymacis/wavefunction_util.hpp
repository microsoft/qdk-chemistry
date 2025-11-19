// Copyright (c) Microsoft Corporation.

#pragma once

#include "hamiltonian.hpp"

/**
 * @brief Generate canonical Hartree-Fock determinant (Python wrapper)
 * @param nalpha Number of alpha electrons
 * @param nbeta Number of beta electrons
 * @param norb Number of orbitals
 * @return Binary string representing the canonical HF determinant
 */
std::string canonical_hf_determinant_wrapper(size_t nalpha, size_t nbeta,
                                             size_t norb);

/**
 * @brief Calculate energy from wavefunction determinants and coefficients
 * (Python wrapper)
 * @param dets_strings Python list of determinant strings
 * @param coeffs NumPy array of coefficients
 * @param ham Hamiltonian object containing molecular integrals
 * @return Calculated energy value
 */
double calculate_energy_wrapper(const py::list &dets_strings,
                                np_double_array coeffs, Hamiltonian &ham);

double diagonalize_wrapper(Hamiltonian &ham, const py::list &dets_strings);

void export_wavefunction_util_pybind(py::module &m);
