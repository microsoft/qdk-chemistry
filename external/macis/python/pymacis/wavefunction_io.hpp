// Copyright (c) Microsoft Corporation.

#pragma once

#include "common.hpp"

/**
 * @brief Write wavefunction to file (Python wrapper)
 * @param filename Output filename
 * @param norb Number of orbitals
 * @param dets_strings Python list of determinant strings
 * @param coeffs NumPy array of coefficients
 */
void write_wavefunction_wrapper(const std::string &filename, size_t norb,
                                const py::list &dets_strings,
                                const np_double_array &coeffs);

/**
 * @brief Read wavefunction from file (Python wrapper)
 * @param filename Input filename
 * @return Python dictionary containing norbitals, determinants, and
 * coefficients
 */
py::dict read_wavefunction_wrapper(const std::string &filename);

void export_wavefunction_io_pybind(py::module &m);
