// Copyright (c) Microsoft Corporation.

#pragma once

#include <macis/util/fcidump.hpp>

#include "hamiltonian.hpp"

/**
 * @brief Read FCIDUMP file and return a Hamiltonian object
 * @param filename Path to FCIDUMP file
 * @return Hamiltonian object containing molecular integrals
 */
Hamiltonian read_fcidump(const std::string& filename);

/**
 * @brief Extract only the one-body integrals from a FCIDUMP file
 * @param filename Path to FCIDUMP file
 * @return NumPy array of one-body integrals
 */
py::array_t<double> read_fcidump_1body(const std::string& filename);

/**
 * @brief Extract only the two-body integrals from a FCIDUMP file
 * @param filename Path to FCIDUMP file
 * @return NumPy array of two-body integrals
 */
py::array_t<double> read_fcidump_2body(const std::string& filename);

/**
 * @brief Extract only the core energy from a FCIDUMP file
 * @param filename Path to FCIDUMP file
 * @return Core energy value
 */
double read_fcidump_core_energy(const std::string& filename);

/**
 * @brief Extract only the number of orbitals from a FCIDUMP file
 * @param filename Path to FCIDUMP file
 * @return Number of orbitals
 */
size_t read_fcidump_norb(const std::string& filename);

/**
 * @brief Read only the header information from a FCIDUMP file
 * @param filename Path to FCIDUMP file
 * @return FCIDumpHeader struct containing system parameters
 */
macis::FCIDumpHeader read_fcidump_header(const std::string& filename);

/**
 * @brief Write molecular integrals to a FCIDUMP file
 * @param filename Output FCIDUMP filename
 * @param header Header containing system parameters (electrons, orbitals,
 * symmetry)
 * @param T One-body integrals array
 * @param V Two-body integrals array
 * @param core_energy Nuclear repulsion energy plus frozen core contributions
 * @param threshold Value threshold for writing integrals (elements with
 * magnitude < threshold are omitted)
 */
void write_fcidump(const std::string& filename,
                   const macis::FCIDumpHeader& header,
                   const py::array_t<double>& T, const py::array_t<double>& V,
                   double core_energy, double threshold);

/**
 * @brief Register FCIDUMP-related functions with pybind11 module
 * @param m PyBind11 module to which the functions will be added
 */
void export_fcidump_pybind(py::module& m);
