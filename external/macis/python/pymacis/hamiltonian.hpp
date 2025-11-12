// Copyright (c) Microsoft Corporation.

#pragma once

#include "common.hpp"

/**
 * @brief Hamiltonian structure for Python exposure
 *
 * Contains molecular integrals and associated metadata for quantum chemistry
 * calculations. Stores 1-body (T) and 2-body (V) integrals along with inactive
 * Fock matrix contributions.
 */
struct Hamiltonian {
  std::vector<double> _T;  ///< 1-body integrals (kinetic + nuclear attraction)
  std::vector<double> _V;  ///< 2-body integrals (electron-electron repulsion)
  std::vector<double> _F_inactive;  ///< Inactive Fock matrix contributions
  size_t norb = 0;                  ///< Number of orbitals
  size_t nbasis = 0;                ///< Number of basis functions
  double core_energy = 0.0;         ///< Core energy contribution
};

void export_hamiltonian_pybind(py::module &m);
