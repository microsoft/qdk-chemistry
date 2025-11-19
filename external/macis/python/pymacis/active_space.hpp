// Copyright (c) Microsoft Corporation.

#include "hamiltonian.hpp"

/**
 * @brief Compute active space Hamiltonian from full molecular Hamiltonian
 * @param nactive Number of active orbitals
 * @param ninactive Number of inactive orbitals
 * @param ham Input full Hamiltonian
 * @return Active space Hamiltonian with downfolded inactive contributions
 */
Hamiltonian compute_active_hamiltonian(size_t nactive, size_t ninactive,
                                       const Hamiltonian &ham);

/**
 * @brief Register active space functions with pybind11 module
 * @param m PyBind11 module to which the functions will be added
 */
void export_active_space_pybind(pybind11::module &m);
