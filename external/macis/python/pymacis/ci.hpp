// Copyright (c) Microsoft Corporation.

#include "hamiltonian.hpp"

/**
 * @brief Run Complete Active Space Configuration Interaction (CASCI)
 * calculation
 * @param nalpha Number of alpha electrons
 * @param nbeta Number of beta electrons
 * @param ham Hamiltonian object containing molecular integrals
 * @param settings Python dictionary with calculation settings
 * @return Python dictionary containing energy, coefficients, and optionally
 * determinants
 */
py::dict run_casci(size_t nalpha, size_t nbeta, Hamiltonian &ham,
                   const py::dict &settings);

/**
 * @brief Run Adaptive Sampling Configuration Interaction (ASCI) calculation
 * @param initial_guess Python list of initial determinant strings
 * @param C0 Initial coefficients vector
 * @param E0 Initial energy estimate
 * @param ham Hamiltonian object containing molecular integrals
 * @param settings Python dictionary with calculation settings
 * @return Python dictionary containing energy, coefficients, and determinants
 */
py::dict run_asci(const py::list &initial_guess, const std::vector<double> &C0,
                  double E0, Hamiltonian &ham, const py::dict &settings);

/**
 * @brief Compute lowest eigenpair of a Hamiltonian projected into a particular
 * configuration basis
 * @param configurations Python list of determinant strings defining the basis
 * @param ham Hamiltonian object containing molecular integrals
 * @param settings Python dictionary with diagonalization calculation settings
 */
py::dict selected_ci_diag(const py::list &configurations, Hamiltonian &ham,
                          const py::dict &settings);

/**
 * @brief Register CI functions with pybind11 module
 * @param m PyBind11 module to which the functions will be added
 */
void export_ci_pybind(py::module &m);
