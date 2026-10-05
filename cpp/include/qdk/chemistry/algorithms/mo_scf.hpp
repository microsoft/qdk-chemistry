// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/ansatz.hpp>

namespace qdk::chemistry::algorithms {

/**
 * @brief Hartree-Fock optimization from integrals in an orthonormal MO basis.
 *
 * Unlike ScfSolver, this algorithm requires no molecular geometry or atomic
 * basis. It optimizes the Hamiltonian's active orbitals, leaving inactive and
 * external orbitals fixed. Electron counts refer to the active space.
 *
 * The output Ansatz contains both the rotated Hamiltonian and its single
 * determinant, with occupied active orbitals ordered first in each spin
 * channel, so the operator and state remain in the same basis.
 */
class MoScfSolver
    : public Algorithm<
          MoScfSolver, std::pair<double, std::shared_ptr<data::Ansatz>>,
          std::shared_ptr<data::Hamiltonian>, unsigned int, unsigned int> {
 public:
  /**
   * @brief Optimize orbitals and transform the input Hamiltonian.
   *
   * \cond DOXYGEN_SUPRESS
   * @param hamiltonian Integrals in a common orthonormal spatial-orbital basis
   * @param n_active_alpha_electrons Number of active alpha electrons
   * @param n_active_beta_electrons Number of active beta electrons
   * \endcond
   * @return Total energy in Hartree and an Ansatz in the optimized basis
   * @throws std::invalid_argument For unsupported input or electron counts
   * @throws std::runtime_error If SCF fails to converge
   */
  using Algorithm::run;
  std::string type_name() const final { return "mo_scf_solver"; }

 protected:
  virtual std::pair<double, std::shared_ptr<data::Ansatz>> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_active_alpha_electrons,
      unsigned int n_active_beta_electrons) const = 0;
};

/** @brief Factory for integral-input SCF algorithms. */
struct MoScfSolverFactory
    : public AlgorithmFactory<MoScfSolver, MoScfSolverFactory> {
  static std::string algorithm_type_name() { return "mo_scf_solver"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk"; }
};

}  // namespace qdk::chemistry::algorithms
