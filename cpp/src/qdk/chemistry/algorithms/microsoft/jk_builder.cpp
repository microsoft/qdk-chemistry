// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "jk_builder.hpp"

#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/eri.h>
#include <qdk/chemistry/scf/core/scf.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "eri/INCORE/incore.h"
#include "eri/LIBINT2_DIRECT/libint2_direct.h"
#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

JKEngine::JKEngine(std::shared_ptr<data::BasisSet> basis_set,
                   double eri_threshold, bool incore, bool use_atomics,
                   double shell_pair_threshold)
    : basis_set_(std::move(basis_set)),
      eri_threshold_(eri_threshold),
      incore_(incore),
      use_atomics_(use_atomics),
      shell_pair_threshold_(shell_pair_threshold) {}

JKEngine::~JKEngine() = default;

qcs::ERI& JKEngine::eri(size_t num_atomic_orbitals, size_t num_densities,
                        bool incore, double omega) {
  if (!scf_basis_set_) {
    scf_basis_set_ = utils::microsoft::convert_basis_set_from_qdk(*basis_set_);
  }
  if (num_atomic_orbitals != scf_basis_set_->num_atomic_orbitals ||
      (num_densities != 1 && num_densities != 2)) {
    throw std::invalid_argument(
        "JKEngine: expected one or two density matrices of " +
        std::to_string(scf_basis_set_->num_atomic_orbitals) +
        " atomic orbitals");
  }
  const auto type = num_densities == 1 ? qcs::SCFOrbitalType::Restricted
                                       : qcs::SCFOrbitalType::Unrestricted;
  auto& engine = (incore ? incore_eri_ : direct_)[num_densities - 1];
  // In-core integrals are computed for one range-separation parameter.
  if (incore && engine && omega != incore_omega_[num_densities - 1]) {
    engine.reset();
  }
  if (!engine) {
    if (incore) {
      engine = std::make_unique<qcs::ERIINCORE>(
          type, *scf_basis_set_, qcs::mpi_default_input(), omega);
      incore_omega_[num_densities - 1] = omega;
    } else {
      engine = std::make_unique<qcs::LIBINT2_DIRECT>(
          type, *scf_basis_set_, qcs::mpi_default_input(), use_atomics_,
          eri_threshold_, shell_pair_threshold_);
    }
  }
  return *engine;
}

void JKEngine::build(const double* P, size_t num_atomic_orbitals,
                     size_t num_densities, double* J, double* K, double alpha,
                     double beta, double omega) {
  eri(num_atomic_orbitals, num_densities, incore_, omega)
      .build_JK(P, J, K, alpha, beta, omega);
}

void JKEngine::build_gradients(const double* P, size_t num_atomic_orbitals,
                               size_t num_densities, size_t num_atoms,
                               double* dJ, double* dK, double alpha,
                               double beta, double omega) {
  const size_t expected_atoms = basis_set_->get_structure()->get_num_atoms();
  if (num_atoms != expected_atoms) {
    throw std::invalid_argument("JKEngine: expected gradients of " +
                                std::to_string(expected_atoms) + " atoms");
  }
  // In-core gradients are not implemented, so gradients are always direct.
  eri(num_atomic_orbitals, num_densities, false, omega)
      .get_gradients(P, dJ, dK, alpha, beta, omega);
}

std::shared_ptr<algorithms::JKEngine> JKBuilder::_run_impl(
    std::shared_ptr<data::BasisSet> basis_set, double eri_threshold) const {
  utils::microsoft::initialize_backend();
  return std::make_shared<JKEngine>(
      std::move(basis_set), eri_threshold,
      _settings->get<std::string>("eri_method") == "incore",
      _settings->get<bool>("eri_use_atomics"),
      _settings->get<double>("shell_pair_threshold"));
}

}  // namespace qdk::chemistry::algorithms::microsoft
