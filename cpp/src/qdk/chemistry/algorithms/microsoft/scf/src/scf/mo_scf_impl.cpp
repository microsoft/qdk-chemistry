// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "scf/mo_scf_impl.h"

#include <cmath>
#include <stdexcept>

namespace qdk::chemistry::scf {

MoSCFImpl::MoSCFImpl(const RowMajorMatrix& one_body_integrals,
                     std::shared_ptr<ERI> eri, int nalpha, int nbeta,
                     double core_energy, const SCFConfig& cfg)
    : SCFImpl(cfg, one_body_integrals.rows(), nalpha, nbeta),
      one_body_integrals_(one_body_integrals),
      core_energy_(core_energy) {
  if (one_body_integrals.rows() != one_body_integrals.cols() ||
      !one_body_integrals.allFinite() ||
      !one_body_integrals.isApprox(one_body_integrals.transpose(), 1e-12) ||
      !std::isfinite(core_energy)) {
    throw std::invalid_argument(
        "MO SCF requires a finite, symmetric one-body matrix and finite core "
        "energy.");
  }
  if (!eri || eri->num_basis_functions() != num_atomic_orbitals_) {
    throw std::invalid_argument("MO SCF integral dimensions are inconsistent.");
  }
  if (cfg.mpi.world_size != 1 || cfg.mpi.world_rank != 0) {
    throw std::invalid_argument("MO SCF currently requires one MPI rank.");
  }
  if (cfg.require_gradient || cfg.require_polarizability || cfg.do_dfj ||
      cfg.scf_algorithm.method == SCFAlgorithmName::ASAHF) {
    throw std::invalid_argument(
        "MO SCF does not support atom-specific guesses, density fitting, or "
        "nuclear properties.");
  }
  if (cfg.density_init_method != DensityInitializationMethod::UserProvided) {
    throw std::invalid_argument(
        "MO SCF uses the input-basis reference; select UserProvided density "
        "initialization.");
  }
  if (cfg.scf_orbital_type == SCFOrbitalType::RestrictedOpenShell &&
      nalpha < nbeta) {
    throw std::invalid_argument("ROHF requires nalpha >= nbeta.");
  }
  eri_ = std::move(eri);
  for (int i = 0; i < nalpha; ++i) {
    P_(i, i) = num_density_matrices_ == 1 ? 2.0 : 1.0;
  }
  if (num_density_matrices_ == 2) {
    for (int i = 0; i < nbeta; ++i) {
      P_(num_atomic_orbitals_ + i, i) = 1.0;
    }
  }
  density_matrix_initialized_ = true;
}

void MoSCFImpl::build_one_electron_integrals_() {
  H_ = one_body_integrals_.replicate(num_density_matrices_, 1);
  S_ = RowMajorMatrix::Identity(num_atomic_orbitals_, num_atomic_orbitals_);
  X_ = S_;
  C_ = S_.replicate(num_orbital_spin_blocks_, 1);
}

}  // namespace qdk::chemistry::scf
