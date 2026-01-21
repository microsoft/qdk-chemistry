// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/scf/scf_solver.h>

#include <qdk/chemistry/utils/logger.hpp>

#include "scf/ks_impl.h"
#include "scf/scf_impl.h"

namespace qdk::chemistry::scf {
std::unique_ptr<SCF> SCF::make_hf_solver(std::shared_ptr<Molecule> mol,
                                         const SCFConfig& cfg) {
  QDK_LOG_TRACE_ENTERING();
  auto impl = std::make_unique<SCFImpl>(mol, cfg, false);
  return std::unique_ptr<SCF>(new SCF(std::move(impl)));
}

std::unique_ptr<SCF> SCF::make_hf_solver(
    std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
    const RowMajorMatrix& density_matrix, std::shared_ptr<BasisSet> basis_set,
    std::shared_ptr<BasisSet> raw_basis_set) {
  QDK_LOG_TRACE_ENTERING();
  auto impl = std::make_unique<SCFImpl>(mol, cfg, density_matrix, basis_set,
                                        raw_basis_set, false);
  return std::unique_ptr<SCF>(new SCF(std::move(impl)));
}

std::unique_ptr<SCF> SCF::make_hf_solver(
    std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
    std::shared_ptr<BasisSet> basis_set,
    std::shared_ptr<BasisSet> raw_basis_set) {
  if (!basis_set)
    throw std::invalid_argument("Basis set pointer cannot be null.");
  if (!raw_basis_set)
    throw std::invalid_argument("Raw basis set pointer cannot be null.");

  auto impl =
      std::make_unique<SCFImpl>(mol, cfg, basis_set, raw_basis_set, false);
  return std::unique_ptr<SCF>(new SCF(std::move(impl)));
}

std::unique_ptr<SCF> SCF::make_ks_solver(std::shared_ptr<Molecule> mol,
                                         const SCFConfig& cfg) {
  QDK_LOG_TRACE_ENTERING();
  auto impl = std::make_unique<KSImpl>(mol, cfg);
  return std::unique_ptr<SCF>(new SCF(std::move(impl)));
}

std::unique_ptr<SCF> SCF::make_ks_solver(
    std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
    const RowMajorMatrix& density_matrix, std::shared_ptr<BasisSet> basis_set,
    std::shared_ptr<BasisSet> raw_basis_set) {
  QDK_LOG_TRACE_ENTERING();
  auto impl = std::make_unique<KSImpl>(mol, cfg, density_matrix, basis_set,
                                       raw_basis_set);
  return std::unique_ptr<SCF>(new SCF(std::move(impl)));
}

std::unique_ptr<SCF> SCF::make_ks_solver(
    std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
    std::shared_ptr<BasisSet> basis_set,
    std::shared_ptr<BasisSet> raw_basis_set) {
  if (!basis_set)
    throw std::invalid_argument("Basis set pointer cannot be null.");
  if (!raw_basis_set)
    throw std::invalid_argument("Raw basis set pointer cannot be null.");

  auto impl = std::make_unique<KSImpl>(mol, cfg, basis_set, raw_basis_set);
  return std::unique_ptr<SCF>(new SCF(std::move(impl)));
}

SCF::SCF(std::unique_ptr<SCFImpl> impl) : impl_(std::move(impl)) {
  QDK_LOG_TRACE_ENTERING();
}

SCF::~SCF() noexcept = default;

const SCFContext& SCF::run() {
  QDK_LOG_TRACE_ENTERING();
  return impl_->run();
}

const SCFContext& SCF::context() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->context();
}

std::vector<std::pair<std::string, const RowMajorMatrix&>> SCF::get_matrices()
    const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_matrices();
}

const RowMajorMatrix& SCF::overlap() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->overlap();
}

bool SCF::get_restricted() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_restricted();
}

std::vector<int> SCF::get_num_electrons() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_num_electrons();
}

int SCF::get_num_atomic_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_num_atomic_orbitals();
}

int SCF::get_num_molecular_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_num_molecular_orbitals();
}

const RowMajorMatrix& SCF::get_eigenvalues() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_eigenvalues();
}

const RowMajorMatrix& SCF::get_density_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_density_matrix();
}

const RowMajorMatrix& SCF::get_fock_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_fock_matrix();
}

const RowMajorMatrix& SCF::get_orbitals_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_orbitals_matrix();
}

int SCF::get_num_density_matrices() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_num_density_matrices();
}
}  // namespace qdk::chemistry::scf
