// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/scf/scf_solver.h>

#include "scf/ks_impl.h"
#include "scf/scf_impl.h"

namespace qdk::chemistry::scf {
std::unique_ptr<SCF> SCF::make_hf_solver(std::shared_ptr<Molecule> mol,
                                         const SCFConfig& cfg) {
  auto impl = std::make_unique<SCFImpl>(mol, cfg, false);
  return std::unique_ptr<SCF>(new SCF(std::move(impl)));
}

std::unique_ptr<SCF> SCF::make_hf_solver(std::shared_ptr<Molecule> mol,
                                         const SCFConfig& cfg,
                                         const RowMajorMatrix& density_matrix) {
  auto impl = std::make_unique<SCFImpl>(mol, cfg, density_matrix, false);
  return std::unique_ptr<SCF>(new SCF(std::move(impl)));
}

std::unique_ptr<SCF> SCF::make_ks_solver(std::shared_ptr<Molecule> mol,
                                         const SCFConfig& cfg) {
  auto impl = std::make_unique<KSImpl>(mol, cfg);
  return std::unique_ptr<SCF>(new SCF(std::move(impl)));
}

std::unique_ptr<SCF> SCF::make_ks_solver(std::shared_ptr<Molecule> mol,
                                         const SCFConfig& cfg,
                                         const RowMajorMatrix& density_matrix) {
  auto impl = std::make_unique<KSImpl>(mol, cfg, density_matrix);
  return std::unique_ptr<SCF>(new SCF(std::move(impl)));
}

SCF::SCF(std::unique_ptr<SCFImpl> impl) : impl_(std::move(impl)) {}

SCF::~SCF() noexcept = default;

const SCFContext& SCF::run() { return impl_->run(); }

const SCFContext& SCF::context() const { return impl_->context(); }

std::vector<std::pair<std::string, const RowMajorMatrix&>> SCF::get_matrices()
    const {
  return impl_->get_matrices();
}

const RowMajorMatrix& SCF::overlap() const { return impl_->overlap(); }

bool SCF::get_restricted() const { return impl_->get_restricted(); }

std::vector<int> SCF::get_num_electrons() const {
  return impl_->get_num_electrons();
}

int SCF::get_num_basis_functions() const {
  return impl_->get_num_basis_functions();
}

int SCF::get_num_molecular_orbitals() const {
  return impl_->get_num_molecular_orbitals();
}

const RowMajorMatrix& SCF::get_eigenvalues() const {
  return impl_->get_eigenvalues();
}

const RowMajorMatrix& SCF::get_density_matrix() const {
  return impl_->get_density_matrix();
}

const RowMajorMatrix& SCF::get_fock_matrix() const {
  return impl_->get_fock_matrix();
}

const RowMajorMatrix& SCF::get_orbitals_matrix() const {
  return impl_->get_orbitals_matrix();
}

int SCF::get_num_density_matrices() const {
  return impl_->get_num_density_matrices();
}
}  // namespace qdk::chemistry::scf
