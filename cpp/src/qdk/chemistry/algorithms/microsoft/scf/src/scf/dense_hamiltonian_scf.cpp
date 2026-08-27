// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/scf/scf_solver.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <qdk/chemistry/scf/core/eri.h>
#include <stdexcept>
#include <utility>

#include "scf_impl.h"

namespace qdk::chemistry::scf {
namespace {

RowMajorMatrix closed_shell_density(std::size_t n_orbitals,
                                    std::size_t n_occupied) {
  if (n_occupied > n_orbitals)
    throw std::invalid_argument(
        "Dense Hamiltonian SCF has more occupied than total orbitals");
  RowMajorMatrix density =
      RowMajorMatrix::Zero(n_orbitals, n_orbitals);
  for (std::size_t i = 0; i < n_occupied; ++i)
    density(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = 2.0;
  return density;
}

class DensePhysicistERI final : public ERI {
 public:
  DensePhysicistERI(BasisSet& basis_set, ParallelConfig mpi,
                    std::vector<double> integrals, std::size_t n_orbitals)
      : ERI(SCFOrbitalType::Restricted, 0.0, basis_set, mpi),
        integrals_(std::move(integrals)),
        n_orbitals_(n_orbitals) {
    if (integrals_.size() != n_orbitals_ * n_orbitals_ * n_orbitals_ *
                                 n_orbitals_)
      throw std::invalid_argument(
          "Dense Hamiltonian SCF two-body tensor has incorrect dimensions");
  }

  void get_gradients(const double*, double*, double*, double, double,
                     double) override {
    throw std::runtime_error(
        "Dense Hamiltonian SCF does not provide nuclear gradients");
  }

 private:
  std::size_t index(std::size_t p, std::size_t q, std::size_t r,
                    std::size_t s) const {
    return ((p * n_orbitals_ + q) * n_orbitals_ + r) * n_orbitals_ + s;
  }

  void build_JK_impl_(const double* density, double* coulomb, double* exchange,
                      double alpha, double beta, double omega) override {
    if (std::abs(omega) > 1e-14)
      throw std::invalid_argument(
          "Dense Hamiltonian SCF does not support range separation");
    const std::size_t matrix_size = n_orbitals_ * n_orbitals_;
    if (coulomb) std::fill_n(coulomb, matrix_size, 0.0);
    if (exchange) std::fill_n(exchange, matrix_size, 0.0);

    for (std::size_t p = 0; p < n_orbitals_; ++p)
      for (std::size_t q = 0; q < n_orbitals_; ++q) {
        double j_value = 0.0;
        double k_value = 0.0;
        for (std::size_t r = 0; r < n_orbitals_; ++r)
          for (std::size_t s = 0; s < n_orbitals_; ++s) {
            const double d = density[r * n_orbitals_ + s];
            j_value += integrals_[index(p, r, q, s)] * d;
            k_value += integrals_[index(p, r, s, q)] * d;
          }
        if (coulomb) coulomb[p * n_orbitals_ + q] = j_value;
        if (exchange)
          exchange[p * n_orbitals_ + q] = (alpha + beta) * k_value;
      }

    for (std::size_t p = 0; p < n_orbitals_; ++p)
      for (std::size_t q = 0; q < p; ++q) {
        if (coulomb) {
          const double value =
              0.5 * (coulomb[p * n_orbitals_ + q] +
                     coulomb[q * n_orbitals_ + p]);
          coulomb[p * n_orbitals_ + q] = value;
          coulomb[q * n_orbitals_ + p] = value;
        }
        if (exchange) {
          const double value =
              0.5 * (exchange[p * n_orbitals_ + q] +
                     exchange[q * n_orbitals_ + p]);
          exchange[p * n_orbitals_ + q] = value;
          exchange[q * n_orbitals_ + p] = value;
        }
      }
  }

  void quarter_trans_impl(std::size_t, const double*, double*) override {
    throw std::runtime_error(
        "Dense Hamiltonian SCF does not provide AO-to-MO transformation");
  }

  std::vector<double> integrals_;
  std::size_t n_orbitals_;
};

class DenseHamiltonianSCFImpl final : public SCFImpl {
 public:
  DenseHamiltonianSCFImpl(const SCFConfig& cfg, RowMajorMatrix one_body,
                          std::vector<double> two_body_physicist,
                          std::size_t n_occupied,
                          std::shared_ptr<BasisSet> basis_set,
                          double scalar_energy)
      : SCFImpl(
            basis_set ? basis_set->mol : nullptr, cfg,
            closed_shell_density(
                static_cast<std::size_t>(one_body.rows()), n_occupied),
            basis_set, basis_set, /*delay_eri=*/true),
        one_body_(std::move(one_body)),
        scalar_energy_(scalar_energy) {
    if (cfg.scf_orbital_type != SCFOrbitalType::Restricted)
      throw std::invalid_argument(
          "Dense Hamiltonian SCF currently supports restricted orbitals only");
    if (cfg.density_init_method !=
        DensityInitializationMethod::UserProvided)
      throw std::invalid_argument(
          "Dense Hamiltonian SCF requires a user-provided density");
    if (one_body_.rows() != one_body_.cols() ||
        static_cast<std::size_t>(one_body_.rows()) != num_atomic_orbitals_)
      throw std::invalid_argument(
          "Dense Hamiltonian SCF one-body matrix has incorrect dimensions");
    if (static_cast<std::size_t>(nelec_[0]) != n_occupied ||
        nelec_[0] != nelec_[1])
      throw std::invalid_argument(
          "Dense Hamiltonian SCF occupation disagrees with molecular metadata");

    eri_ = std::make_shared<DensePhysicistERI>(
        *ctx_.basis_set, cfg.mpi, std::move(two_body_physicist),
        num_atomic_orbitals_);
  }

 private:
  void build_one_electron_integrals_() override {
    S_ = RowMajorMatrix::Identity(num_atomic_orbitals_, num_atomic_orbitals_);
    X_ = S_;
    H_ = one_body_;
    num_molecular_orbitals_ = num_atomic_orbitals_;
    ctx_.num_molecular_orbitals = num_molecular_orbitals_;
    C_ = RowMajorMatrix::Identity(num_atomic_orbitals_, num_atomic_orbitals_);
    eigenvalues_ = RowMajorMatrix::Zero(1, num_molecular_orbitals_);
  }

  double calc_nuclear_repulsion_energy_() override { return scalar_energy_; }

  void properties_() override {}

  RowMajorMatrix one_body_;
  double scalar_energy_;
};

}  // namespace

std::unique_ptr<SCF> SCF::make_restricted_dense_hamiltonian_solver(
    const SCFConfig& cfg, const RowMajorMatrix& one_body,
    const std::vector<double>& two_body_physicist, std::size_t n_occupied,
    std::shared_ptr<BasisSet> basis_set, double scalar_energy) {
  if (!basis_set)
    throw std::invalid_argument(
        "Dense Hamiltonian SCF basis set pointer cannot be null");
  auto impl = std::make_unique<DenseHamiltonianSCFImpl>(
      cfg, one_body, two_body_physicist, n_occupied, std::move(basis_set),
      scalar_energy);
  return std::unique_ptr<SCF>(new SCF(std::move(impl)));
}

}  // namespace qdk::chemistry::scf