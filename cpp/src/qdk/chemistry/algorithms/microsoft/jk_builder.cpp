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

namespace {

bool is_restricted(const data::SymmetryBlockedTensor<2>& densities) {
  return densities.symmetries()[0]->axis(data::AxisName::Spin).equivalent();
}

/// The spin densities as consecutive matrices, P_alpha alone if restricted,
/// times scale
Eigen::MatrixXd stacked(const data::SymmetryBlockedTensor<2>& densities,
                        size_t num_atomic_orbitals, bool restricted,
                        double scale) {
  const size_t n = num_atomic_orbitals;
  const auto alpha = data::axes::alpha();
  const auto beta = data::axes::beta();
  const auto& P_alpha = densities.block({alpha, alpha});
  const auto& P_beta = densities.block({beta, beta});
  if (static_cast<size_t>(P_alpha.rows()) != n ||
      static_cast<size_t>(P_alpha.cols()) != n ||
      static_cast<size_t>(P_beta.rows()) != n ||
      static_cast<size_t>(P_beta.cols()) != n) {
    throw std::invalid_argument("Spin densities must be matrices in the " +
                                std::to_string(n) +
                                " atomic orbitals of the basis set");
  }
  Eigen::MatrixXd P(n, (restricted ? 1 : 2) * n);
  P.leftCols(n) = scale * P_alpha;
  if (!restricted) P.rightCols(n) = scale * P_beta;
  return P;
}

/// Converts basis_set unless it is the cached basis set, releasing the
/// engines of the cached one
void cache_basis_set(const std::shared_ptr<data::BasisSet>& basis_set,
                     std::shared_ptr<data::BasisSet>& cached,
                     std::shared_ptr<qcs::BasisSet>& converted,
                     std::unique_ptr<qcs::ERI> (&eri)[2]) {
  if (basis_set == cached) return;
  eri[0].reset();
  eri[1].reset();
  converted = utils::microsoft::convert_basis_set_from_qdk(*basis_set);
  cached = basis_set;
}

}  // namespace

JKBuilder::JKBuilder() { _settings = std::make_unique<JKBuilderSettings>(); }

JKBuilder::~JKBuilder() = default;

std::pair<std::shared_ptr<const data::SymmetryBlockedTensor<2>>,
          std::shared_ptr<const data::SymmetryBlockedTensor<2>>>
JKBuilder::_run_impl(
    std::shared_ptr<data::BasisSet> basis_set,
    std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities) const {
  utils::microsoft::initialize_backend();
  const size_t n = basis_set->get_num_atomic_orbitals();
  const bool restricted = is_restricted(*densities);
  const Eigen::MatrixXd P = stacked(*densities, n, restricted, 1.0);

  cache_basis_set(basis_set, _basis_set, _scf_basis_set, _eri);
  auto& eri = _eri[restricted ? 0 : 1];
  if (!eri) {
    const auto type = restricted ? qcs::SCFOrbitalType::Restricted
                                 : qcs::SCFOrbitalType::Unrestricted;
    if (_settings->get<std::string>("eri_method") == "incore") {
      eri = std::make_unique<qcs::ERIINCORE>(type, *_scf_basis_set,
                                             qcs::mpi_default_input(),
                                             _settings->get<double>("omega"));
    } else {
      eri = std::make_unique<qcs::LIBINT2_DIRECT>(
          type, *_scf_basis_set, qcs::mpi_default_input(),
          _settings->get<bool>("eri_use_atomics"),
          _settings->get<double>("eri_threshold"),
          _settings->get<double>("shell_pair_threshold"));
    }
  }

  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, P.cols());
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, P.cols());
  eri->build_JK(P.data(), J.data(), K.data(), _settings->get<double>("alpha"),
                _settings->get<double>("beta"),
                _settings->get<double>("omega"));

  auto spin_blocks = [&](const Eigen::MatrixXd& X) {
    return std::make_shared<const data::SymmetryBlockedTensor<2>>(
        data::make_spin_diagonal_rank2_sbt(X.leftCols(n), X.rightCols(n),
                                           restricted));
  };
  return {spin_blocks(J), spin_blocks(K)};
}

JKGradientBuilder::JKGradientBuilder() {
  _settings = std::make_unique<JKGradientBuilderSettings>();
}

JKGradientBuilder::~JKGradientBuilder() = default;

std::pair<std::shared_ptr<data::NuclearGradients>,
          std::shared_ptr<data::NuclearGradients>>
JKGradientBuilder::_run_impl(
    std::shared_ptr<data::BasisSet> basis_set,
    std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities) const {
  utils::microsoft::initialize_backend();
  const size_t n = basis_set->get_num_atomic_orbitals();
  const bool restricted = is_restricted(*densities);
  // For one density, the engines take the total density P = 2 P_alpha.
  const Eigen::MatrixXd P =
      stacked(*densities, n, restricted, restricted ? 2.0 : 1.0);

  cache_basis_set(basis_set, _basis_set, _scf_basis_set, _eri);
  auto& eri = _eri[restricted ? 0 : 1];
  if (!eri) {
    eri = std::make_unique<qcs::LIBINT2_DIRECT>(
        restricted ? qcs::SCFOrbitalType::Restricted
                   : qcs::SCFOrbitalType::Unrestricted,
        *_scf_basis_set, qcs::mpi_default_input(), false /*use_atomics*/,
        _settings->get<double>("eri_threshold"),
        _settings->get<double>("shell_pair_threshold"));
  }

  const auto structure = basis_set->get_structure();
  using Gradient = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>;
  Gradient dJ = Gradient::Zero(3, structure->get_num_atoms());
  Gradient dK = Gradient::Zero(3, structure->get_num_atoms());
  eri->get_gradients(
      P.data(), dJ.data(), dK.data(), _settings->get<double>("alpha"),
      _settings->get<double>("beta"), _settings->get<double>("omega"));

  auto gradients = [&](const Gradient& d) {
    const Eigen::Matrix3Xd atom_major = d;
    return std::make_shared<data::NuclearGradients>(
        structure,
        Eigen::Map<const Eigen::VectorXd>(atom_major.data(), d.size()));
  };
  return {gradients(dJ), gradients(dK)};
}

}  // namespace qdk::chemistry::algorithms::microsoft
