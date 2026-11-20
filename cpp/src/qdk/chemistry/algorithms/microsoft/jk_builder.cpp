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

namespace detail {

/// Whether the densities are one density matrix (restricted) rather than two
bool is_restricted(const data::SymmetryBlockedTensor<2>& densities) {
  const auto& symmetries = densities.symmetries()[0];
  if (!symmetries || !symmetries->has_axis(data::AxisName::Spin)) {
    throw std::invalid_argument("Densities must be blocked by spin");
  }
  return symmetries->axis(data::AxisName::Spin).equivalent();
}

/// The density matrices of the spin blocks as consecutive matrices, the alpha
/// one alone if restricted, times scale
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
    throw std::invalid_argument("Densities must be matrices in the " +
                                std::to_string(n) +
                                " atomic orbitals of the basis set");
  }
  Eigen::MatrixXd P(n, (restricted ? 1 : 2) * n);
  P.leftCols(n) = scale * P_alpha;
  if (!restricted) P.rightCols(n) = scale * P_beta;
  return P;
}

/// Converts basis_set unless it is the cached basis set, and releases the
/// integral engines unless they are for the cached basis set and restricted
void cache_basis_set(const std::shared_ptr<data::BasisSet>& basis_set,
                     bool restricted, std::shared_ptr<data::BasisSet>& cached,
                     std::shared_ptr<qcs::BasisSet>& converted,
                     KernelEngines& engines, bool& engines_restricted) {
  if (basis_set != cached) {
    engines.clear();
    cached.reset();
    converted = utils::microsoft::convert_basis_set_from_qdk(*basis_set);
    // The engines size their buffers from the converted basis set.
    if (converted->num_atomic_orbitals !=
            basis_set->get_num_atomic_orbitals() ||
        converted->mol->n_atoms !=
            basis_set->get_structure()->get_num_atoms()) {
      throw std::runtime_error(
          "Converted basis set does not match the atomic orbitals and atoms "
          "of the basis set");
    }
    cached = basis_set;
  }
  if (engines_restricted != restricted) engines.clear();
  engines_restricted = restricted;
}

/// The engine of kernel in engines, made by make_engine if absent
template <class MakeEngine>
qcs::ERI& kernel_engine(
    const std::shared_ptr<const data::TwoBodyKernel>& kernel,
    KernelEngines& engines, const MakeEngine& make_engine) {
  const auto key = kernel->content_hash();
  auto it = engines.find(key);
  if (it == engines.end()) it = engines.emplace(key, make_engine(kernel)).first;
  return *it->second;
}

}  // namespace detail

JKBuilder::JKBuilder() { _settings = std::make_unique<JKBuilderSettings>(); }

JKBuilder::~JKBuilder() = default;

std::pair<std::shared_ptr<const data::SymmetryBlockedTensor<2>>,
          std::shared_ptr<const data::SymmetryBlockedTensor<2>>>
JKBuilder::_run_impl(
    std::shared_ptr<data::BasisSet> basis_set,
    std::shared_ptr<const data::TwoBodyKernel> j_kernel,
    std::shared_ptr<const data::TwoBodyKernel> k_kernel,
    std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities) const {
  utils::microsoft::initialize_backend();
  const size_t n = basis_set->get_num_atomic_orbitals();
  const bool restricted = detail::is_restricted(*densities);
  const Eigen::MatrixXd P = detail::stacked(*densities, n, restricted, 1.0);

  // The converted basis set and the integral engines of its kernels are kept
  // for later runs with the same basis set object and number of density
  // matrices.
  detail::cache_basis_set(basis_set, restricted, _basis_set, _scf_basis_set,
                          _engines, _restricted);
  const auto type = restricted ? qcs::SCFOrbitalType::Restricted
                               : qcs::SCFOrbitalType::Unrestricted;
  auto make_engine =
      [&](const std::shared_ptr<const data::TwoBodyKernel>& kernel)
      -> std::unique_ptr<qcs::ERI> {
    if (_settings->get<std::string>("eri_method") == "incore") {
      return std::make_unique<qcs::ERIINCORE>(
          type, *_scf_basis_set, qcs::mpi_default_input(), 0.0, kernel);
    }
    return std::make_unique<qcs::LIBINT2_DIRECT>(
        type, *_scf_basis_set, qcs::mpi_default_input(),
        _settings->get<bool>("eri_use_atomics"),
        _settings->get<double>("eri_threshold"),
        _settings->get<double>("shell_pair_threshold"), kernel);
  };

  // J and K of the same kernel are built from the same integrals.
  const bool fused = j_kernel && k_kernel &&
                     j_kernel->content_hash() == k_kernel->content_hash();
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, P.cols());
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, P.cols());
  if (j_kernel) {
    detail::kernel_engine(j_kernel, _engines, make_engine)
        .build_JK(P.data(), J.data(), fused ? K.data() : nullptr, 1.0, 0.0,
                  0.0);
  }
  if (k_kernel && !fused) {
    detail::kernel_engine(k_kernel, _engines, make_engine)
        .build_JK(P.data(), nullptr, K.data(), 1.0, 0.0, 0.0);
  }

  auto spin_blocks = [&](const Eigen::MatrixXd& X) {
    return std::make_shared<const data::SymmetryBlockedTensor<2>>(
        data::make_spin_diagonal_rank2_sbt(X.leftCols(n), X.rightCols(n),
                                           restricted));
  };
  return {j_kernel ? spin_blocks(J) : nullptr,
          k_kernel ? spin_blocks(K) : nullptr};
}

JKGradientBuilder::JKGradientBuilder() {
  _settings = std::make_unique<JKGradientBuilderSettings>();
}

JKGradientBuilder::~JKGradientBuilder() = default;

std::pair<std::shared_ptr<data::NuclearGradients>,
          std::shared_ptr<data::NuclearGradients>>
JKGradientBuilder::_run_impl(
    std::shared_ptr<data::BasisSet> basis_set,
    std::shared_ptr<const data::TwoBodyKernel> j_kernel,
    std::shared_ptr<const data::TwoBodyKernel> k_kernel,
    std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities) const {
  utils::microsoft::initialize_backend();
  const size_t n = basis_set->get_num_atomic_orbitals();
  const bool restricted = detail::is_restricted(*densities);
  // For one density, the engines take the total density P = 2 P_alpha.
  const Eigen::MatrixXd P =
      detail::stacked(*densities, n, restricted, restricted ? 2.0 : 1.0);

  // The converted basis set and the integral engines of its kernels are kept
  // for later runs with the same basis set object and number of density
  // matrices.
  detail::cache_basis_set(basis_set, restricted, _basis_set, _scf_basis_set,
                          _engines, _restricted);
  const auto type = restricted ? qcs::SCFOrbitalType::Restricted
                               : qcs::SCFOrbitalType::Unrestricted;
  auto make_engine =
      [&](const std::shared_ptr<const data::TwoBodyKernel>& kernel)
      -> std::unique_ptr<qcs::ERI> {
    return std::make_unique<qcs::LIBINT2_DIRECT>(
        type, *_scf_basis_set, qcs::mpi_default_input(), false /*use_atomics*/,
        _settings->get<double>("eri_threshold"),
        _settings->get<double>("shell_pair_threshold"), kernel);
  };

  const auto structure = basis_set->get_structure();
  using Gradient = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>;
  Gradient dJ = Gradient::Zero(3, structure->get_num_atoms());
  Gradient dK = Gradient::Zero(3, structure->get_num_atoms());
  // The gradients of E_J and E_K of the same kernel share the integrals.
  const bool fused = j_kernel && k_kernel &&
                     j_kernel->content_hash() == k_kernel->content_hash();
  if (j_kernel) {
    detail::kernel_engine(j_kernel, _engines, make_engine)
        .get_gradients(P.data(), dJ.data(), fused ? dK.data() : nullptr, 1.0,
                       0.0, 0.0);
  }
  if (k_kernel && !fused) {
    detail::kernel_engine(k_kernel, _engines, make_engine)
        .get_gradients(P.data(), nullptr, dK.data(), 1.0, 0.0, 0.0);
  }

  auto gradients = [&](const Gradient& d) {
    const Eigen::Matrix3Xd atom_major = d;
    return std::make_shared<data::NuclearGradients>(
        structure,
        Eigen::Map<const Eigen::VectorXd>(atom_major.data(), d.size()));
  };
  return {j_kernel ? gradients(dJ) : nullptr,
          k_kernel ? gradients(dK) : nullptr};
}

}  // namespace qdk::chemistry::algorithms::microsoft
