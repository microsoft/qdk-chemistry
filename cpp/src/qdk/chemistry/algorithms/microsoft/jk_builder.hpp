// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/jk_builder.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::scf {
class BasisSet;
class ERI;
}  // namespace qdk::chemistry::scf

namespace qdk::chemistry::algorithms::microsoft {

class JKBuilderSettings : public data::Settings {
 public:
  JKBuilderSettings() {
    set_default("eri_method", std::string("direct"),
                "ERI evaluation method: 'direct' computes integrals "
                "on-the-fly, 'incore' stores all integrals in memory",
                data::ListConstraint<std::string>{
                    {std::vector<std::string>{"direct", "incore"}}});
    set_default("eri_use_atomics", false,
                "Accumulate direct J/K builds with atomic operations instead "
                "of per-thread buffers");
    set_default("shell_pair_threshold", 1e-12,
                "Overlap-based shell pair pre-screening threshold. Shell "
                "pairs with overlap norm below this value are excluded.");
  }
  ~JKBuilderSettings() override = default;
};

/**
 * @class JKEngine
 * @brief J/K engine of the "qdk" builder.
 *
 * Evaluates the integrals with Libint2, directly or in memory. Gradients are
 * always evaluated directly.
 */
class JKEngine : public algorithms::JKEngine {
 public:
  JKEngine(std::shared_ptr<data::BasisSet> basis_set, double eri_threshold,
           bool incore, bool use_atomics, double shell_pair_threshold);
  ~JKEngine() override;

  void build(const double* P, size_t num_atomic_orbitals, size_t num_densities,
             double* J, double* K, double alpha, double beta,
             double omega) override;

  void build_gradients(const double* P, size_t num_atomic_orbitals,
                       size_t num_densities, size_t num_atoms, double* dJ,
                       double* dK, double alpha, double beta,
                       double omega) override;

 private:
  /// Integral engine for num_densities density matrices
  qdk::chemistry::scf::ERI& eri(size_t num_atomic_orbitals,
                                size_t num_densities, bool incore,
                                double omega);

  std::shared_ptr<data::BasisSet> basis_set_;
  double eri_threshold_;
  bool incore_;
  bool use_atomics_;
  double shell_pair_threshold_;
  std::shared_ptr<qdk::chemistry::scf::BasisSet> scf_basis_set_;
  /// Engines for one (restricted) and two (unrestricted) densities
  std::unique_ptr<qdk::chemistry::scf::ERI> direct_[2];
  std::unique_ptr<qdk::chemistry::scf::ERI> incore_eri_[2];
  double incore_omega_[2] = {};
};

/**
 * @class JKBuilder
 * @brief QDK/Chemistry's J/K builder ("qdk").
 */
class JKBuilder : public algorithms::JKBuilder {
 public:
  JKBuilder() { _settings = std::make_unique<JKBuilderSettings>(); }

  std::string name() const override { return "qdk"; }

 protected:
  std::shared_ptr<algorithms::JKEngine> _run_impl(
      std::shared_ptr<data::BasisSet> basis_set,
      double eri_threshold) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
