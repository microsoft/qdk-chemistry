// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/jk_builder.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::scf {
class BasisSet;
class ERI;
}  // namespace qdk::chemistry::scf

namespace qdk::chemistry::algorithms::microsoft {

class JKBuilderSettings : public algorithms::JKBuilderSettings {
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
};

class JKGradientBuilderSettings : public algorithms::JKBuilderSettings {
 public:
  JKGradientBuilderSettings() {
    set_default("shell_pair_threshold", 1e-12,
                "Overlap-based shell pair pre-screening threshold. Shell "
                "pairs with overlap norm below this value are excluded.");
  }
};

/**
 * @class JKBuilder
 * @brief QDK/Chemistry's J/K builder ("qdk").
 *
 * Evaluates the integrals with Libint2, directly or in memory, and keeps the
 * integral data of the last basis set for later runs.
 */
class JKBuilder : public algorithms::JKBuilder {
 public:
  JKBuilder();
  ~JKBuilder() override;

  std::string name() const override { return "qdk"; }

 protected:
  std::pair<std::shared_ptr<const data::SymmetryBlockedTensor<2>>,
            std::shared_ptr<const data::SymmetryBlockedTensor<2>>>
  _run_impl(std::shared_ptr<data::BasisSet> basis_set,
            std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities)
      const override;

 private:
  mutable std::shared_ptr<data::BasisSet> _basis_set;
  mutable std::shared_ptr<scf::BasisSet> _scf_basis_set;
  /// Integral engines for one (restricted) and two (unrestricted) densities
  mutable std::unique_ptr<scf::ERI> _eri[2];
};

/**
 * @class JKGradientBuilder
 * @brief QDK/Chemistry's J/K gradient builder ("qdk").
 *
 * Evaluates the integral derivatives directly with Libint2 and keeps the
 * integral screening data of the last basis set for later runs.
 */
class JKGradientBuilder : public algorithms::JKGradientBuilder {
 public:
  JKGradientBuilder();
  ~JKGradientBuilder() override;

  std::string name() const override { return "qdk"; }

 protected:
  std::pair<std::shared_ptr<data::NuclearGradients>,
            std::shared_ptr<data::NuclearGradients>>
  _run_impl(std::shared_ptr<data::BasisSet> basis_set,
            std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities)
      const override;

 private:
  mutable std::shared_ptr<data::BasisSet> _basis_set;
  mutable std::shared_ptr<scf::BasisSet> _scf_basis_set;
  /// Integral engines for one (restricted) and two (unrestricted) densities
  mutable std::unique_ptr<scf::ERI> _eri[2];
};

}  // namespace qdk::chemistry::algorithms::microsoft
