// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/nuclear_gradients.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <string>
#include <utility>

namespace qdk::chemistry::algorithms {

/**
 * @class JKBuilderSettings
 * @brief Settings of J/K builders and J/K gradient builders.
 *
 * The exchange operator is v = alpha / r + beta erfc(omega r) / r.
 */
class JKBuilderSettings : public data::Settings {
 public:
  JKBuilderSettings() {
    set_default("eri_threshold", 1e-12, "Integral screening threshold");
    set_default("alpha", 1.0,
                "Coefficient of the full-range exchange operator");
    set_default("beta", 0.0,
                "Coefficient of the short-range exchange operator");
    set_default("omega", 0.0,
                "Range-separation parameter of the short-range exchange "
                "operator");
  }
};

/**
 * @class JKBuilder
 * @brief Builds the Coulomb (J) and exchange (K) matrices of spin densities.
 *
 * J[P]_{mn} = sum_{ls} (mn|ls) P_{ls} and K[P]_{mn} = sum_{ls} (ml|v|ns) P_{ls}
 * with the exchange operator v of the settings, for each spin density P_s.
 * The "qdk" ScfSolver builds its J and K matrices with the builder named by
 * its "jk_builder" setting.
 */
class JKBuilder
    : public Algorithm<
          JKBuilder,
          std::pair<std::shared_ptr<const data::SymmetryBlockedTensor<2>>,
                    std::shared_ptr<const data::SymmetryBlockedTensor<2>>>,
          std::shared_ptr<data::BasisSet>,
          std::shared_ptr<const data::SymmetryBlockedTensor<2>>> {
 public:
  JKBuilder() { _settings = std::make_unique<JKBuilderSettings>(); }
  virtual ~JKBuilder() = default;

  /**
   * @brief Build J and K of spin densities
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param basis_set Basis set whose atomic orbitals, in their order and with
   * each contracted shell normalized, index the matrices
   * @param densities Symmetric spin densities P_alpha and P_beta, the
   * alpha-alpha and beta-beta blocks of a rank-2 spin-diagonal tensor
   * \endcond
   * @return J[P_s] and K[P_s] in the spin structure of the densities
   */
  using Algorithm::run;

  std::string type_name() const final { return "jk_builder"; }

  std::string name() const override = 0;

 protected:
  /**
   * @brief Build J and K of spin densities
   *
   * @param basis_set Basis set of the matrices
   * @param densities Spin densities
   * @return J[P_s] and K[P_s] in the spin structure of the densities
   */
  virtual std::pair<std::shared_ptr<const data::SymmetryBlockedTensor<2>>,
                    std::shared_ptr<const data::SymmetryBlockedTensor<2>>>
  _run_impl(std::shared_ptr<data::BasisSet> basis_set,
            std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities)
      const override = 0;
};

/**
 * @class JKGradientBuilder
 * @brief Builds the nuclear gradients of the Coulomb and exchange energies.
 *
 * Derivatives with respect to the nuclear coordinates, at fixed spin
 * densities, of E_J = 1/2 Tr(P J[P]) for P = P_alpha + P_beta and of
 * E_K = -1/2 sum_s Tr(P_s K[P_s]), with J and K as in JKBuilder. The "qdk"
 * ScfSolver builds its analytic gradients with the builder named by its
 * "jk_gradient_builder" setting.
 */
class JKGradientBuilder
    : public Algorithm<JKGradientBuilder,
                       std::pair<std::shared_ptr<data::NuclearGradients>,
                                 std::shared_ptr<data::NuclearGradients>>,
                       std::shared_ptr<data::BasisSet>,
                       std::shared_ptr<const data::SymmetryBlockedTensor<2>>> {
 public:
  JKGradientBuilder() { _settings = std::make_unique<JKBuilderSettings>(); }
  virtual ~JKGradientBuilder() = default;

  /**
   * @brief Build the nuclear gradients of E_J and E_K
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param basis_set Basis set, as in JKBuilder::run
   * @param densities Spin densities, as in JKBuilder::run
   * \endcond
   * @return The gradients of E_J and E_K on the basis set's structure
   */
  using Algorithm::run;

  std::string type_name() const final { return "jk_gradient_builder"; }

  std::string name() const override = 0;

 protected:
  /**
   * @brief Build the nuclear gradients of E_J and E_K
   *
   * @param basis_set Basis set of the densities
   * @param densities Spin densities
   * @return The gradients of E_J and E_K
   */
  virtual std::pair<std::shared_ptr<data::NuclearGradients>,
                    std::shared_ptr<data::NuclearGradients>>
  _run_impl(std::shared_ptr<data::BasisSet> basis_set,
            std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities)
      const override = 0;
};

/**
 * @brief Factory for J/K builders.
 *
 * The library provides "qdk" (the default); plugins can register others.
 */
struct JKBuilderFactory : public AlgorithmFactory<JKBuilder, JKBuilderFactory> {
  static std::string algorithm_type_name() { return "jk_builder"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk"; }
};

/**
 * @brief Factory for J/K gradient builders.
 *
 * The library provides "qdk" (the default); plugins can register others.
 */
struct JKGradientBuilderFactory
    : public AlgorithmFactory<JKGradientBuilder, JKGradientBuilderFactory> {
  static std::string algorithm_type_name() { return "jk_gradient_builder"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk"; }
};

}  // namespace qdk::chemistry::algorithms
