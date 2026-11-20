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
#include <qdk/chemistry/data/two_body_kernel.hpp>
#include <string>
#include <utility>

namespace qdk::chemistry::algorithms {

/**
 * @class JKBuilderSettings
 * @brief Settings of J/K builders and J/K gradient builders.
 */
class JKBuilderSettings : public data::Settings {
 public:
  /**
   * @brief Default constructor
   */
  JKBuilderSettings() {
    set_default("eri_threshold", 1e-12, "Integral screening threshold");
  }
};

/**
 * @class JKBuilder
 * @brief Builds the mean-field Coulomb (J) and exchange (K) matrices of
 * density matrices.
 *
 * J[P]_{mn} = sum_{ls} (mn|g_J|ls) P_{ls} and
 * K[P]_{mn} = sum_{ls} (ml|g_K|ns) P_{ls} with the two-body kernels g_J and
 * g_K, for the density matrix P_s of each spin block. A null kernel skips its
 * matrix. The "qdk" ScfSolver builds its J and K matrices with the builder
 * named by its "jk_builder" setting.
 */
class JKBuilder
    : public Algorithm<
          JKBuilder,
          std::pair<std::shared_ptr<const data::SymmetryBlockedTensor<2>>,
                    std::shared_ptr<const data::SymmetryBlockedTensor<2>>>,
          std::shared_ptr<data::BasisSet>,
          std::shared_ptr<const data::TwoBodyKernel>,
          std::shared_ptr<const data::TwoBodyKernel>,
          std::shared_ptr<const data::SymmetryBlockedTensor<2>>> {
 public:
  /**
   * @brief Default constructor
   */
  JKBuilder() { _settings = std::make_unique<JKBuilderSettings>(); }

  /**
   * @brief Virtual destructor for proper inheritance
   */
  virtual ~JKBuilder() = default;

  /**
   * @brief Build J and K of density matrices
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param basis_set Basis set whose atomic orbitals, in their order and with
   * each contracted shell normalized, index the matrices
   * @param j_kernel Kernel g_J of J, or null to skip J
   * @param k_kernel Kernel g_K of K, or null to skip K
   * @param densities Symmetric density matrices P_alpha and P_beta, the
   * alpha-alpha and beta-beta blocks of a rank-2 spin-diagonal tensor
   * \endcond
   * @return J[P_s] and K[P_s] in the spin structure of the densities, each
   * null if skipped
   */
  using Algorithm::run;

  /**
   * @brief Access the algorithm's type name
   *
   * @return "jk_builder"
   */
  std::string type_name() const final { return "jk_builder"; }

  /**
   * @brief Access the builder's name
   *
   * @return The name under which the builder is registered
   */
  std::string name() const override = 0;

 protected:
  /**
   * @brief Build J and K of density matrices
   *
   * @param basis_set Basis set of the matrices
   * @param j_kernel Kernel of J, or null to skip J
   * @param k_kernel Kernel of K, or null to skip K
   * @param densities Density matrices, one per spin block
   * @return J[P_s] and K[P_s] in the spin structure of the densities, each
   * null if skipped
   */
  virtual std::pair<std::shared_ptr<const data::SymmetryBlockedTensor<2>>,
                    std::shared_ptr<const data::SymmetryBlockedTensor<2>>>
  _run_impl(std::shared_ptr<data::BasisSet> basis_set,
            std::shared_ptr<const data::TwoBodyKernel> j_kernel,
            std::shared_ptr<const data::TwoBodyKernel> k_kernel,
            std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities)
      const override = 0;
};

/**
 * @class JKGradientBuilder
 * @brief Builds the nuclear gradients of the mean-field Coulomb and exchange
 * energies.
 *
 * Derivatives with respect to the nuclear coordinates, at fixed density
 * matrices, of E_J = 1/2 Tr(P J[P]) for P = P_alpha + P_beta and of
 * E_K = -1/2 sum_s Tr(P_s K[P_s]), with J and K of the kernels g_J and g_K
 * as in JKBuilder. The "qdk" ScfSolver builds its analytic gradients with the
 * builder named by its "jk_gradient_builder" setting.
 */
class JKGradientBuilder
    : public Algorithm<JKGradientBuilder,
                       std::pair<std::shared_ptr<data::NuclearGradients>,
                                 std::shared_ptr<data::NuclearGradients>>,
                       std::shared_ptr<data::BasisSet>,
                       std::shared_ptr<const data::TwoBodyKernel>,
                       std::shared_ptr<const data::TwoBodyKernel>,
                       std::shared_ptr<const data::SymmetryBlockedTensor<2>>> {
 public:
  /**
   * @brief Default constructor
   */
  JKGradientBuilder() { _settings = std::make_unique<JKBuilderSettings>(); }

  /**
   * @brief Virtual destructor for proper inheritance
   */
  virtual ~JKGradientBuilder() = default;

  /**
   * @brief Build the nuclear gradients of E_J and E_K
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param basis_set Basis set, as in JKBuilder::run
   * @param j_kernel Kernel of J, or null to skip E_J
   * @param k_kernel Kernel of K, or null to skip E_K
   * @param densities Density matrices, as in JKBuilder::run
   * \endcond
   * @return The gradients of E_J and E_K on the basis set's structure, each
   * null if skipped
   */
  using Algorithm::run;

  /**
   * @brief Access the algorithm's type name
   *
   * @return "jk_gradient_builder"
   */
  std::string type_name() const final { return "jk_gradient_builder"; }

  /**
   * @brief Access the builder's name
   *
   * @return The name under which the builder is registered
   */
  std::string name() const override = 0;

 protected:
  /**
   * @brief Build the nuclear gradients of E_J and E_K
   *
   * @param basis_set Basis set of the density matrices
   * @param j_kernel Kernel of J, or null to skip E_J
   * @param k_kernel Kernel of K, or null to skip E_K
   * @param densities Density matrices, one per spin block
   * @return The gradients of E_J and E_K, each null if skipped
   */
  virtual std::pair<std::shared_ptr<data::NuclearGradients>,
                    std::shared_ptr<data::NuclearGradients>>
  _run_impl(std::shared_ptr<data::BasisSet> basis_set,
            std::shared_ptr<const data::TwoBodyKernel> j_kernel,
            std::shared_ptr<const data::TwoBodyKernel> k_kernel,
            std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities)
      const override = 0;
};

/**
 * @brief Factory for J/K builders.
 *
 * The library provides "qdk" (the default); plugins can register others.
 */
struct JKBuilderFactory : public AlgorithmFactory<JKBuilder, JKBuilderFactory> {
  /**
   * @brief Access the type name of the algorithms the factory creates
   *
   * @return "jk_builder"
   */
  static std::string algorithm_type_name() { return "jk_builder"; }

  /**
   * @brief Register the J/K builders of the library
   */
  static void register_default_instances();

  /**
   * @brief Access the name of the default J/K builder
   *
   * @return "qdk"
   */
  static std::string default_algorithm_name() { return "qdk"; }
};

/**
 * @brief Factory for J/K gradient builders.
 *
 * The library provides "qdk" (the default); plugins can register others.
 */
struct JKGradientBuilderFactory
    : public AlgorithmFactory<JKGradientBuilder, JKGradientBuilderFactory> {
  /**
   * @brief Access the type name of the algorithms the factory creates
   *
   * @return "jk_gradient_builder"
   */
  static std::string algorithm_type_name() { return "jk_gradient_builder"; }

  /**
   * @brief Register the J/K gradient builders of the library
   */
  static void register_default_instances();

  /**
   * @brief Access the name of the default J/K gradient builder
   *
   * @return "qdk"
   */
  static std::string default_algorithm_name() { return "qdk"; }
};

}  // namespace qdk::chemistry::algorithms
