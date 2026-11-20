// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <map>
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

/// Integral engines of two-body kernels, by kernel content hash
using KernelEngines = std::map<std::string, std::unique_ptr<scf::ERI>>;

/**
 * @class JKBuilderSettings
 * @brief Settings of the "qdk" J/K builder.
 *
 * Adds the integral evaluation method and the screening options of the
 * Libint2 engines to the J/K builder settings.
 */
class JKBuilderSettings : public algorithms::JKBuilderSettings {
 public:
  /**
   * @brief Default constructor
   */
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

/**
 * @class JKGradientBuilderSettings
 * @brief Settings of the "qdk" J/K gradient builder.
 *
 * Adds the screening options of the Libint2 engine to the J/K builder
 * settings.
 */
class JKGradientBuilderSettings : public algorithms::JKBuilderSettings {
 public:
  /**
   * @brief Default constructor
   */
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
 * Evaluates the integrals of the Coulomb, erf, and erfc kernels with Libint2,
 * directly or in memory, and keeps the integral data of each kernel of the
 * last basis set for later runs.
 */
class JKBuilder : public algorithms::JKBuilder {
 public:
  /**
   * @brief Default constructor
   */
  JKBuilder();

  /**
   * @brief Destructor
   */
  ~JKBuilder() override;

  /**
   * @brief Access the builder's name
   *
   * @return "qdk"
   */
  std::string name() const override { return "qdk"; }

 protected:
  /**
   * @brief Build J and K of the density matrices
   *
   * @param basis_set Basis set of the density matrices
   * @param j_kernel Kernel of J, or null to skip J
   * @param k_kernel Kernel of K, or null to skip K
   * @param densities Density matrices, one per spin block
   * @return J and K in the spin structure of the densities, each null if
   * skipped
   * @throws std::invalid_argument If a kernel is not supported, or the
   * densities are not blocked by spin or are not matrices in the atomic
   * orbitals of the basis set
   */
  std::pair<std::shared_ptr<const data::SymmetryBlockedTensor<2>>,
            std::shared_ptr<const data::SymmetryBlockedTensor<2>>>
  _run_impl(std::shared_ptr<data::BasisSet> basis_set,
            std::shared_ptr<const data::TwoBodyKernel> j_kernel,
            std::shared_ptr<const data::TwoBodyKernel> k_kernel,
            std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities)
      const override;

 private:
  /// Basis set of the cached integral engine
  mutable std::shared_ptr<data::BasisSet> _basis_set;
  /// The cached basis set in the format of the integral engine
  mutable std::shared_ptr<scf::BasisSet> _scf_basis_set;
  /// Integral engines of the kernels of the cached basis set
  mutable KernelEngines _engines;
  /// Whether the engines take one density matrix rather than two
  mutable bool _restricted = false;
};

/**
 * @class JKGradientBuilder
 * @brief QDK/Chemistry's J/K gradient builder ("qdk").
 *
 * Evaluates the integral derivatives of the Coulomb, erf, and erfc kernels
 * directly with Libint2 and keeps the integral screening data of each kernel
 * of the last basis set for later runs.
 */
class JKGradientBuilder : public algorithms::JKGradientBuilder {
 public:
  /**
   * @brief Default constructor
   */
  JKGradientBuilder();

  /**
   * @brief Destructor
   */
  ~JKGradientBuilder() override;

  /**
   * @brief Access the builder's name
   *
   * @return "qdk"
   */
  std::string name() const override { return "qdk"; }

 protected:
  /**
   * @brief Build the nuclear gradients of E_J and E_K
   *
   * @param basis_set Basis set of the density matrices
   * @param j_kernel Kernel of J, or null to skip E_J
   * @param k_kernel Kernel of K, or null to skip E_K
   * @param densities Density matrices, one per spin block
   * @return The gradients of E_J and E_K on the basis set's structure, each
   * null if skipped
   * @throws std::invalid_argument If a kernel is not supported, or the
   * densities are not blocked by spin or are not matrices in the atomic
   * orbitals of the basis set
   */
  std::pair<std::shared_ptr<data::NuclearGradients>,
            std::shared_ptr<data::NuclearGradients>>
  _run_impl(std::shared_ptr<data::BasisSet> basis_set,
            std::shared_ptr<const data::TwoBodyKernel> j_kernel,
            std::shared_ptr<const data::TwoBodyKernel> k_kernel,
            std::shared_ptr<const data::SymmetryBlockedTensor<2>> densities)
      const override;

 private:
  /// Basis set of the cached integral engine
  mutable std::shared_ptr<data::BasisSet> _basis_set;
  /// The cached basis set in the format of the integral engine
  mutable std::shared_ptr<scf::BasisSet> _scf_basis_set;
  /// Integral engines of the kernels of the cached basis set
  mutable KernelEngines _engines;
  /// Whether the engines take one density matrix rather than two
  mutable bool _restricted = false;
};

}  // namespace qdk::chemistry::algorithms::microsoft
