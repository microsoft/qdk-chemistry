// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class DensityFittedHamiltonianSettings
 * @brief Settings for density-fitted Hamiltonian construction
 *
 * This class provides configuration options for the
 * DensityFittedHamiltonianConstructor algorithm. Currently empty but
 * reserved for future settings such as integral thresholds.
 */
class DensityFittedHamiltonianSettings : public qdk::chemistry::data::Settings {
 public:
  DensityFittedHamiltonianSettings() {}
  ~DensityFittedHamiltonianSettings() override = default;
};

/**
 * @class DensityFittedHamiltonianConstructor
 * @brief Constructs molecular Hamiltonians using density fitting approximation
 *
 * This class implements Hamiltonian construction using density fitting
 * (also known as resolution-of-the-identity, RI) to approximate two-electron
 * integrals. Instead of computing and storing full four-center integrals
 * (ij|kl), it computes three-center integrals (P|ij) where P indexes an
 * auxiliary basis set.
 *
 * The density fitting approximation expresses four-center integrals as:
 * @f[
 *   (ij|kl) \approx \sum_P (ij|P)(P|kl)
 * @f]
 *
 * This approach significantly reduces memory requirements from O(N^4) to
 * O(N_aux * N^2) where N is the number of molecular orbitals and N_aux is
 * the size of the auxiliary basis.
 *
 * @note An auxiliary basis set must be provided when calling run().
 *
 * @see DensityFittedHamiltonianContainer
 * @see HamiltonianConstructor
 */
class DensityFittedHamiltonianConstructor
    : public qdk::chemistry::algorithms::HamiltonianConstructor {
 public:
  DensityFittedHamiltonianConstructor() {
    _settings = std::make_unique<DensityFittedHamiltonianSettings>();
  }
  ~DensityFittedHamiltonianConstructor() override = default;

  virtual std::string name() const final { return "qdk_density_fitted"; }

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Orbitals> orbitals,
      OptionalAuxBasis aux_basis) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
