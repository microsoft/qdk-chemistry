// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <vector>

namespace qdk::chemistry::scf {
class BasisSet;
}

namespace qdk::chemistry::data {
class BasisSet;
}

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {

/**
 * @brief Check whether sorted, unique orbital indices contain no gaps.
 * @param indices Sorted, unique orbital indices.
 * @return True when the sequence is empty or contiguous.
 */
inline bool indices_are_contiguous(const std::vector<std::size_t>& indices) {
  return indices.empty() ||
         indices.back() - indices.front() + 1 == indices.size();
}

std::pair<std::shared_ptr<qdk::chemistry::scf::BasisSet>, Eigen::MatrixXd>
build_one_body_ao(const data::BasisSet& basis_set,
                  const std::string& integral_dressing);

std::shared_ptr<data::Hamiltonian> construct_canonical_hamiltonian(
    std::shared_ptr<data::Orbitals> orbitals,
    const std::shared_ptr<qdk::chemistry::scf::BasisSet>& internal_basis_set,
    const Eigen::MatrixXd& one_body_ao, const std::string& eri_method);

}  // namespace detail

class HamiltonianSettings : public qdk::chemistry::data::Settings {
 public:
  HamiltonianSettings() {
    set_default("integral_dressing", std::string(""),
                "One-electron integral dressing: '' for nonrelativistic, "
                "'x2c_1e' for decontracted X2C-1e, or "
                "'x2c_1e_contracted' for X2C-1e in the contracted basis",
                data::ListConstraint<std::string>{{std::vector<std::string>{
                    "", "x2c_1e", "x2c_1e_contracted"}}});
  }
  ~HamiltonianSettings() override = default;
};

class CanonicalHamiltonianSettings : public HamiltonianSettings {
 public:
  CanonicalHamiltonianSettings() {
    set_default("eri_method", std::string("direct"),
                "ERI evaluation method: 'direct' computes integrals "
                "on-the-fly, 'incore' stores all integrals in memory",
                data::ListConstraint<std::string>{
                    {std::vector<std::string>{"direct", "incore"}}});
  }
  ~CanonicalHamiltonianSettings() override = default;
};

class HamiltonianConstructor
    : public qdk::chemistry::algorithms::HamiltonianConstructor {
 public:
  HamiltonianConstructor() {
    _settings = std::make_unique<CanonicalHamiltonianSettings>();
  };
  ~HamiltonianConstructor() override = default;

  virtual std::string name() const final { return "qdk"; };

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Orbitals> orbitals) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
