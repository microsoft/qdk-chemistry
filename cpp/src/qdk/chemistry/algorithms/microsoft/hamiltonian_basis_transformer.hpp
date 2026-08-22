// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <mutex>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/** QDK implementation for restricted Cholesky Hamiltonians. */
class QdkHamiltonianBasisTransformer final
    : public algorithms::HamiltonianBasisTransformer {
 public:
  QdkHamiltonianBasisTransformer();

  std::string name() const final { return "qdk"; }
  std::shared_ptr<data::Hamiltonian> run(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<data::Orbitals> target_orbitals) const final;

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<data::Orbitals> target_orbitals) const final;

 private:
  mutable std::mutex _run_mutex;
};

}  // namespace qdk::chemistry::algorithms::microsoft
