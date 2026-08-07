// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Core>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"

namespace testing {

namespace sw = qdk::chemistry::algorithms::microsoft::swpt2;

struct SpinOrbitalTensors {
  double core_energy = 0.0;
  Eigen::MatrixXd one_body;
  Eigen::VectorXd two_body;
};

inline SpinOrbitalTensors build_spin_orbital_tensors(
    const Eigen::MatrixXd& h1a, const Eigen::MatrixXd& h1b,
    const Eigen::VectorXd& g_aaaa, const Eigen::VectorXd& g_aabb,
    const Eigen::VectorXd& g_bbbb, double core_energy, int norb) {
  const int n_so = 2 * norb;
  SpinOrbitalTensors tensors{core_energy,
                             sw::spin_orbital_one_body(h1a, h1b, norb),
                             Eigen::VectorXd::Zero(n_so * n_so * n_so * n_so)};

  std::vector<double> ordered(tensors.two_body.size(), 0.0);
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s) {
          const auto integral = sw::idx4(p, q, r, s, norb);
          ordered[sw::idx4(2 * p, 2 * r, 2 * s, 2 * q, n_so)] +=
              0.5 * g_aaaa(integral);
          ordered[sw::idx4(2 * p + 1, 2 * r + 1, 2 * s + 1, 2 * q + 1, n_so)] +=
              0.5 * g_bbbb(integral);
          ordered[sw::idx4(2 * p, 2 * r + 1, 2 * s + 1, 2 * q, n_so)] +=
              g_aabb(integral);
        }
  for (int p = 0; p < n_so; ++p)
    for (int q = 0; q < n_so; ++q)
      for (int r = 0; r < n_so; ++r)
        for (int s = 0; s < n_so; ++s)
          tensors.two_body(sw::idx4(p, q, r, s, n_so)) =
              ordered[sw::idx4(p, q, r, s, n_so)] -
              ordered[sw::idx4(q, p, r, s, n_so)] -
              ordered[sw::idx4(p, q, s, r, n_so)] +
              ordered[sw::idx4(q, p, s, r, n_so)];
  return tensors;
}

}  // namespace testing
