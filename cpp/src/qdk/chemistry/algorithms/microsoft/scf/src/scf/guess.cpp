// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "scf/guess.h"

#include <qdk/chemistry/scf/core/types.h>
#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "util/macros.h"

namespace qdk::chemistry::scf {
void atom_guess(const BasisSet& obs, const Molecule& mol, double* D) {
  std::filesystem::path guess_chk(std::filesystem::temp_directory_path() /
                                  "qdk" / "chemistry");
  if (obs.pure) {
    guess_chk = guess_chk / "guess" / (obs.name + "_pure");
  } else {
    guess_chk = guess_chk / "guess" / (obs.name + "_cart");
  }

  if (!std::filesystem::exists(guess_chk)) {
    spdlog::error(
        "{} not found, use `scripts/generate_guess.py` to prepare basis",
        obs.name);
    exit(EXIT_FAILURE);
  }
  std::map<int, RowMajorMatrix> atom_dm;
  std::ifstream fin(guess_chk);
  int atomic_number;
  while (fin >> atomic_number) {
    int n;
    fin >> n;
    RowMajorMatrix d(n, n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        fin >> d(i, j);
      }
    }
    atom_dm[atomic_number] = d;
  }
  int N = obs.num_basis_funcs;
  RowMajorMatrix tD = RowMajorMatrix::Zero(N, N);

  for (size_t i = 0, p = 0; i < mol.n_atoms; i++) {
    int z = mol.atomic_nums[i];
    VERIFY_INPUT(atom_dm.count(z),
                 fmt::format("No basis for Atom(Z={}) in {}", z, obs.name));
    const auto& d = atom_dm[z];
    tD.block(p, p, d.rows(), d.cols()) = d;
    p += d.rows();
  }

  std::vector<int> order(obs.shells.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int p1, int p2) {
    const auto& a = obs.shells[p1];
    const auto& b = obs.shells[p2];
    return a.atom_index != b.atom_index
               ? a.atom_index < b.atom_index
               : (a.angular_momentum != b.angular_momentum
                      ? a.angular_momentum < b.angular_momentum
                      : a.exponents[0] > b.exponents[0]);
  });

  auto shell_num_basis_funcs = [&](int am) {
    return obs.pure ? 2 * am + 1 : (am + 1) * (am + 2) / 2;
  };
  std::vector<int> sh2bf;
  for (size_t i = 0, p = 0; i < obs.shells.size(); i++) {
    sh2bf.push_back(p);
    int am = obs.shells[i].angular_momentum;
    p += shell_num_basis_funcs(am);
  }

  RowMajorMatrix nD = RowMajorMatrix::Zero(N, N);
  for (size_t i = 0, bf1 = 0; i < obs.shells.size(); i++) {
    int am1 = obs.shells[order[i]].angular_momentum;
    int n1 = shell_num_basis_funcs(am1);
    for (size_t j = 0, bf2 = 0; j < obs.shells.size(); j++) {
      int am2 = obs.shells[order[j]].angular_momentum;
      int n2 = shell_num_basis_funcs(am2);
      nD.block(sh2bf[order[i]], sh2bf[order[j]], n1, n2) =
          tD.block(bf1, bf2, n1, n2);
      bf2 += n2;
    }
    bf1 += n1;
  }
  memcpy(D, nD.data(), N * N * sizeof(double));
}
}  // namespace qdk::chemistry::scf
