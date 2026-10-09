// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
//
// Portions of this file are adapted from code by Felix Rupprecht published at
// https://zenodo.org/records/20393500, Copyright 2026 German Aerospace Center
// (DLR), licensed under the Apache License, Version 2.0, and modified for QDK
// Chemistry.

#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "unitary_synthesis_detail.hpp"

namespace qdk::chemistry::utils::detail {

DenseSiteSynthesis dense_unitary_synthesis(
    const data::MPSSite& site, Eigen::Index ancilla_dim,
    const Eigen::MatrixXd& following_right_factor) {
  validate_site(site, ancilla_dim);
  const auto left = static_cast<Eigen::Index>(site.left_bond_dimension());
  const auto physical = static_cast<Eigen::Index>(site.physical_dimension());
  const auto right = static_cast<Eigen::Index>(site.right_bond_dimension());
  if (physical != 2 && physical != 4) {
    throw std::invalid_argument(
        "Dense site synthesis requires two or four physical states.");
  }
  if (following_right_factor.size() != 0) {
    if (following_right_factor.rows() != right ||
        following_right_factor.cols() != right ||
        !following_right_factor.allFinite()) {
      throw std::invalid_argument(
          "Dense site synthesis requires a finite successor factor matching "
          "the right bond.");
    }
    if (!is_isometry(following_right_factor)) {
      throw std::invalid_argument(
          "Dense site synthesis requires an orthogonal successor factor.");
    }
  } else if (following_right_factor.rows() != 0 ||
             following_right_factor.cols() != 0) {
    throw std::invalid_argument(
        "Dense site synthesis requires an empty or square successor factor.");
  }

  // Row p * ancilla_dim + b and column a hold M^p_{ab}; to_dense packs it at
  // row a * physical + p and column b.
  const auto dense = std::get<Eigen::MatrixXd>(site.to_dense());
  Eigen::MatrixXd isometry =
      Eigen::MatrixXd::Zero(physical * ancilla_dim, left);
  for (Eigen::Index a = 0; a < left; ++a) {
    for (Eigen::Index p = 0; p < physical; ++p) {
      isometry.block(p * ancilla_dim, a, right, 1) =
          dense.row(a * physical + p).transpose();
    }
  }
  if (!is_isometry(isometry)) {
    throw std::invalid_argument(
        "Dense site synthesis requires an isometric site.");
  }

  std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> blocks;
  if (physical == 2) {
    blocks.emplace_back(isometry.topRows(ancilla_dim),
                        isometry.bottomRows(ancilla_dim));
  } else {
    // o-block CSDs.Three-step CSD peel: two QRs split the four physical blocks
    // into three independent tw
    auto [b, r] =
        decompose_qr(isometry.bottomRows(3 * ancilla_dim), ancilla_dim);
    auto [c, s] = decompose_qr(b.bottomRows(2 * ancilla_dim), ancilla_dim);
    blocks.emplace_back(isometry.topRows(ancilla_dim), std::move(r));
    blocks.emplace_back(b.topRows(ancilla_dim), std::move(s));
    blocks.emplace_back(c.topRows(ancilla_dim), c.bottomRows(ancilla_dim));
  }
  std::vector<TwoBlockCsd> csds(blocks.size());
  run_tasks(
      static_cast<std::ptrdiff_t>(blocks.size()), [&](std::ptrdiff_t index) {
        const auto step = static_cast<std::size_t>(index);
        csds[step] = decompose_csd(blocks[step].first, blocks[step].second);
      });

  DenseSiteSynthesis result;
  std::vector<Eigen::MatrixXd> unitaries;
  for (const auto& csd : csds) {
    result.rotation_angles.push_back(
        rotation_angles(csd.d_1, csd.d_2, ancilla_dim));
  }
  if (csds.size() == 3) {
    unitaries.push_back(csds[1].v * csds[0].u_2);
    unitaries.push_back(csds[2].v * csds[1].u_2);
  }
  for (auto& csd : csds) {
    unitaries.push_back(std::move(csd.u_1));
  }
  unitaries.push_back(std::move(csds.back().u_2));
  if (following_right_factor.size() != 0) {
    // Rotating the right bond changes only the terminal blocks, not the CSD
    // angles, mixing unitaries, or this site's right factor.
    const auto num_blocks = static_cast<std::ptrdiff_t>(csds.size() + 1);
    for (auto block = unitaries.end() - num_blocks; block != unitaries.end();
         ++block) {
      block->topRows(right) = following_right_factor * block->topRows(right);
    }
  }
  result.right_factor = std::move(csds.front().v);

  std::vector<std::reference_wrapper<const Eigen::MatrixXd>> matrices;
  for (const auto& unitary : unitaries) {
    matrices.push_back(std::cref(unitary));
  }
  auto givens = decompose_unitaries_to_givens(matrices);
  const auto terminal = givens.begin() + (physical == 4 ? 2 : 0);
  result.mixing_givens.assign(std::make_move_iterator(givens.begin()),
                              std::make_move_iterator(terminal));
  result.block_givens = merge_block_givens(
      std::vector<GivensDecomposition>(std::make_move_iterator(terminal),
                                       std::make_move_iterator(givens.end())));
  return result;
}

MPSSynthesis matrix_product_state_synthesis(
    const data::MPSContainer& mps, Eigen::Index ancilla_dim,
    std::string_view unitary_synthesis) {
  if (unitary_synthesis != "dense" && unitary_synthesis != "block_sparse") {
    throw std::invalid_argument(
        "MPS unitary synthesis must be 'dense' or 'block_sparse'.");
  }
  const auto& sites = mps.sites();
  validate_site(*sites.front(), ancilla_dim);
  const auto count = sites.size() - 1;
  if (unitary_synthesis == "dense") {
    std::vector<DenseSiteSynthesis> results(count);
    for (std::size_t index = count; index > 0; --index) {
      const auto& site = *sites[index];
      const Eigen::MatrixXd empty;
      const auto& following =
          index < count ? results[index].right_factor : empty;
      results[index - 1] =
          dense_unitary_synthesis(site, ancilla_dim, following);
    }
    return results;
  }
  std::vector<SparseSiteSynthesis> results(count);
  run_tasks(static_cast<std::ptrdiff_t>(count), [&](std::ptrdiff_t index) {
    const auto position = static_cast<std::size_t>(index);
    const auto& site = *sites[position + 1];
    results[position] = block_sparse_unitary_synthesis(site, ancilla_dim);
  });
  return results;
}

void validate_site(const data::MPSSite& site, Eigen::Index ancilla_dim) {
  if (site.is_complex()) {
    throw std::invalid_argument(
        "MPS site synthesis requires a real site tensor.");
  }
  const auto left = static_cast<Eigen::Index>(site.left_bond_dimension());
  const auto physical = static_cast<Eigen::Index>(site.physical_dimension());
  const auto right = static_cast<Eigen::Index>(site.right_bond_dimension());
  if (ancilla_dim <= 0 || left > ancilla_dim || right > ancilla_dim ||
      ancilla_dim > std::numeric_limits<Eigen::Index>::max() / physical) {
    throw std::invalid_argument(
        "MPS site synthesis requires both bond dimensions to be at most the "
        "ancilla dimension.");
  }
}

}  // namespace qdk::chemistry::utils::detail
