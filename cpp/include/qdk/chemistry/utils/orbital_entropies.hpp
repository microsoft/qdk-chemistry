// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <macis/util/entropies.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <vector>

namespace qdk::chemistry::utils::orbital_entropies {

/// Callable that maps an eigenvalue spectrum to a scalar entropy value.
using EntropyFunction = std::function<double(const std::vector<double>&)>;

// ---- Factory functions for predefined entropies ----

/**
 * @brief Create a von Neumann entropy functor.
 *
 * S = -sum_k lambda_k * ln(lambda_k)
 */
inline EntropyFunction von_neumann_entropy() {
  return [](const std::vector<double>& eigenvalues) {
    constexpr double eps = std::numeric_limits<double>::epsilon();
    double s = 0.0;
    for (double v : eigenvalues) {
      if (v > eps) s -= v * std::log(v);
    }
    return s;
  };
}

/**
 * @brief Create a Rényi entropy functor of order alpha.
 *
 * S_alpha = 1/(1 - alpha) * ln(sum_k lambda_k^alpha)
 *
 * @param alpha  Rényi order (must not be 1).
 */
inline EntropyFunction renyi_entropy(double alpha) {
  return [alpha](const std::vector<double>& eigenvalues) {
    constexpr double eps = std::numeric_limits<double>::epsilon();
    double sum = 0.0;
    for (double v : eigenvalues) {
      if (v > eps) sum += std::pow(v, alpha);
    }
    if (sum <= 0.0) return 0.0;
    return std::log(sum) / (1.0 - alpha);
  };
}

/**
 * @brief Create a min-entropy functor.
 *
 * S_min = -ln(max_k lambda_k)
 */
inline EntropyFunction min_entropy() {
  return [](const std::vector<double>& eigenvalues) {
    constexpr double eps = std::numeric_limits<double>::epsilon();
    double max_val = 0.0;
    for (double v : eigenvalues) {
      if (v > max_val) max_val = v;
    }
    if (max_val <= eps) return 0.0;
    return -std::log(max_val);
  };
}

/**
 * @brief Create a max-entropy functor.
 *
 * S_max = ln(|{k : lambda_k > 0}|)
 *
 * Counts the number of non-zero eigenvalues.
 */
inline EntropyFunction max_entropy() {
  return [](const std::vector<double>& eigenvalues) {
    constexpr double eps = std::numeric_limits<double>::epsilon();
    int count = 0;
    for (double v : eigenvalues) {
      if (v > eps) ++count;
    }
    if (count <= 0) return 0.0;
    return std::log(static_cast<double>(count));
  };
}

// ---- Implementation details ----

namespace detail {

/**
 * @brief Extract the 16 eigenvalues of a 16×16 two-orbital RDM block.
 *
 * @param elem  Accessor `elem(p, q)` returning element (p,q) of the block.
 */
template <typename ElemFunc>
inline std::vector<double> extract_two_orbital_eigenvalues(ElemFunc&& elem) {
  auto eigenvalues_2x2 = [](double a, double b, double d) {
    double half_sum = 0.5 * (a + d);
    double half_diff = 0.5 * (a - d);
    double w = std::sqrt(half_diff * half_diff + b * b);
    return std::array<double, 2>{half_sum - w, half_sum + w};
  };

  std::vector<double> eigs;
  eigs.reserve(16);

  eigs.push_back(elem(0, 0));

  {
    auto e = eigenvalues_2x2(elem(1, 1), elem(1, 2), elem(2, 2));
    eigs.push_back(e[0]);
    eigs.push_back(e[1]);
  }
  {
    auto e = eigenvalues_2x2(elem(3, 3), elem(3, 4), elem(4, 4));
    eigs.push_back(e[0]);
    eigs.push_back(e[1]);
  }

  eigs.push_back(elem(5, 5));
  eigs.push_back(elem(6, 6));

  {
    double block[4][4];
    for (int p = 0; p < 4; ++p)
      for (int q = 0; q < 4; ++q) block[p][q] = elem(7 + p, 7 + q);
    auto e = macis::detail::eigenvalues_4x4(block);
    for (auto v : e) eigs.push_back(v);
  }

  {
    auto e = eigenvalues_2x2(elem(11, 11), elem(11, 12), elem(12, 12));
    eigs.push_back(e[0]);
    eigs.push_back(e[1]);
  }
  {
    auto e = eigenvalues_2x2(elem(13, 13), elem(13, 14), elem(14, 14));
    eigs.push_back(e[0]);
    eigs.push_back(e[1]);
  }

  eigs.push_back(elem(15, 15));

  return eigs;
}

}  // namespace detail

// ---- Low-level API (raw eigenvalue data) ----

/**
 * @brief Compute single-orbital entropies from eigenvalue matrix.
 *
 * Each row of eigenvalues contains the spectrum for that orbital.
 *
 * @param eigenvalues   norb × n_eig matrix of eigenvalues per orbital.
 * @param entropy_func  Entropy measure to apply to each spectrum.
 * @return Vector of norb entropies.
 */
inline Eigen::VectorXd build_single_orbital_entropies(
    const Eigen::MatrixXd& eigenvalues, const EntropyFunction& entropy_func) {
  const Eigen::Index norb = eigenvalues.rows();
  const Eigen::Index ncols = eigenvalues.cols();
  Eigen::VectorXd s1(norb);
  std::vector<double> eigs(ncols);
  for (Eigen::Index i = 0; i < norb; ++i) {
    for (Eigen::Index k = 0; k < ncols; ++k) eigs[k] = eigenvalues(i, k);
    s1(i) = entropy_func(eigs);
  }
  return s1;
}

/**
 * @brief Compute two-orbital entropies from eigenvalue matrix.
 *
 * Row (i*norb + j) holds the eigenvalue spectrum for pair (i, j).
 * Only upper-triangle entries (j > i) are read; result is symmetrized.
 *
 * @param pair_eigenvalues  (norb*norb) × n_eig matrix.
 * @param norb              Number of orbitals.
 * @param entropy_func      Entropy measure to apply to each spectrum.
 * @return norb × norb symmetric entropy matrix.
 */
inline Eigen::MatrixXd build_two_orbital_entropies(
    const Eigen::MatrixXd& pair_eigenvalues, Eigen::Index norb,
    const EntropyFunction& entropy_func) {
  const Eigen::Index ncols = pair_eigenvalues.cols();
  Eigen::MatrixXd s2 = Eigen::MatrixXd::Zero(norb, norb);
  std::vector<double> eigs(ncols);
  for (Eigen::Index i = 0; i < norb; ++i) {
    for (Eigen::Index j = i + 1; j < norb; ++j) {
      Eigen::Index row = i * norb + j;
      for (Eigen::Index k = 0; k < ncols; ++k)
        eigs[k] = pair_eigenvalues(row, k);
      s2(i, j) = entropy_func(eigs);
      s2(j, i) = s2(i, j);
    }
  }
  return s2;
}

/**
 * @brief Compute two-orbital entropies from the 2-orbital RDM tensor.
 *
 * Convenience overload for the MACIS orbital RDM format: a flat column-major
 * tensor of shape (norb, norb, 16, 16).
 *
 * @param two_ordm      Flat vector of size norb*norb*16*16.
 * @param norb          Number of orbitals.
 * @param entropy_func  Entropy measure to apply to each spectrum.
 * @return norb × norb symmetric entropy matrix.
 */
inline Eigen::MatrixXd build_two_orbital_entropies(
    const Eigen::VectorXd& two_ordm, Eigen::Index norb,
    const EntropyFunction& entropy_func) {
  Eigen::MatrixXd s2 = Eigen::MatrixXd::Zero(norb, norb);
  for (Eigen::Index i = 0; i < norb; ++i) {
    for (Eigen::Index j = i + 1; j < norb; ++j) {
      auto elem = [&](int p, int q) -> double {
        return two_ordm(i + norb * (j + norb * (p + 16 * q)));
      };
      auto eigs = detail::extract_two_orbital_eigenvalues(elem);
      s2(i, j) = entropy_func(eigs);
      s2(j, i) = s2(i, j);
    }
  }
  return s2;
}

/**
 * @brief Compute mutual information from single- and two-orbital entropies.
 *
 * I(i,j) = S_1(i) + S_1(j) - S_2(i,j)
 *
 * @param s1  Vector of norb single-orbital entropies.
 * @param s2  norb × norb matrix of two-orbital entropies.
 * @return norb × norb symmetric mutual information matrix.
 */
inline Eigen::MatrixXd build_mutual_information(const Eigen::VectorXd& s1,
                                                const Eigen::MatrixXd& s2) {
  const Eigen::Index norb = s1.size();
  Eigen::MatrixXd mi = Eigen::MatrixXd::Zero(norb, norb);
  for (Eigen::Index i = 0; i < norb; ++i) {
    for (Eigen::Index j = i + 1; j < norb; ++j) {
      mi(i, j) = s1(i) + s1(j) - s2(i, j);
      mi(j, i) = mi(i, j);
    }
  }
  return mi;
}

// ---- Wavefunction API (primary user-facing) ----

/**
 * @brief Compute single-orbital entropies from a wavefunction.
 *
 * Requires that the wavefunction has 1-orbital RDMs available.
 *
 * @param wavefunction  Wavefunction with orbital RDM data.
 * @param entropy_func  Entropy measure (default: von Neumann).
 * @return Vector of norb single-orbital entropies.
 * @throws std::runtime_error if 1-orbital RDMs are not available.
 */
inline Eigen::VectorXd build_single_orbital_entropies(
    const data::Wavefunction& wavefunction,
    const EntropyFunction& entropy_func = von_neumann_entropy()) {
  if (!wavefunction.has_one_orbital_rdm()) {
    throw std::runtime_error(
        "Wavefunction does not have 1-orbital RDMs. "
        "Request calculate_one_orbital_rdm=true when running the solver.");
  }
  return build_single_orbital_entropies(wavefunction.get_one_orbital_rdm(),
                                        entropy_func);
}

/**
 * @brief Compute two-orbital entropies from a wavefunction.
 *
 * Requires that the wavefunction has 2-orbital RDMs available.
 *
 * @param wavefunction  Wavefunction with orbital RDM data.
 * @param entropy_func  Entropy measure (default: von Neumann).
 * @return norb × norb symmetric entropy matrix.
 * @throws std::runtime_error if 2-orbital RDMs are not available.
 */
inline Eigen::MatrixXd build_two_orbital_entropies(
    const data::Wavefunction& wavefunction,
    const EntropyFunction& entropy_func = von_neumann_entropy()) {
  if (!wavefunction.has_two_orbital_rdm()) {
    throw std::runtime_error(
        "Wavefunction does not have 2-orbital RDMs. "
        "Request calculate_two_orbital_rdm=true when running the solver.");
  }
  if (!wavefunction.has_one_orbital_rdm()) {
    throw std::runtime_error(
        "Wavefunction does not have 1-orbital RDMs (needed for norb). "
        "Request calculate_one_orbital_rdm=true when running the solver.");
  }
  Eigen::Index norb = wavefunction.get_one_orbital_rdm().rows();
  return build_two_orbital_entropies(wavefunction.get_two_orbital_rdm(), norb,
                                     entropy_func);
}

/**
 * @brief Compute mutual information from a wavefunction.
 *
 * Requires that the wavefunction has both 1- and 2-orbital RDMs.
 *
 * @param wavefunction  Wavefunction with orbital RDM data.
 * @param entropy_func  Entropy measure (default: von Neumann).
 * @return norb × norb symmetric mutual information matrix.
 */
inline Eigen::MatrixXd build_mutual_information(
    const data::Wavefunction& wavefunction,
    const EntropyFunction& entropy_func = von_neumann_entropy()) {
  auto s1 = build_single_orbital_entropies(wavefunction, entropy_func);
  auto s2 = build_two_orbital_entropies(wavefunction, entropy_func);
  return build_mutual_information(s1, s2);
}

}  // namespace qdk::chemistry::utils::orbital_entropies
