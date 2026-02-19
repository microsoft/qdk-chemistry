// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <tuple>
#include <type_traits>
#include <vector>

#include "qdk/chemistry/constants.hpp"
#include "qdk/chemistry/data/hamiltonian_containers/model_hamil.hpp"
#include "qdk/chemistry/data/lattice_graph.hpp"

namespace qdk::chemistry::utils::model_hamiltonians {

using qdk::chemistry::data::Hamiltonian;
using qdk::chemistry::data::LatticeGraph;
using qdk::chemistry::data::ModelHamiltonianContainer;

namespace detail {
/**
 * @brief True if T (after decay) is double or Eigen::VectorXd — valid per-site
 * parameter.
 */
template <typename T>
constexpr bool is_site_param_v =
    std::is_same_v<std::decay_t<T>, double> ||
    std::is_same_v<std::decay_t<T>, Eigen::VectorXd>;

/**
 * @brief True if T (after decay) is double or Eigen::MatrixXd — valid per-pair
 * parameter.
 */
template <typename T>
constexpr bool is_pair_param_v =
    std::is_same_v<std::decay_t<T>, double> ||
    std::is_same_v<std::decay_t<T>, Eigen::MatrixXd>;

/**
 * @brief Convert a per-site parameter to VectorXd. Pass-through for VectorXd.
 */
inline const Eigen::VectorXd& to_site_param(const Eigen::VectorXd& v, int) {
  return v;
}
/**
 * @brief Convert a scalar per-site parameter to a constant VectorXd of size n.
 */
inline Eigen::VectorXd to_site_param(double val, int n) {
  return Eigen::VectorXd::Constant(n, val);
}
/**
 * @brief Convert a per-pair parameter to MatrixXd. Pass-through for MatrixXd.
 */
inline const Eigen::MatrixXd& to_pair_param(const Eigen::MatrixXd& m, int) {
  return m;
}
/** @brief Convert a scalar per-pair parameter to a constant n x n MatrixXd. */
inline Eigen::MatrixXd to_pair_param(double val, int n) {
  return Eigen::MatrixXd::Constant(n, n, val);
}

/**
 * @brief e^2 / (4 pi epsilon_0), used in Ohno and Mataga-Nishimoto potentials.
 */
constexpr double COULOMB_CONSTANT =
    constants::hartree_to_ev * constants::bohr_to_angstrom;

/**
 * @brief Construct a Hückel Hamiltonian on a lattice.
 *
 * Builds the one-body Hamiltonian:
 *   H = sum_i epsilon_i n_i - sum_{<i,j>} t_ij (a_i^dag a_j + a_j^dag a_i)
 * where n_i = sum_sigma a_{i,sigma}^dag a_{i,sigma} and the sum over <i,j>
 * runs over edges of the lattice graph.
 *
 * @tparam EpsT  double or Eigen::VectorXd
 * @tparam TT    double or Eigen::MatrixXd
 * @param lattice  Symmetric lattice graph defining the connectivity.
 * @param epsilon_in  On-site orbital energies. Scalar or VectorXd of size n.
 * @param t_in  Hopping integrals. Scalar or n x n MatrixXd.
 * @return Sparse one-body integral matrix (n x n).
 * @throws std::invalid_argument if dimensions mismatch, lattice is asymmetric,
 * or empty.
 */
template <typename EpsT, typename TT>
inline Eigen::SparseMatrix<double> from_huckel(const LatticeGraph& lattice,
                                               EpsT&& epsilon_in, TT&& t_in) {
  // Check template types
  static_assert(detail::is_site_param_v<EpsT>,
                "epsilon must be double or Eigen::VectorXd");
  static_assert(detail::is_pair_param_v<TT>,
                "t must be double or Eigen::MatrixXd");

  auto n = static_cast<int>(lattice.num_sites());
  const auto& epsilon =
      detail::to_site_param(std::forward<EpsT>(epsilon_in), n);
  const auto& t = detail::to_pair_param(std::forward<TT>(t_in), n);

  // Check dimensions and lattice symmetry
  if (n != epsilon.size()) {
    throw std::invalid_argument(
        "Epsilon vector size must match the number of lattice sites.");
  }
  if (n != t.rows() || n != t.cols()) {
    throw std::invalid_argument(
        "T matrix dimensions must match the number of lattice sites.");
  }
  if (lattice.symmetry() == false) {
    throw std::invalid_argument(
        "Lattice graph must be symmetric for a valid Hamiltonian.");
  }
  if (n == 0) {
    throw std::invalid_argument("Lattice graph must have at least one site.");
  }

  // Build sparse one-body integrals from triplets
  using T = Eigen::Triplet<double>;
  std::vector<T> triplets;
  triplets.reserve(n + lattice.num_nonzeros());

  // Diagonal: on-site energies
  for (int i = 0; i < n; i++) {
    if (epsilon(i) != 0.0) {
      triplets.emplace_back(i, i, epsilon(i));
    }
  }

  // Off-diagonal: hopping along lattice edges
  const auto& adj = lattice.sparse_adjacency_matrix();
  for (int k = 0; k < adj.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adj, k); it; ++it) {
      int i = static_cast<int>(it.row());
      int j = static_cast<int>(it.col());
      if (i < j) {
        double adj_val = it.value();
        triplets.emplace_back(i, j, -t(i, j) * adj_val);
        triplets.emplace_back(j, i, -t(j, i) * adj_val);
      }
    }
  }

  Eigen::SparseMatrix<double> h1(n, n);
  h1.setFromTriplets(triplets.begin(), triplets.end());
  h1.makeCompressed();

  return h1;
}

/**
 * @brief Construct a Hubbard Hamiltonian on a lattice.
 *
 * Extends the Hückel model with on-site electron-electron repulsion:
 *   H = H_huckel + U sum_i n_{i,up} n_{i,down}
 *
 * @tparam EpsT  double or Eigen::VectorXd
 * @tparam TT    double or Eigen::MatrixXd
 * @tparam UT    double or Eigen::VectorXd
 * @param lattice  Symmetric lattice graph defining the connectivity.
 * @param epsilon_in  On-site orbital energies. Scalar or VectorXd of size n.
 * @param t_in  Hopping integrals. Scalar or n x n MatrixXd.
 * @param U_in  On-site Coulomb repulsion. Scalar or VectorXd of size n.
 * @return Tuple of (sparse one-body matrix, two-body map).
 * @throws std::invalid_argument if U vector size mismatches the number of
 * sites.
 */
template <typename EpsT, typename TT, typename UT>
inline std::tuple<Eigen::SparseMatrix<double>,
                  ModelHamiltonianContainer::TwoBodyMap>
from_hubbard(const LatticeGraph& lattice, EpsT&& epsilon_in, TT&& t_in,
             UT&& U_in) {
  // Check template types
  static_assert(detail::is_site_param_v<EpsT>,
                "epsilon must be double or Eigen::VectorXd");
  static_assert(detail::is_pair_param_v<TT>,
                "t must be double or Eigen::MatrixXd");
  static_assert(detail::is_site_param_v<UT>,
                "U must be double or Eigen::VectorXd");

  // check dimensions and lattice symmetry
  auto n = static_cast<int>(lattice.num_sites());
  const auto& epsilon =
      detail::to_site_param(std::forward<EpsT>(epsilon_in), n);
  const auto& t = detail::to_pair_param(std::forward<TT>(t_in), n);
  const auto& U = detail::to_site_param(std::forward<UT>(U_in), n);
  if (n != U.size()) {
    throw std::invalid_argument(
        "U vector size must match the number of lattice sites.");
  }

  // Build the one-body part using the Hückel constructor
  auto h1 = from_huckel(lattice, epsilon, t);

  // Build the two-body map for on-site repulsion: (p,q,r,s) = (i,i,i,i) -> U_i
  ModelHamiltonianContainer::TwoBodyMap h2;
  for (int i = 0; i < n; i++) {
    double U_i = U(i);
    if (U_i != 0.0) {
      h2[{i, i, i, i}] = U_i;
    }
  }

  return {std::move(h1), std::move(h2)};
}

/**
 * @brief Construct a Pariser-Parr-Pople (PPP) Hamiltonian on a lattice.
 *
 * Extends the Hubbard model with long-range intersite Coulomb interactions:
 *   H = H_hubbard + 1/2 sum_{i!=j} V_ij (n_i - z_i)(n_j - z_j)
 *
 * @tparam EpsT  double or Eigen::VectorXd
 * @tparam TT    double or Eigen::MatrixXd
 * @tparam UT    double or Eigen::VectorXd
 * @tparam VT    double or Eigen::MatrixXd
 * @tparam ZT    double or Eigen::VectorXd
 * @param lattice  Symmetric lattice graph defining the connectivity.
 * @param epsilon_in  On-site orbital energies. Scalar or VectorXd of size n.
 * @param t_in  Hopping integrals. Scalar or n x n MatrixXd.
 * @param U_in  On-site Coulomb repulsion. Scalar or VectorXd of size n.
 * @param V_in  Intersite Coulomb interaction matrix. Scalar or n x n MatrixXd.
 * @param z_in  Effective core charges. Scalar or VectorXd of size n.
 * @return Tuple of (sparse one-body matrix, two-body map, energy offset).
 * @throws std::invalid_argument if V or z dimensions mismatch.
 */
template <typename EpsT, typename TT, typename UT, typename VT, typename ZT>
inline std::tuple<Eigen::SparseMatrix<double>,
                  ModelHamiltonianContainer::TwoBodyMap, double>
from_ppp(const LatticeGraph& lattice, EpsT&& epsilon_in, TT&& t_in, UT&& U_in,
         VT&& V_in, ZT&& z_in) {
  // Check template types
  static_assert(detail::is_site_param_v<EpsT>,
                "epsilon must be double or Eigen::VectorXd");
  static_assert(detail::is_pair_param_v<TT>,
                "t must be double or Eigen::MatrixXd");
  static_assert(detail::is_site_param_v<UT>,
                "U must be double or Eigen::VectorXd");
  static_assert(detail::is_pair_param_v<VT>,
                "V must be double or Eigen::MatrixXd");
  static_assert(detail::is_site_param_v<ZT>,
                "z must be double or Eigen::VectorXd");

  // Check dimensions and lattice symmetry
  auto n = static_cast<int>(lattice.num_sites());
  const auto& epsilon =
      detail::to_site_param(std::forward<EpsT>(epsilon_in), n);
  const auto& t = detail::to_pair_param(std::forward<TT>(t_in), n);
  const auto& U = detail::to_site_param(std::forward<UT>(U_in), n);
  const auto& V = detail::to_pair_param(std::forward<VT>(V_in), n);
  const auto& z = detail::to_site_param(std::forward<ZT>(z_in), n);
  if (n != V.rows() || n != V.cols()) {
    throw std::invalid_argument(
        "V matrix dimensions must match the number of lattice sites.");
  }
  if (n != z.size()) {
    throw std::invalid_argument(
        "z vector size must match the number of lattice sites.");
  }

  // Build the Hubbard part using the from_hubbard constructor
  auto [h1_sparse, h2] = from_hubbard(lattice, epsilon, t, U);

  // Convert to dense for efficient diagonal modification
  // (avoids costly insertions into compressed sparse)
  Eigen::MatrixXd h1_dense = Eigen::MatrixXd(h1_sparse);
  double energy_offset = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      double V_ij = 0.5 * V(i, j);
      if (V_ij == 0.0) continue;

      // two-body: (ii|jj) = V_ij for both orderings
      h2[{i, i, j, j}] += V_ij;

      // one-body correction: -sum_{j!=i} V_ij z_j applied to h_ii
      h1_dense(i, i) -= V_ij * z(j);
      h1_dense(j, j) -= V_ij * z(i);

      // scalar offset: 1/2 sum_{i!=j} V_ij z_i z_j = sum_{i<j} V_ij z_i z_j
      energy_offset += V_ij * z(i) * z(j);
    }
  }
  Eigen::SparseMatrix<double> h1 = h1_dense.sparseView();
  h1.makeCompressed();
  return {std::move(h1), std::move(h2), energy_offset};
}

}  // namespace detail

/**
 * @brief Create a Hückel model Hamiltonian.
 *
 * @tparam EpsT  double or Eigen::VectorXd
 * @tparam TT    double or Eigen::MatrixXd
 * @param lattice  Symmetric lattice graph defining the connectivity.
 * @param epsilon_in  On-site orbital energies. Scalar or VectorXd of size n.
 * @param t_in  Hopping integrals. Scalar or n x n MatrixXd.
 * @return Hamiltonian for the Hückel model.
 */
template <typename EpsT, typename TT>
inline Hamiltonian create_huckel_hamiltonian(const LatticeGraph& lattice,
                                             EpsT&& epsilon_in, TT&& t_in) {
  auto h1 = detail::from_huckel(lattice, std::forward<EpsT>(epsilon_in),
                                std::forward<TT>(t_in));
  return Hamiltonian(
      std::make_unique<ModelHamiltonianContainer>(std::move(h1)));
}

/**
 * @brief Create a Hubbard model Hamiltonian.
 *
 * @tparam EpsT  double or Eigen::VectorXd
 * @tparam TT    double or Eigen::MatrixXd
 * @tparam UT    double or Eigen::VectorXd
 * @param lattice  Symmetric lattice graph defining the connectivity.
 * @param epsilon_in  On-site orbital energies. Scalar or VectorXd of size n.
 * @param t_in  Hopping integrals. Scalar or n x n MatrixXd.
 * @param U_in  On-site Coulomb repulsion. Scalar or VectorXd of size n.
 * @return Hamiltonian for the Hubbard model.
 */
template <typename EpsT, typename TT, typename UT>
inline Hamiltonian create_hubbard_hamiltonian(const LatticeGraph& lattice,
                                              EpsT&& epsilon_in, TT&& t_in,
                                              UT&& U_in) {
  auto [h1, h2] =
      detail::from_hubbard(lattice, std::forward<EpsT>(epsilon_in),
                           std::forward<TT>(t_in), std::forward<UT>(U_in));
  return Hamiltonian(std::make_unique<ModelHamiltonianContainer>(
      std::move(h1), std::move(h2)));
}

/**
 * @brief Create a Pariser-Parr-Pople (PPP) model Hamiltonian.
 *
 * @tparam EpsT  double or Eigen::VectorXd
 * @tparam TT    double or Eigen::MatrixXd
 * @tparam UT    double or Eigen::VectorXd
 * @tparam VT    double or Eigen::MatrixXd
 * @tparam ZT    double or Eigen::VectorXd
 * @param lattice  Symmetric lattice graph defining the connectivity.
 * @param epsilon_in  On-site orbital energies. Scalar or VectorXd of size n.
 * @param t_in  Hopping integrals. Scalar or n x n MatrixXd.
 * @param U_in  On-site Coulomb repulsion. Scalar or VectorXd of size n.
 * @param V_in  Intersite Coulomb interaction matrix. Scalar or n x n MatrixXd.
 * @param z_in  Effective core charges. Scalar or VectorXd of size n.
 * @return Hamiltonian for the PPP model.
 */
template <typename EpsT, typename TT, typename UT, typename VT, typename ZT>
inline Hamiltonian create_ppp_hamiltonian(const LatticeGraph& lattice,
                                          EpsT&& epsilon_in, TT&& t_in,
                                          UT&& U_in, VT&& V_in, ZT&& z_in) {
  auto [h1, h2, core_energy] = detail::from_ppp(
      lattice, std::forward<EpsT>(epsilon_in), std::forward<TT>(t_in),
      std::forward<UT>(U_in), std::forward<VT>(V_in), std::forward<ZT>(z_in));
  return Hamiltonian(std::make_unique<ModelHamiltonianContainer>(
      std::move(h1), std::move(h2), core_energy));
}

/**
 * @brief Compute a symmetric pairwise potential matrix from a custom formula.
 *
 * For each unique pair (i < j), computes the geometric mean
 * U_ij = sqrt(U_i * U_j), reads R_ij, and evaluates func(i, j, U_ij, R_ij).
 * The result is stored symmetrically: V(i,j) = V(j,i).
 *
 * When nearest_neighbor_only is true, only pairs connected by a lattice edge
 * are evaluated; all other entries remain zero.
 *
 * @tparam UT  double or Eigen::VectorXd — on-site Coulomb parameter(s).
 * @tparam RT  double or Eigen::MatrixXd — intersite distances.
 * @tparam PotentialFunc  Callable with signature (int i, int j, double Uij,
 * double Rij) -> double.
 * @param lattice  Lattice graph defining the connectivity and number of sites.
 * @param U_in  On-site Coulomb parameter(s). Scalar or VectorXd of size n.
 * @param R_in  Distance matrix. Scalar or n x n MatrixXd.
 * @param func  Potential formula to evaluate for each pair.
 * @param nearest_neighbor_only  If true, restrict to lattice-connected pairs
 * (default false).
 * @return n x n symmetric MatrixXd of pairwise potential values.
 * @throws std::invalid_argument if U size or R dimensions mismatch.
 */
template <typename UT, typename RT, typename PotentialFunc>
inline Eigen::MatrixXd pairwise_potential(const LatticeGraph& lattice,
                                          UT&& U_in, RT&& R_in,
                                          PotentialFunc&& func,
                                          bool nearest_neighbor_only = false) {
  // Check template types
  static_assert(detail::is_site_param_v<UT>,
                "U must be double or Eigen::VectorXd");
  static_assert(detail::is_pair_param_v<RT>,
                "R must be double or Eigen::MatrixXd");
  static_assert(
      std::is_invocable_r_v<double, PotentialFunc, int, int, double, double>,
      "func must be callable as double(int, int, double, double)");

  // Check dimensions and lattice symmetry
  auto n = static_cast<int>(lattice.num_sites());
  const auto& U = detail::to_site_param(std::forward<UT>(U_in), n);
  const auto& R = detail::to_pair_param(std::forward<RT>(R_in), n);
  if (n != U.size()) {
    throw std::invalid_argument(
        "U vector size must match the number of lattice sites.");
  }
  if (n != R.rows() || n != R.cols()) {
    throw std::invalid_argument(
        "R matrix dimensions must match the number of lattice sites.");
  }

  // Compute pairwise potential matrix
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      // Skip non-neighbor pairs when nearest_neighbor_only is set
      if (nearest_neighbor_only && !lattice.are_connected(i, j)) continue;
      double Uij = std::sqrt(U(i) * U(j));
      double Rij = R(i, j);
      double Vij = func(i, j, Uij, Rij);
      V(i, j) = Vij;
      V(j, i) = Vij;
    }
  }
  return V;
}

/**
 * @brief Compute the Ohno intersite potential matrix.
 *
 * V_ij = U_ij / sqrt(1 + (U_ij * epsilon_r * R_ij / constant)^2)
 *
 * where U_ij = sqrt(U_i * U_j) is the geometric mean of on-site parameters.
 * The constant is e^2 / (4 pi epsilon_0).
 *
 * @tparam UT  double or Eigen::VectorXd
 * @tparam RT  double or Eigen::MatrixXd
 * @param lattice  Lattice graph (used for the number of sites).
 * @param U  On-site Coulomb parameter(s). Scalar or VectorXd of size n.
 * @param R  Intersite distances. Scalar or n x n MatrixXd.
 * @param epsilon_r  Relative permittivity (dimensionless, default 1.0).
 * @param nearest_neighbor_only  If true, restrict to lattice-connected pairs
 * (default false).
 * @return n x n symmetric MatrixXd of Ohno potential values.
 */
template <typename UT, typename RT>
inline Eigen::MatrixXd ohno_potential(const LatticeGraph& lattice, UT&& U,
                                      RT&& R, double epsilon_r = 1.0,
                                      bool nearest_neighbor_only = false) {
  // Check template types
  static_assert(detail::is_site_param_v<UT>,
                "U must be double or Eigen::VectorXd");
  static_assert(detail::is_pair_param_v<RT>,
                "R must be double or Eigen::MatrixXd");

  // compute potential
  return pairwise_potential(
      lattice, std::forward<UT>(U), std::forward<RT>(R),
      [epsilon_r](int, int, double Uij, double Rij) {
        double x = Uij * epsilon_r * Rij / detail::COULOMB_CONSTANT;
        return Uij / std::sqrt(1.0 + x * x);
      },
      nearest_neighbor_only);
}

/**
 * @brief Compute the Mataga-Nishimoto intersite potential matrix.
 *
 * V_ij = U_ij / (1 + U_ij * epsilon_r * R_ij / constant)
 *
 * where U_ij = sqrt(U_i * U_j) is the geometric mean of on-site parameters.
 * The constant e^2 / (4 pi epsilon_0).
 *
 * @tparam UT  double or Eigen::VectorXd
 * @tparam RT  double or Eigen::MatrixXd
 * @param lattice  Lattice graph (used for the number of sites).
 * @param U  On-site Coulomb parameter(s). Scalar or VectorXd of size n.
 * @param R  Intersite distances. Scalar or n x n MatrixXd.
 * @param epsilon_r  Relative permittivity (dimensionless, default 1.0).
 * @param nearest_neighbor_only  If true, restrict to lattice-connected pairs
 * (default false).
 * @return n x n symmetric MatrixXd of Mataga-Nishimoto potential values.
 */
template <typename UT, typename RT>
inline Eigen::MatrixXd mataga_nishimoto_potential(
    const LatticeGraph& lattice, UT&& U, RT&& R, double epsilon_r = 1.0,
    bool nearest_neighbor_only = false) {
  // Check template types
  static_assert(detail::is_site_param_v<UT>,
                "U must be double or Eigen::VectorXd");
  static_assert(detail::is_pair_param_v<RT>,
                "R must be double or Eigen::MatrixXd");
  // compute potential
  return pairwise_potential(
      lattice, std::forward<UT>(U), std::forward<RT>(R),
      [epsilon_r](int, int, double Uij, double Rij) {
        return Uij / (1.0 + Uij * epsilon_r * Rij / detail::COULOMB_CONSTANT);
      },
      nearest_neighbor_only);
}
}  // namespace qdk::chemistry::utils::model_hamiltonians
