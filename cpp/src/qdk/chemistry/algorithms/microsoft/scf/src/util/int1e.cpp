// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/util/int1e.h>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif
#include <array>
#include <functional>
#include <iostream>
#include <qdk/chemistry/utils/omp_utils.hpp>
#ifdef ENABLE_NVTX3
#include <nvtx3/nvtx3.hpp>
#endif
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <set>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "libecpint.hpp"
#include "libint2.hpp"
#include "spdlog/spdlog.h"
#include "util/macros.h"
#include "util/mpi_vars.h"
#include "util/timer.h"

namespace qdk::chemistry::scf {
/**
 * @brief Libint2Engine class for computing one-body integrals using the Libint2
 * library
 *
 * This class provides a unified interface for evaluating various one-electron
 * operators including overlap, kinetic energy, nuclear attraction, and
 * multipole moment integrals. It wraps the Libint2 engine to provide
 * thread-safe computation of one-body integrals with support for basis set
 * derivatives up to second order.
 *
 * The engine automatically handles different angular momentum combinations and
 * contraction schemes. Buffer management is handled internally to ensure
 * efficient memory usage during integral evaluation.
 *
 * @note This class is thread-safe when used with separate instances per thread
 */
class Libint2Engine : public OneBodyIntegralEngine {
  using EigenVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

 public:
  /**
   * @brief Constructor for Libint2Engine
   *
   * Initializes the libint2 engine for computing one-body integrals with
   * specified operator, basis set, and derivative order.
   *
   * @param op Libint2 operator type (overlap, kinetic, nuclear, etc.)
   * @param obs Basis set in libint2 format
   * @param deriv Derivative order (0=integrals, 1=gradients, 2=hessians)
   * @param basis_mode Spherical or Cartesian basis function mode
   */
  Libint2Engine(libint2::Operator op, const libint2::BasisSet& obs, int deriv,
                BasisMode basis_mode)
      : obs_(obs),
        engine_(op, obs.max_nprim(), obs.max_l(), deriv),
        basis_mode_(basis_mode) {
    auto maxBF = (obs.max_l() + 1) * (obs.max_l() + 2) / 2;
    buf_ = std::vector<EigenVector>(1, EigenVector(maxBF * maxBF));
  }

  /**
   * @brief Destructor for Libint2Engine
   */
  ~Libint2Engine() {}

  /**
   * @brief Get reference to underlying libint2 engine
   * @return Reference to libint2::Engine for parameter setting
   */
  libint2::Engine& get() { return engine_; }

  /**
   * @brief Get number of operator components
   * @return Number of matrices/operators this engine computes
   */
  size_t nopers() const { return engine_.results().size(); }

  /**
   * @brief Compute integrals between two shells
   *
   * Evaluates integrals between basis function shells i and j using
   * the configured libint2 engine and operator.
   *
   * @param i Index of first shell
   * @param j Index of second shell
   * @return Vector of pointers to integral matrices (one per operator)
   */
  std::vector<const double*> compute(int i, int j) override {
    auto& res = engine_.compute(obs_[i], obs_[j]);
    return std::vector<const double*>(res.begin(), res.end());
  }

 private:
  const libint2::BasisSet& obs_;  ///< Reference to basis set in libint2 format
  libint2::Engine engine_;        ///< Libint2 integral engine for computation
  BasisMode basis_mode_;          ///< Spherical vs Cartesian basis mode
  std::vector<EigenVector> buf_;  ///< Buffer for storing computed integrals
};

/**
 * @brief ECPIntEngine class for computing effective core potential (ECP)
 * integrals
 *
 * This class handles the evaluation of effective core potential integrals for
 * elements where relativistic effects are important. ECPs replace the core
 * electrons and nuclear Coulomb potential with an effective potential that
 * accounts for relativistic effects in the core region.
 *
 * The implementation uses the libecpint library for efficient ECP integral
 * evaluation, supporting both semi-local and local ECPs. It handles the
 * complex angular momentum coupling required for ECP matrix elements and
 * provides derivatives for geometry optimization.
 *
 * @note Requires libecpint library for ECP integral computation
 * @note Pure vs. Cartesian basis function conventions are handled automatically
 */
class ECPIntEngine : public OneBodyIntegralEngine {
  using EigenVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

 public:
  /**
   * @brief Constructor for ECPIntEngine
   *
   * Initializes the libecpint engine for computing ECP integrals and
   * derivatives. Sets up buffers for storing results and validates derivative
   * order.
   *
   * @param shells Gaussian shells in libecpint format
   * @param ecps ECP definitions for each atom with ECPs
   * @param maxLB Maximum angular momentum in basis set
   * @param maxLU Maximum angular momentum in ECP basis
   * @param natom Total number of atoms in system
   * @param deriv Derivative order (0=integrals, 1=gradients, 2=hessians)
   * @param pure True for spherical harmonics, false for Cartesian
   * @param basis_mode Basis function convention (spherical/Cartesian)
   */
  ECPIntEngine(const std::vector<libecpint::GaussianShell>& shells,
               const std::vector<libecpint::ECP>& ecps, int maxLB, int maxLU,
               int natom, int deriv, bool pure, BasisMode basis_mode)
      : natom_(natom),
        deriv_(deriv),
        engine_(maxLB, maxLU, deriv),
        shells_(shells),
        ecps_(ecps),
        pure_(pure),
        basis_mode_(basis_mode) {
    VERIFY(deriv_ >= 0 && deriv_ <= 2);

    auto maxBF = (maxLB + 1) * (maxLB + 2) / 2;
    auto count = deriv == 0 ? 1 : (deriv == 1 ? 3 * natom : 45);
    buf_ = std::vector<EigenVector>(count, EigenVector(maxBF * maxBF));
  }

  /**
   * @brief Get number of operator components
   * @return Number of matrices this engine computes (1 for integrals, 3*natom
   * for gradients)
   */
  size_t nopers() const { return buf_.size(); }

  /**
   * @brief Compute ECP integrals between two shells
   *
   * Evaluates ECP matrix elements between shells i and j, including all
   * relevant ECPs. Handles spherical harmonic transformations and
   * accumulates contributions from multiple ECP centers.
   *
   * @param i Index of first shell
   * @param j Index of second shell
   * @return Vector of pointers to integral/derivative matrices
   */
  std::vector<const double*> compute(int i, int j) override {
    std::for_each(buf_.begin(), buf_.end(), [](auto& v) { v.setZero(); });

    auto& sh1 = shells_[i];
    auto& sh2 = shells_[j];
    if (deriv_ == 0) {
      compute_integrals(sh1, sh2);
    } else if (deriv_ == 1) {
      compute_gradients(sh1, sh2);
    } else if (deriv_ == 2) {
      compute_hessians(sh1, sh2);
    } else {
      VERIFY(false && "not implemented");
    }

    std::vector<const double*> res(buf_.size());
    std::transform(buf_.begin(), buf_.end(), res.begin(),
                   [](auto& v) { return v.data(); });
    return res;
  }

 private:
  /**
   * @brief Transform Cartesian integrals to spherical harmonics
   *
   * Converts ECP integrals from Cartesian to spherical harmonic basis
   * functions using libint2's solid harmonics transformation routines.
   *
   * @param l1 Angular momentum of first shell
   * @param l2 Angular momentum of second shell
   * @param cart_ints Input Cartesian integrals
   * @param sph_ints Output spherical harmonic integrals
   */
  void cartesian_to_spherical(int l1, int l2, const double* cart_ints,
                              double* sph_ints) {
    int n1 = l1 * 2 + 1, n2 = l2 * 2 + 1;
    if (l1 >= 2 && l2 >= 2) {
      libint2::solidharmonics::tform(l1, l2, cart_ints, sph_ints);
    } else if (l1 >= 2) {
      libint2::solidharmonics::tform_rows(l1, n2, cart_ints, sph_ints);
    } else if (l2 >= 2) {
      libint2::solidharmonics::tform_cols(n1, l2, cart_ints, sph_ints);
    } else {
      std::copy(cart_ints, cart_ints + n1 * n2, sph_ints);
    }
  }

  /**
   * @brief Compute ECP integral values (zero-order derivatives)
   *
   * Evaluates ECP matrix elements between two shells by iterating over
   * all relevant ECPs and accumulating contributions. Handles spherical
   * harmonic transformations when required.
   *
   * @param sh1 First Gaussian shell
   * @param sh2 Second Gaussian shell
   */
  void compute_integrals(const libecpint::GaussianShell& sh1,
                         const libecpint::GaussianShell& sh2) {
    libecpint::TwoIndex<double> res;
    int size = sh1.ncartesian() * sh2.ncartesian();
    for (auto& ecp : ecps_) {
      engine_.compute_shell_pair(ecp, sh1, sh2, res);
      int l1 = sh1.am(), l2 = sh2.am();
      if (!pure_ || (l1 < 2 && l2 < 2)) {
        buf_[0].head(size) +=
            Eigen::Map<const EigenVector>(res.data.data(), size);
      } else {
        int n1 = l1 * 2 + 1, n2 = l2 * 2 + 1;
        std::vector<double> sph(n1 * n2);
        cartesian_to_spherical(l1, l2, res.data.data(), sph.data());
        buf_[0].head(sph.size()) +=
            Eigen::Map<const EigenVector>(sph.data(), sph.size());
      }
    }
  }

  /**
   * @brief Compute ECP integral gradients (first-order derivatives)
   *
   * Evaluates first derivatives of ECP integrals with respect to nuclear
   * coordinates. Accumulates contributions from shell centers and ECP
   * centers that can move during geometry optimization.
   *
   * @param sh1 First Gaussian shell
   * @param sh2 Second Gaussian shell
   */
  void compute_gradients(const libecpint::GaussianShell& sh1,
                         const libecpint::GaussianShell& sh2) {
    int centers[3] = {sh1.atom_id, sh2.atom_id, -1};
    auto size = sh1.ncartesian() * sh2.ncartesian();
    for (auto& ecp : ecps_) {
      centers[2] = ecp.atom_id;
      std::array<libecpint::TwoIndex<double>, 9> res;
      engine_.compute_shell_pair_derivative(ecp, sh1, sh2, res);
      int i = 0;
      for (auto c : centers) {
        for (auto xyz = 0; xyz < 3; ++xyz) {
          int l1 = sh1.am(), l2 = sh2.am();
          if (!pure_ || (l1 < 2 && l2 < 2)) {
            buf_[c * 3 + xyz].head(size) +=
                Eigen::Map<const EigenVector>(res[i].data.data(), size);
          } else {
            int n1 = l1 * 2 + 1, n2 = l2 * 2 + 1;
            std::vector<double> sph(n1 * n2);
            cartesian_to_spherical(l1, l2, res[i].data.data(), sph.data());
            buf_[c * 3 + xyz].head(sph.size()) +=
                Eigen::Map<const EigenVector>(sph.data(), sph.size());
          }
          i++;
        }
      }
    }
  }

  /**
   * @brief Compute ECP integral Hessians (second-order derivatives)
   *
   * Placeholder for second derivative evaluation of ECP integrals.
   * Currently not implemented in libecpint library.
   *
   * @param sh1 First Gaussian shell
   * @param sh2 Second Gaussian shell
   */
  void compute_hessians(const libecpint::GaussianShell& sh1,
                        const libecpint::GaussianShell& sh2) {
    VERIFY(false && "not implemented");
  }

  int natom_;  ///< Total number of atoms in system
  int deriv_;  ///< Derivative order (0=integrals, 1=gradients, 2=hessians)
  libecpint::ECPIntegral
      engine_;  ///< Libecpint engine for ECP integral evaluation
  const std::vector<libecpint::GaussianShell>&
      shells_;  ///< Reference to Gaussian shells in libecpint format
  const std::vector<libecpint::ECP>&
      ecps_;              ///< Reference to ECP definitions for relevant atoms
  bool pure_;             ///< True for spherical harmonics, false for Cartesian
  BasisMode basis_mode_;  ///< Basis function convention (spherical/Cartesian)

  std::vector<EigenVector>
      buf_;  ///< Buffers for storing computed integrals/derivatives
};

void OneBodyIntegral::convert_to_libecp_shells_(const BasisSet& obs) {
  for (auto& sh : obs.shells) {
    std::array<double, 3> O = sh.O;
    auto& shell = libecp_shells_.emplace_back(O, sh.angular_momentum);
    for (size_t i = 0; i < sh.contraction; ++i) {
      shell.addPrim(sh.exponents[i], sh.coefficients[i]);
    }
  }

  max_ecp_am_ = -1;
  std::vector<int> ecp_atoms;
  for (auto& sh : obs.ecp_shells) {
    if (ecp_atoms.empty() ||
        static_cast<int>(sh.atom_index) != ecp_atoms.back()) {
      double O[3]{sh.O[0], sh.O[1], sh.O[2]};
      libecp_ecps_.emplace_back(O);
      ecp_atoms.push_back(static_cast<int>(sh.atom_index));
    }
    auto& ecp = libecp_ecps_.back();
    for (size_t i = 0; i < sh.contraction; ++i) {
      ecp.addPrimitive(sh.rpowers[i], sh.angular_momentum, sh.exponents[i],
                       sh.coefficients[i], i == sh.contraction - 1 /*sort*/);
    }
    max_ecp_am_ = std::max(max_ecp_am_, static_cast<int>(sh.angular_momentum));
  }

  // libecpint does not copy atom_id when copying GaussianShell!
  // so we manually set atom_id after building the vector of shells.
  for (size_t i = 0; i < libecp_shells_.size(); ++i) {
    libecp_shells_[i].atom_id = obs.shells[i].atom_index;
  }
  for (size_t i = 0; i < libecp_ecps_.size(); ++i) {
    libecp_ecps_[i].atom_id = ecp_atoms[i];
  }
}

OneBodyIntegral::OneBodyIntegral(const BasisSet* basis_set, const Molecule* mol,
                                 ParallelConfig mpi)
    : mpi_(mpi) {
  obs_ = libint2_util::convert_to_libint_basisset(*basis_set);
  basis_mode_ = basis_set->mode;
  pure_ = basis_set->pure;
  if (basis_set->ecp_shells.size() > 0) {
    convert_to_libecp_shells_(*basis_set);
  }
  for (size_t i = 0; i < mol->n_atoms; i++) {
    atoms_.push_back({1.0 * mol->atomic_charges[i], mol->coords[i]});
  }
  for (size_t i = 0; i < basis_set->shells.size(); i++) {
    sh2atom_.push_back(basis_set->shells[i].atom_index);
  }
  shell_pairs_ = basis_set->get_shell_pairs();
}

OneBodyIntegral::~OneBodyIntegral() {}

std::vector<std::pair<int, int>> OneBodyIntegral::compute_shell_pairs(
    const std::vector<Shell>& shells, const double threshold) {
  AutoTimer __timer("int1e::prepare shell pairs");
  std::vector<libint2::Shell> shs;
  for (auto& sh : shells) {
    shs.push_back(libint2_util::convert_to_libint_shell(sh, true));
  }
  auto obs = libint2::BasisSet(shs);

  RowMajorMatrix S = RowMajorMatrix::Zero(obs.size(), obs.size());
  auto engine_fn = [&]() {
    return libint2::Engine(libint2::Operator::overlap, obs.max_nprim(),
                           obs.max_l(), 0);
  };
  int nthreads = omp_get_max_threads();
  std::vector<libint2::Engine> engines(nthreads, engine_fn());
#pragma omp parallel num_threads(nthreads)
  {
    int local_thread_id = omp_get_thread_num();
    int world_thread_size = mpi::get_world_size() * nthreads;
    int world_thread_id = mpi::get_world_rank() * nthreads + local_thread_id;
    libint2::Engine& engine = engines[local_thread_id];

    for (size_t i = 0, job_idx = 0; i < obs.size(); i++) {
      for (size_t j = 0; j <= i; j++, job_idx++) {
        if (job_idx % world_thread_size != world_thread_id) {
          continue;
        }
        if (obs[i].O == obs[j].O) {
          S(i, j) = 1.0;
          continue;
        }
        size_t n1 = obs[i].size(), n2 = obs[j].size();
        auto& buf = engine.compute(obs[i], obs[j]);
        Eigen::Map<const RowMajorMatrix> mat(buf[0], n1, n2);
        S(i, j) = mat.norm();
      }
    }
  }
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi::get_world_size() > 1) {
    MPI_Allreduce(MPI_IN_PLACE, S.data(), S.size(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  }
#endif
  std::vector<std::pair<int, int>> shell_pairs;
  for (size_t i = 0; i < obs.size(); i++) {
    for (size_t j = 0; j <= i; j++) {
      if (S(i, j) > threshold) {
        shell_pairs.emplace_back(i, j);
      }
    }
  }
  spdlog::trace("int1e::n_shell_pairs: {}", shell_pairs.size());
  return shell_pairs;
}

void OneBodyIntegral::integral_(size_t nopers, EngineFactory engine_fn,
                                RowMajorMatrix* res) {
  const auto& shell2bf = obs_.shell2bf();
  int nthreads = omp_get_max_threads();
#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{"int1e"};
#endif
#pragma omp parallel num_threads(nthreads)
  {
    int local_thread_id = omp_get_thread_num();
    int world_thread_size = mpi_.world_size * nthreads;
    int world_thread_id = mpi_.world_rank * nthreads + local_thread_id;
    auto engine = engine_fn();
    if (engine->nopers() != nopers) {
      throw std::invalid_argument("nopers is inconsistent");
    }

    for (size_t p = world_thread_id; p < shell_pairs_.size();
         p += world_thread_size) {
      auto [i, j] = shell_pairs_[p];
      size_t bf1 = shell2bf[i], bf2 = shell2bf[j];
      size_t n1 = obs_[i].size(), n2 = obs_[j].size();
      auto buf = engine->compute(i, j);
      for (auto k = 0; k < nopers; ++k) {
        Eigen::Map<const RowMajorMatrix> mat(buf[k], n1, n2);
        res[k].block(bf1, bf2, n1, n2) = mat;
        if (i != j) {
          res[k].block(bf2, bf1, n2, n1) = mat.transpose();
        }
      }
    }
  }
}

void OneBodyIntegral::overlap_integral(double* res) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  RowMajorMatrix mat = RowMajorMatrix::Zero(obs_.nbf(), obs_.nbf());
  auto engine_fn = [&]() {
    return std::make_unique<Libint2Engine>(libint2::Operator::overlap, obs_, 0,
                                           basis_mode_);
  };
  integral_(1, engine_fn, &mat);
  memcpy(res, mat.data(), sizeof(double) * mat.size());
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res, mat.size(),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void OneBodyIntegral::kinetic_integral(double* res) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  RowMajorMatrix mat = RowMajorMatrix::Zero(obs_.nbf(), obs_.nbf());
  auto engine_fn = [&]() {
    return std::make_unique<Libint2Engine>(libint2::Operator::kinetic, obs_, 0,
                                           basis_mode_);
  };
  integral_(1, engine_fn, &mat);
  memcpy(res, mat.data(), sizeof(double) * mat.size());
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res, mat.size(),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void OneBodyIntegral::nuclear_integral(double* res) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  RowMajorMatrix mat = RowMajorMatrix::Zero(obs_.nbf(), obs_.nbf());
  auto engine_fn = [&]() {
    auto engine = std::make_unique<Libint2Engine>(libint2::Operator::nuclear,
                                                  obs_, 0, basis_mode_);
    engine->get().set_params(atoms_);
    return engine;
  };
  integral_(1, engine_fn, &mat);
  memcpy(res, mat.data(), sizeof(double) * mat.size());
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res, mat.size(),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void OneBodyIntegral::dipole_integral(double* res, std::array<double, 3> cen) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  std::vector<RowMajorMatrix> mat(4,
                                  RowMajorMatrix::Zero(obs_.nbf(), obs_.nbf()));
  auto engine_fn = [&]() {
    auto engine = std::make_unique<Libint2Engine>(
        libint2::Operator::emultipole1, obs_, 0, basis_mode_);
    engine->get().set_params(cen);
    return engine;
  };
  integral_(4, engine_fn, mat.data());
  // First term is the overlap
  memcpy(res + 0 * mat[0].size(), mat[1].data(),
         sizeof(double) * mat[0].size());
  memcpy(res + 1 * mat[0].size(), mat[2].data(),
         sizeof(double) * mat[0].size());
  memcpy(res + 2 * mat[0].size(), mat[3].data(),
         sizeof(double) * mat[0].size());
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res,
               3 * mat[0].size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void OneBodyIntegral::quadrupole_integral(double* res,
                                          std::array<double, 3> cen) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  std::vector<RowMajorMatrix> mat(10,
                                  RowMajorMatrix::Zero(obs_.nbf(), obs_.nbf()));
  auto engine_fn = [&]() {
    auto engine = std::make_unique<Libint2Engine>(
        libint2::Operator::emultipole2, obs_, 0, basis_mode_);
    engine->get().set_params(cen);
    return engine;
  };
  integral_(10, engine_fn, mat.data());
  // First term is the overlap, then 3 dipole terms, then 6 quadrupole terms
  // Quadrupole order: xx, xy, xz, yy, yz, zz (indices 4-9)
  memcpy(res + 0 * mat[0].size(), mat[4].data(),
         sizeof(double) * mat[0].size());
  memcpy(res + 1 * mat[0].size(), mat[5].data(),
         sizeof(double) * mat[0].size());
  memcpy(res + 2 * mat[0].size(), mat[6].data(),
         sizeof(double) * mat[0].size());
  memcpy(res + 3 * mat[0].size(), mat[7].data(),
         sizeof(double) * mat[0].size());
  memcpy(res + 4 * mat[0].size(), mat[8].data(),
         sizeof(double) * mat[0].size());
  memcpy(res + 5 * mat[0].size(), mat[9].data(),
         sizeof(double) * mat[0].size());
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res,
               6 * mat[0].size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

#ifdef QDK_CHEMISTRY_ENABLE_QMMM
void OneBodyIntegral::point_charge_integral(const PointCharges* charges,
                                            double* res) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  RowMajorMatrix mat = RowMajorMatrix::Zero(obs_.nbf(), obs_.nbf());
  auto engine_fn = [&]() {
    auto engine = std::make_unique<Libint2Engine>(libint2::Operator::nuclear,
                                                  obs_, 0, basis_mode_);
    // engine->get().set_params(atoms_);
    std::vector<std::pair<double, std::array<double, 3>>> q;
    for (int i = 0; i < charges->n_points; i++) {
      q.push_back({charges->charges[i],
                   {charges->coords[i][0], charges->coords[i][1],
                    charges->coords[i][2]}});
    }
    engine->get().set_params(q);
    return engine;
  };
  integral_(1, engine_fn, &mat);
  memcpy(res, mat.data(), sizeof(double) * mat.size());
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res, mat.size(),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}
#endif  // QDK_CHEMISTRY_ENABLE_QMMM

void OneBodyIntegral::ecp_integral(double* res) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  RowMajorMatrix mat = RowMajorMatrix::Zero(obs_.nbf(), obs_.nbf());
  auto engine_fn = [&]() {
    return std::make_unique<ECPIntEngine>(libecp_shells_, libecp_ecps_,
                                          obs_.max_l(), max_ecp_am_,
                                          atoms_.size(), 0, pure_, basis_mode_);
  };
  integral_(1, engine_fn, &mat);
  memcpy(res, mat.data(), sizeof(double) * mat.size());
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res, mat.size(),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void OneBodyIntegral::integral_deriv_(EngineFactory engine_fn,
                                      const RowMajorMatrix& coeff,
                                      AtomCenterFn center_fn, double* res) {
  const auto& shell2bf = obs_.shell2bf();
  int nthreads = omp_get_max_threads();
#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{"int1e_deriv"};
#endif
#pragma omp parallel num_threads(nthreads)
  {
    int local_thread_id = omp_get_thread_num();
    int world_thread_size = mpi_.world_size * nthreads;
    int world_thread_id = mpi_.world_rank * nthreads + local_thread_id;
    auto engine = engine_fn();

    for (size_t p = world_thread_id; p < shell_pairs_.size();
         p += world_thread_size) {
      auto [i, j] = shell_pairs_[p];

      size_t bf1 = shell2bf[i], bf2 = shell2bf[j];
      size_t n1 = obs_[i].size(), n2 = obs_[j].size();
      auto buf = engine->compute(i, j);

      auto centers = center_fn(i, j);
      for (auto& [atom, idx] : centers) {
        for (int xyz = 0; xyz < 3; xyz++) {
          Eigen::Map<const RowMajorMatrix> mat(buf[idx * 3 + xyz], n1, n2);
          double value = coeff.block(bf1, bf2, n1, n2).cwiseProduct(mat).sum();
#pragma omp atomic
          res[xyz * atoms_.size() + atom] += value;
          if (i != j) {
            value = coeff.block(bf2, bf1, n2, n1)
                        .cwiseProduct(mat.transpose())
                        .sum();
#pragma omp atomic
            res[xyz * atoms_.size() + atom] += value;
          }
        }
      }
    }
  }
}

void OneBodyIntegral::kinetic_integral_deriv(const double* D, double* res) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  std::vector<RowMajorMatrix> mat;
  auto engine_fn = [&]() {
    return std::make_unique<Libint2Engine>(libint2::Operator::kinetic, obs_, 1,
                                           basis_mode_);
  };
  auto center_fn = [&](int i, int j) {
    return std::vector<std::pair<int, int>>{{sh2atom_[i], 0}, {sh2atom_[j], 1}};
  };
  Eigen::Map<const RowMajorMatrix> coeff(D, obs_.nbf(), obs_.nbf());
  integral_deriv_(engine_fn, coeff, center_fn, res);

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    auto size = 3 * atoms_.size();
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res, size, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void OneBodyIntegral::overlap_integral_deriv(const double* W, double* res) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  std::vector<RowMajorMatrix> mat;
  auto engine_fn = [&]() {
    return std::make_unique<Libint2Engine>(libint2::Operator::overlap, obs_, 1,
                                           basis_mode_);
  };
  auto center_fn = [&](int i, int j) {
    return std::vector<std::pair<int, int>>{{sh2atom_[i], 0}, {sh2atom_[j], 1}};
  };
  Eigen::Map<const RowMajorMatrix> coeff(W, obs_.nbf(), obs_.nbf());
  integral_deriv_(engine_fn, coeff, center_fn, res);

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    auto size = 3 * atoms_.size();
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res, size, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void OneBodyIntegral::nuclear_integral_deriv(const double* D, double* res) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  std::vector<RowMajorMatrix> mat;
  auto engine_fn = [&]() {
    auto engine = std::make_unique<Libint2Engine>(libint2::Operator::nuclear,
                                                  obs_, 1, basis_mode_);
    engine->get().set_params(atoms_);
    return engine;
  };
  auto center_fn = [&](int i, int j) {
    std::vector<std::pair<int, int>> centers;
    centers.emplace_back(sh2atom_[i], 0);
    centers.emplace_back(sh2atom_[j], 1);
    for (auto i = 0; i < atoms_.size(); ++i) {
      centers.emplace_back(i, i + 2);
    }
    return centers;
  };
  Eigen::Map<const RowMajorMatrix> coeff(D, obs_.nbf(), obs_.nbf());
  integral_deriv_(engine_fn, coeff, center_fn, res);

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    auto size = 3 * atoms_.size();
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res, size, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

#ifdef QDK_CHEMISTRY_ENABLE_QMMM
void OneBodyIntegral::pointcharge_integral_deriv(const double* D, double* res,
                                                 double* pointcharges_res,
                                                 const PointCharges* charges) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  std::vector<RowMajorMatrix> mat;
  auto engine_fn = [&]() {
    auto engine = std::make_unique<Libint2Engine>(libint2::Operator::nuclear,
                                                  obs_, 1, basis_mode_);
    std::vector<std::pair<double, std::array<double, 3>>> q;
    for (int i = 0; i < charges->n_points; i++) {
      q.push_back({charges->charges[i],
                   {charges->coords[i][0], charges->coords[i][1],
                    charges->coords[i][2]}});
    }
    engine->get().set_params(q);
    return engine;
  };
  auto center_fn = [&](int i, int j) {
    std::vector<std::pair<int, int>> centers;
    centers.emplace_back(sh2atom_[i], 0);
    centers.emplace_back(sh2atom_[j], 1);
    for (auto p = 0; p < charges->n_points; ++p) {
      centers.emplace_back(p, p + 2);
    }
    return centers;
  };
  Eigen::Map<const RowMajorMatrix> coeff(D, obs_.nbf(), obs_.nbf());
  const auto& shell2bf = obs_.shell2bf();
  int nthreads = omp_get_max_threads();
#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{"int1e_deriv"};
#endif
#pragma omp parallel num_threads(nthreads)
  {
    int local_thread_id = omp_get_thread_num();
    int world_thread_size = mpi_.world_size * nthreads;
    int world_thread_id = mpi_.world_rank * nthreads + local_thread_id;
    auto engine = engine_fn();

    for (size_t p = world_thread_id; p < shell_pairs_.size();
         p += world_thread_size) {
      auto [i, j] = shell_pairs_[p];
      size_t bf1 = shell2bf[i], bf2 = shell2bf[j];
      size_t n1 = obs_[i].size(), n2 = obs_[j].size();
      auto buf = engine->compute(i, j);
      auto centers = center_fn(i, j);
      for (auto& [atom_or_pcidx, idx] : centers) {
        for (int xyz = 0; xyz < 3; xyz++) {
          Eigen::Map<const RowMajorMatrix> mat(
              buf[idx * 3 + xyz], n1, n2);  // map the buffer to a matrix,
                                            // starting from atom idx * 3 + xyz
          double value = coeff.block(bf1, bf2, n1, n2).cwiseProduct(mat).sum();
          if (idx < 2) {
#pragma omp atomic
            res[xyz * atoms_.size() + atom_or_pcidx] += value;
          } else {
#pragma omp atomic
            pointcharges_res[xyz * charges->n_points + atom_or_pcidx] += value;
          }
          if (i != j) {
            value = coeff.block(bf2, bf1, n2, n1)
                        .cwiseProduct(mat.transpose())
                        .sum();
            if (idx < 2) {
#pragma omp atomic
              res[xyz * atoms_.size() + atom_or_pcidx] += value;
            } else {
#pragma omp atomic
              pointcharges_res[xyz * charges->n_points + atom_or_pcidx] +=
                  value;
            }
          }
        }
      }
    }
  }

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    auto size = 3 * atoms_.size();
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res, size, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}
#endif  // QDK_CHEMISTRY_ENABLE_QMMM

void OneBodyIntegral::ecp_integral_deriv(const double* D, double* res) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  std::vector<RowMajorMatrix> mat;
  auto engine_fn = [&]() {
    return std::make_unique<ECPIntEngine>(libecp_shells_, libecp_ecps_,
                                          obs_.max_l(), max_ecp_am_,
                                          atoms_.size(), 1, pure_, basis_mode_);
  };
  std::set<int> ecp_centers;
  for (auto& ecp : libecp_ecps_) {
    ecp_centers.insert(ecp.atom_id);
  }
  auto center_fn = [&](int i, int j) {
    std::set<int> atoms(ecp_centers.begin(), ecp_centers.end());
    atoms.insert(sh2atom_[i]);
    atoms.insert(sh2atom_[j]);
    std::vector<std::pair<int, int>> centers;
    for (auto atom : atoms) {
      centers.emplace_back(atom, atom);
    }
    return centers;
  };
  Eigen::Map<const RowMajorMatrix> coeff(D, obs_.nbf(), obs_.nbf());
  integral_deriv_(engine_fn, coeff, center_fn, res);

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    auto size = 3 * atoms_.size();
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : res, res, size, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}
}  // namespace qdk::chemistry::scf
