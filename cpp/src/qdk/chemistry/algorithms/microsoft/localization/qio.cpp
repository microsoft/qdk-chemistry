// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qio.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <variant>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft {

namespace {

// Jacobi-sweep controls. The maximum sweep count, convergence tolerance and
// coarse angle step are configurable through QIOLocalizerSettings; the
// following are fixed implementation details.
constexpr int kFineSamples = 201;      // fine-refinement samples
constexpr double kImproveTol = 1e-12;  // minimum accepted entropy decrease
constexpr double kPi = 3.14159265358979323846;

// Minimum active-space dimension at which the 2-RDM rotation is threaded.
// Below this, the tensor is small enough that OpenMP fork/join overhead
// outweighs the benefit and the loop runs serially (bitwise-identical results).
// [[maybe_unused]]: referenced only inside the OpenMP `if` clause below, which
// is compiled out when OpenMP is disabled (e.g. the Windows build).
[[maybe_unused]] constexpr std::size_t kParallelMinDim = 32;

// Boguslawski & Tecmer (2015), doi:10.1002/qua.24832 single-orbital (von
// Neumann) entropy from the orbital occupation eigenvalues
// {1 - na - nb + d, na - d, nb - d, d}.
double single_orbital_entropy(double na, double nb, double d) {
  const double omega[4] = {1.0 - na - nb + d, na - d, nb - d, d};
  double s = 0.0;
  for (double w : omega) {
    if (w > 1e-14) {
      s -= w * std::log(w);
    }
  }
  return s;
}

// Flat row-major index into an (n x n x n x n) tensor.
inline std::size_t idx4(std::size_t n, std::size_t i, std::size_t j,
                        std::size_t k, std::size_t l) {
  return ((i * n + j) * n + k) * n + l;
}

// Apply an in-place Givens rotation G(i, j; c, s) to one axis of a rank-4
// tensor whose four axes each have dimension n. The transform on the active
// axis is v_i <- c v_i + s v_j, v_j <- -s v_i + c v_j.
void rotate_two_rdm_axis(std::vector<double>& g2, std::size_t n, int axis,
                         std::size_t i, std::size_t j, double c, double s) {
  const std::size_t strides[4] = {n * n * n, n * n, n, 1};
  const std::size_t sa = strides[axis];
  int other[3];
  int t = 0;
  for (int ax = 0; ax < 4; ++ax) {
    if (ax != axis) {
      other[t++] = ax;
    }
  }
  // Each (x, y, z, active in {i, j}) maps to a unique flat index, so every
  // touched element is written at most once: the (x, y) iterations are
  // independent and safe to run in parallel.
#pragma omp parallel for collapse(2) schedule(static) if (n >= kParallelMinDim)
  for (std::size_t x = 0; x < n; ++x) {
    for (std::size_t y = 0; y < n; ++y) {
      for (std::size_t z = 0; z < n; ++z) {
        const std::size_t base = x * strides[other[0]] + y * strides[other[1]] +
                                 z * strides[other[2]];
        double& ti = g2[base + i * sa];
        double& tj = g2[base + j * sa];
        const double vi = ti;
        const double vj = tj;
        ti = c * vi + s * vj;
        tj = -s * vi + c * vj;
      }
    }
  }
}

// Apply a Givens rotation G(i, j; c, s) to all four axes of a 2-RDM tensor.
void rotate_two_rdm(std::vector<double>& g2, std::size_t n, std::size_t i,
                    std::size_t j, double c, double s) {
  for (int axis = 0; axis < 4; ++axis) {
    rotate_two_rdm_axis(g2, n, axis, i, j, c, s);
  }
}

// Apply a Givens rotation G(i, j; c, s) to both axes of a (symmetric) 1-RDM.
void rotate_one_rdm(Eigen::MatrixXd& g, std::size_t i, std::size_t j, double c,
                    double s) {
  // Left rotation: mix rows i and j (iterate over all columns p).
  for (Eigen::Index p = 0; p < g.cols(); ++p) {
    const double gi = g(static_cast<Eigen::Index>(i), p);
    const double gj = g(static_cast<Eigen::Index>(j), p);
    g(static_cast<Eigen::Index>(i), p) = c * gi + s * gj;
    g(static_cast<Eigen::Index>(j), p) = -s * gi + c * gj;
  }
  // Right rotation: mix columns i and j (iterate over all rows p).
  for (Eigen::Index p = 0; p < g.rows(); ++p) {
    const double gi = g(p, static_cast<Eigen::Index>(i));
    const double gj = g(p, static_cast<Eigen::Index>(j));
    g(p, static_cast<Eigen::Index>(i)) = c * gi + s * gj;
    g(p, static_cast<Eigen::Index>(j)) = -s * gi + c * gj;
  }
}

// Total single-orbital entropy of the current (rotated) RDMs.
double entropy_sum(const Eigen::MatrixXd& ga, const Eigen::MatrixXd& gb,
                   const std::vector<double>& g2, std::size_t n) {
  double f = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const Eigen::Index ii = static_cast<Eigen::Index>(i);
    f +=
        single_orbital_entropy(ga(ii, ii), gb(ii, ii), g2[idx4(n, i, i, i, i)]);
  }
  return f;
}

// Minimize sum_i S(rho_i) by gradient-free Jacobi sweeps. The input RDMs ga,
// gb (n x n) and g2 (n^4 alpha-beta block) are rotated in place and the
// accumulated active-space rotation U (n x n) is returned.
Eigen::MatrixXd optimize_rotation(Eigen::MatrixXd& ga, Eigen::MatrixXd& gb,
                                  std::vector<double>& g2, std::size_t n,
                                  std::size_t max_cycles, double tol,
                                  double coarse_step) {
  Eigen::MatrixXd u = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n),
                                                static_cast<Eigen::Index>(n));
  if (n < 2) {
    return u;
  }

  double f_prev = entropy_sum(ga, gb, g2, n);
  for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        const Eigen::Index ii = static_cast<Eigen::Index>(i);
        const Eigen::Index jj = static_cast<Eigen::Index>(j);
        const std::size_t pr[2] = {i, j};
        const double gaa = ga(ii, ii), gab = ga(ii, jj), gbb = ga(jj, jj);
        const double haa = gb(ii, ii), hab = gb(ii, jj), hbb = gb(jj, jj);

        // Gamma_{i'ibar'i'ibar'} for orbital weights (w0, w1) over the pair.
        auto contract4 = [&](double w0, double w1) {
          const double w[2] = {w0, w1};
          double d = 0.0;
          for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
              for (int cc = 0; cc < 2; ++cc) {
                for (int dd = 0; dd < 2; ++dd) {
                  d += w[a] * w[b] * w[cc] * w[dd] *
                       g2[idx4(n, pr[a], pr[b], pr[cc], pr[dd])];
                }
              }
            }
          }
          return d;
        };

        // Pair entropy after rotating by theta: i' = (c, s), j' = (-s, c).
        auto eval = [&](double theta) {
          const double c = std::cos(theta), s = std::sin(theta);
          const double na_i = c * c * gaa + 2.0 * c * s * gab + s * s * gbb;
          const double nb_i = c * c * haa + 2.0 * c * s * hab + s * s * hbb;
          const double na_j = s * s * gaa - 2.0 * c * s * gab + c * c * gbb;
          const double nb_j = s * s * haa - 2.0 * c * s * hab + c * c * hbb;
          const double d_i = contract4(c, s);
          const double d_j = contract4(-s, c);
          return single_orbital_entropy(na_i, nb_i, d_i) +
                 single_orbital_entropy(na_j, nb_j, d_j);
        };

        const double f_current = eval(0.0);
        double best_theta = 0.0;
        double best_val = f_current;
        for (double theta = 0.0; theta < kPi; theta += coarse_step) {
          const double v = eval(theta);
          if (v < best_val) {
            best_val = v;
            best_theta = theta;
          }
        }
        const double lo = best_theta - coarse_step;
        const double hi = best_theta + coarse_step;
        for (int k = 0; k < kFineSamples; ++k) {
          const double theta =
              lo + (hi - lo) * k / static_cast<double>(kFineSamples - 1);
          const double v = eval(theta);
          if (v < best_val) {
            best_val = v;
            best_theta = theta;
          }
        }

        if (best_val < f_current - kImproveTol) {
          const double c = std::cos(best_theta), s = std::sin(best_theta);
          rotate_one_rdm(ga, i, j, c, s);
          rotate_one_rdm(gb, i, j, c, s);
          rotate_two_rdm(g2, n, i, j, c, s);
          // Accumulate U <- U * G(i, j; c, s) (in-place column update).
          for (Eigen::Index p = 0; p < u.rows(); ++p) {
            const double ui = u(p, ii);
            const double uj = u(p, jj);
            u(p, ii) = c * ui + s * uj;
            u(p, jj) = -s * ui + c * uj;
          }
        }
      }
    }
    const double f_now = entropy_sum(ga, gb, g2, n);
    if (std::abs(f_prev - f_now) < tol) {
      break;
    }
    f_prev = f_now;
  }
  return u;
}

}  // namespace

std::shared_ptr<data::Wavefunction> QIOLocalizer::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& loc_indices_a,
    const std::vector<size_t>& loc_indices_b) const {
  QDK_LOG_TRACE_ENTERING();
  auto orbitals = wavefunction->get_orbitals();

  // QIO produces a single spatial orbital set.
  if (loc_indices_a != loc_indices_b) {
    throw std::invalid_argument(
        "loc_indices_a and loc_indices_b must be identical for QIO "
        "localization.");
  }
  if (!std::is_sorted(loc_indices_a.begin(), loc_indices_a.end())) {
    throw std::invalid_argument("loc_indices_a must be sorted");
  }
  if (std::adjacent_find(loc_indices_a.begin(), loc_indices_a.end()) !=
      loc_indices_a.end()) {
    throw std::invalid_argument("loc_indices_a contains duplicate indices");
  }

  // Empty selection is a no-op, but still returns the standard single-reference
  // (Aufbau determinant) carrier for consistency with the Localizer contract.
  if (loc_indices_a.empty()) {
    return detail::new_aufbau_determinant_wavefunction(wavefunction, orbitals);
  }

  if (!orbitals->is_restricted()) {
    throw std::invalid_argument(
        "QIOLocalizer requires a single spatial orbital set (RHF/ROHF); "
        "unrestricted (UHF) orbitals are not supported.");
  }

  if (!orbitals->has_active_space()) {
    throw std::invalid_argument(
        "QIOLocalizer requires an active space to be defined in the orbitals.");
  }

  // The output Orbitals carry the AO overlap matrix; require it up front so a
  // missing overlap fails as std::invalid_argument (consistent with the other
  // input checks) rather than a std::runtime_error from get_overlap_matrix().
  if (!orbitals->has_overlap_matrix()) {
    throw std::invalid_argument(
        "QIOLocalizer requires an overlap matrix to be available in the "
        "orbitals.");
  }

  const auto& [active_indices_a, active_indices_b] =
      orbitals->get_active_space_indices();
  if (loc_indices_a != active_indices_a || loc_indices_b != active_indices_b) {
    throw std::invalid_argument(
        "QIOLocalizer requires loc_indices_a and loc_indices_b to match the "
        "orbitals' active-space indices.");
  }

  const std::size_t num_molecular_orbitals =
      orbitals->get_num_molecular_orbitals();
  if (loc_indices_a.back() >= num_molecular_orbitals) {
    throw std::invalid_argument(
        "loc_indices_a contains invalid orbital index >= "
        "num_molecular_orbitals");
  }

  // Single-orbital entropies require correlated spin-dependent RDMs.
  if (!wavefunction->has_one_rdm_spin_dependent() ||
      !wavefunction->has_two_rdm_spin_dependent()) {
    throw std::invalid_argument(
        "QIOLocalizer requires spin-dependent active 1- and 2-RDMs in the "
        "wavefunction.");
  }

  const std::size_t n = active_indices_a.size();

  // Extract the active alpha/beta 1-RDM and the alpha-beta (aabb) 2-RDM block.
  auto [rdm_aa_variant, rdm_bb_variant] =
      wavefunction->get_active_one_rdm_spin_dependent();
  auto [rdm_aaaa_variant, rdm_aabb_variant, rdm_bbbb_variant] =
      wavefunction->get_active_two_rdm_spin_dependent();
  (void)rdm_aaaa_variant;  // QIO only needs the alpha-beta (aabb) block.
  (void)rdm_bbbb_variant;

  const auto* rdm_aa = std::get_if<Eigen::MatrixXd>(&rdm_aa_variant);
  const auto* rdm_bb = std::get_if<Eigen::MatrixXd>(&rdm_bb_variant);
  const auto* rdm_aabb = std::get_if<Eigen::VectorXd>(&rdm_aabb_variant);
  if (!rdm_aa || !rdm_bb || !rdm_aabb) {
    throw std::invalid_argument(
        "QIOLocalizer requires real-valued active RDMs.");
  }
  if (static_cast<std::size_t>(rdm_aa->rows()) != n ||
      static_cast<std::size_t>(rdm_aa->cols()) != n ||
      static_cast<std::size_t>(rdm_bb->rows()) != n ||
      static_cast<std::size_t>(rdm_bb->cols()) != n ||
      static_cast<std::size_t>(rdm_aabb->size()) != n * n * n * n) {
    throw std::invalid_argument(
        "Active RDM dimensions do not match the active-space size.");
  }

  Eigen::MatrixXd ga = *rdm_aa;
  Eigen::MatrixXd gb = *rdm_bb;
  std::vector<double> g2(rdm_aabb->data(), rdm_aabb->data() + rdm_aabb->size());

  // Minimize the single-orbital entropy sum -> accumulated rotation U.
  const auto max_cycles =
      static_cast<std::size_t>(_settings->get<int64_t>("max_cycles"));
  const double tol = _settings->get<double>("convergence_tolerance");
  const double coarse_step = _settings->get<double>("coarse_angle_step");
  const Eigen::MatrixXd u =
      optimize_rotation(ga, gb, g2, n, max_cycles, tol, coarse_step);

  // Apply the rotation to the active orbital columns (alpha == beta basis).
  const Eigen::MatrixXd& coeffs_alpha = orbitals->get_coefficients_alpha();
  Eigen::MatrixXd selected_coeffs(coeffs_alpha.rows(),
                                  static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) {
    selected_coeffs.col(static_cast<Eigen::Index>(i)) =
        coeffs_alpha.col(static_cast<Eigen::Index>(active_indices_a[i]));
  }
  const Eigen::MatrixXd rotated_coeffs = selected_coeffs * u;

  Eigen::MatrixXd coeffs = coeffs_alpha;
  for (std::size_t i = 0; i < n; ++i) {
    coeffs.col(static_cast<Eigen::Index>(active_indices_a[i])) =
        rotated_coeffs.col(static_cast<Eigen::Index>(i));
  }

  // Create output orbitals, preserving active/inactive metadata. Energies are
  // invalidated by the rotation.
  auto new_orbitals = std::make_shared<data::Orbitals>(
      coeffs,
      std::nullopt,  // no energies for entropy-optimized orbitals
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      orbitals->active_indices(), orbitals->inactive_indices());

  // Attach the rotated spin-traced active 1-RDM (alpha + beta) as a payload so
  // downstream consumers see the density in the new orbital basis. Unlike the
  // natural-orbital localizer, QIO does not diagonalize the 1-RDM, so this
  // payload is generally non-diagonal.
  const Eigen::MatrixXd rotated_one_rdm_spin_traced = ga + gb;
  return detail::new_aufbau_determinant_wavefunction(
      wavefunction, new_orbitals,
      data::ContainerTypes::MatrixVariant(rotated_one_rdm_spin_traced));
}

}  // namespace qdk::chemistry::algorithms::microsoft
