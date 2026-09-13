// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <blas.hh>
#include <cmath>
#include <fstream>
#include <lapack.hh>
#include <limits>
#include <qdk/chemistry/data/lattice_geometry.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "hdf5_serialization.hpp"
#include "json_serialization.hpp"

namespace qdk::chemistry::data {
namespace {

bool same_distance(double lhs, double rhs, double tolerance) {
  return std::abs(lhs - rhs) <=
         tolerance * std::max(std::abs(lhs), std::abs(rhs));
}

Eigen::RowVector2d canonical_axis(const Eigen::RowVector2d& displacement,
                                  double distance, double tolerance) {
  Eigen::RowVector2d axis = displacement / distance;
  if (axis.x() < -tolerance ||
      (std::abs(axis.x()) <= tolerance && axis.y() < 0.0)) {
    axis = -axis;
  }
  return axis;
}

bool connection_less(const NeighborConnection& lhs,
                     const NeighborConnection& rhs) {
  return std::tie(lhs.bond_class.shell, lhs.bond_class.orientation, lhs.site_i,
                  lhs.site_j, lhs.image_shift) <
         std::tie(rhs.bond_class.shell, rhs.bond_class.orientation, rhs.site_i,
                  rhs.site_j, rhs.image_shift);
}

void validate_dimensions(const std::string& name, std::uint64_t nx,
                         std::uint64_t ny, bool periodic_x, bool periodic_y) {
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument(name + ": nx and ny must be > 0.");
  }
  if (periodic_x && nx < 2) {
    throw std::invalid_argument(name + ": periodic_x requires nx > 1.");
  }
  if (periodic_y && ny < 2) {
    throw std::invalid_argument(name + ": periodic_y requires ny > 1.");
  }
}

}  // namespace

LatticeGeometry::LatticeGeometry(Eigen::MatrixXd positions,
                                 std::optional<Eigen::MatrixXd> periods)
    : _positions(std::move(positions)), _periods(std::move(periods)) {
  if (_positions.cols() != 2 || !_positions.allFinite()) {
    throw std::invalid_argument(
        "Lattice positions must be a finite (num_sites, 2) matrix.");
  }
  if (!_periods.has_value()) return;
  if ((_periods->rows() != 1 && _periods->rows() != 2) ||
      _periods->cols() != 2 || !_periods->allFinite()) {
    throw std::invalid_argument(
        "Periodic vectors must be a finite (1, 2) or (2, 2) matrix.");
  }
  const double position_scale =
      _positions.size() == 0 ? 0.0 : _positions.cwiseAbs().maxCoeff();
  const double geometry_scale =
      std::max(position_scale, _periods->cwiseAbs().maxCoeff());
  for (Eigen::Index i = 0; i < _periods->rows(); ++i) {
    if (_periods->row(i).cwiseAbs().maxCoeff() == 0.0) {
      throw std::invalid_argument("Periodic vectors must be nonzero.");
    }
    if (!std::isfinite(std::hypot((*_periods)(i, 0), (*_periods)(i, 1)))) {
      throw std::invalid_argument("Periodic vector lengths must be finite.");
    }
    if ((_periods->row(i) / geometry_scale).cwiseAbs().maxCoeff() == 0.0) {
      throw std::invalid_argument(
          "Periodic vectors must be representable at the geometry scale.");
    }
  }
  if (_periods->rows() == 2) {
    const Eigen::RowVector2d p0 =
        _periods->row(0) / _periods->row(0).cwiseAbs().maxCoeff();
    const Eigen::RowVector2d p1 =
        _periods->row(1) / _periods->row(1).cwiseAbs().maxCoeff();
    if (p0.x() * p1.y() - p0.y() * p1.x() == 0.0) {
      throw std::invalid_argument(
          "Two periodic vectors must be linearly independent.");
    }
  }
}

const Eigen::MatrixXd& LatticeGeometry::positions() const { return _positions; }

const std::optional<Eigen::MatrixXd>& LatticeGeometry::periods() const {
  return _periods;
}

std::uint64_t LatticeGeometry::num_sites() const {
  return static_cast<std::uint64_t>(_positions.rows());
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>
LatticeGeometry::mth_nearest_neighbors(std::uint64_t m,
                                       double tolerance) const {
  return nearest_neighbor_shells({m}, tolerance).at(m);
}

std::map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>>
LatticeGeometry::nearest_neighbor_shells(
    const std::vector<std::uint64_t>& shells, double tolerance) const {
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    throw std::invalid_argument("Neighbor shell tolerance must be positive.");
  }
  std::map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>>
      results;
  for (std::uint64_t shell : shells) {
    if (shell == 0) {
      throw std::invalid_argument("Neighbor shell index m must be > 0.");
    }
    results.try_emplace(shell);
  }
  if (_periods.has_value()) {
    throw std::runtime_error(
        "Geometric neighbor shells support open lattices only.");
  }
  for (const auto& connection : neighbor_connections(shells, tolerance)) {
    results.at(connection.bond_class.shell)
        .emplace_back(connection.site_i, connection.site_j);
  }
  for (auto& [shell, pairs] : results) {
    (void)shell;
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  }
  return results;
}

std::vector<NeighborConnection> LatticeGeometry::_integer_neighbor_connections(
    const std::set<std::uint64_t>& requested_shells, double tolerance) const {
  const auto& embedding = *_integer_embedding;
  const int basis_size = static_cast<int>(embedding.basis.rows());
  const Eigen::RowVector2d a1 = embedding.primitive_vectors.row(0);
  const Eigen::RowVector2d a2 = embedding.primitive_vectors.row(1);
  const auto site_at = [&](int x, int y, int basis) {
    if (x < 0 || x >= embedding.nx || y < 0 || y >= embedding.ny) return -1;
    const int index = basis_size * (y * embedding.nx + x) + basis;
    return embedding.site_by_coordinate[index];
  };
  const auto wrap = [](int coordinate, int extent) {
    int image = coordinate / extent;
    int wrapped = coordinate % extent;
    if (wrapped < 0) {
      wrapped += extent;
      --image;
    }
    return std::pair{wrapped, image};
  };

  struct Stencil {
    int source_basis;
    int target_basis;
    int dx;
    int dy;
    double distance;
    std::uint64_t shell = 0;
    Eigen::RowVector2d axis;
    std::uint32_t orientation = 0;
  };
  std::vector<Stencil> stencils;
  int max_shell = 0;
  if (embedding.periodic_x || embedding.periodic_y) {
    const auto limit = static_cast<std::uint64_t>(
        std::numeric_limits<int>::max() - std::max(embedding.nx, embedding.ny));
    if (*requested_shells.rbegin() > limit) {
      throw std::overflow_error(
          "Neighbor connection stencil exceeds the supported integer range.");
    }
    max_shell = static_cast<int>(*requested_shells.rbegin());
  }
  const int min_dx = embedding.periodic_x ? -max_shell : 1 - embedding.nx;
  const int max_dx = embedding.periodic_x ? max_shell : embedding.nx - 1;
  const int min_dy = embedding.periodic_y ? -max_shell : 1 - embedding.ny;
  const int max_dy = embedding.periodic_y ? max_shell : embedding.ny - 1;
  for (int source_basis = 0; source_basis < basis_size; ++source_basis) {
    const Eigen::RowVector2d source_offset = embedding.basis.row(source_basis);
    for (int target_basis = 0; target_basis < basis_size; ++target_basis) {
      const Eigen::RowVector2d target_offset =
          embedding.basis.row(target_basis);
      for (int dy = min_dy; dy <= max_dy; ++dy) {
        for (int dx = min_dx; dx <= max_dx; ++dx) {
          const auto key = std::tie(dx, dy, source_basis, target_basis);
          const auto reverse =
              std::make_tuple(-dx, -dy, target_basis, source_basis);
          if (key >= reverse) continue;

          // Finite shells count only displacements realized by actual sites,
          // including the missing corners of an open plaquette patch.
          bool exists = false;
          const int y_begin = embedding.periodic_y ? 0 : std::max(0, -dy);
          const int y_end =
              embedding.ny - (embedding.periodic_y ? 0 : std::max(0, dy));
          const int x_begin = embedding.periodic_x ? 0 : std::max(0, -dx);
          const int x_end =
              embedding.nx - (embedding.periodic_x ? 0 : std::max(0, dx));
          for (int y = y_begin; !exists && y < y_end; ++y) {
            for (int x = x_begin; x < x_end; ++x) {
              const int target_x = embedding.periodic_x
                                       ? wrap(x + dx, embedding.nx).first
                                       : x + dx;
              const int target_y = embedding.periodic_y
                                       ? wrap(y + dy, embedding.ny).first
                                       : y + dy;
              if (site_at(x, y, source_basis) >= 0 &&
                  site_at(target_x, target_y, target_basis) >= 0) {
                exists = true;
                break;
              }
            }
          }
          if (!exists) continue;

          const Eigen::RowVector2d displacement = static_cast<double>(dx) * a1 +
                                                  static_cast<double>(dy) * a2 +
                                                  target_offset - source_offset;
          const double distance = blas::nrm2(2, displacement.data(), 1);
          if (!std::isfinite(distance)) {
            throw std::overflow_error(
                "Neighbor connection distance exceeds the supported range.");
          }
          if (distance == 0.0) continue;
          stencils.push_back({source_basis, target_basis, dx, dy, distance, 0,
                              canonical_axis(displacement, distance, tolerance),
                              0});
        }
      }
    }
  }

  std::sort(
      stencils.begin(), stencils.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.distance, lhs.dx, lhs.dy, lhs.source_basis,
                        lhs.target_basis) < std::tie(rhs.distance, rhs.dx,
                                                     rhs.dy, rhs.source_basis,
                                                     rhs.target_basis);
      });
  std::uint64_t shell = 0;
  double shell_distance = 0.0;
  std::map<std::uint64_t, std::vector<Eigen::RowVector2d>> axes_by_shell;
  for (auto& stencil : stencils) {
    if (shell == 0 ||
        !same_distance(stencil.distance, shell_distance, tolerance)) {
      ++shell;
      shell_distance = stencil.distance;
    }
    stencil.shell = shell;
    if (!requested_shells.contains(shell)) continue;
    auto& axes = axes_by_shell[shell];
    if (std::none_of(axes.begin(), axes.end(), [&](const auto& axis) {
          return (axis - stencil.axis).norm() <= tolerance;
        })) {
      axes.push_back(stencil.axis);
    }
  }
  for (auto& [current_shell, axes] : axes_by_shell) {
    (void)current_shell;
    std::sort(axes.begin(), axes.end(), [](const auto& lhs, const auto& rhs) {
      return std::atan2(lhs.y(), lhs.x()) < std::atan2(rhs.y(), rhs.x());
    });
  }
  for (auto& stencil : stencils) {
    if (!requested_shells.contains(stencil.shell)) continue;
    const auto& axes = axes_by_shell.at(stencil.shell);
    stencil.orientation = static_cast<std::uint32_t>(std::distance(
        axes.begin(),
        std::find_if(axes.begin(), axes.end(), [&](const auto& axis) {
          return (axis - stencil.axis).norm() <= tolerance;
        })));
  }

  std::vector<NeighborConnection> result;
  std::set<std::tuple<std::uint64_t, std::uint64_t, std::int64_t, std::int64_t>>
      seen;
  for (const auto& stencil : stencils) {
    if (!requested_shells.contains(stencil.shell)) continue;
    const int y_begin = embedding.periodic_y ? 0 : std::max(0, -stencil.dy);
    const int y_end =
        embedding.ny - (embedding.periodic_y ? 0 : std::max(0, stencil.dy));
    const int x_begin = embedding.periodic_x ? 0 : std::max(0, -stencil.dx);
    const int x_end =
        embedding.nx - (embedding.periodic_x ? 0 : std::max(0, stencil.dx));
    const Eigen::RowVector2d source_offset =
        embedding.basis.row(stencil.source_basis);
    const Eigen::RowVector2d target_offset =
        embedding.basis.row(stencil.target_basis);
    for (int y = y_begin; y < y_end; ++y) {
      for (int x = x_begin; x < x_end; ++x) {
        const auto [target_x, image_x] =
            embedding.periodic_x ? wrap(x + stencil.dx, embedding.nx)
                                 : std::pair{x + stencil.dx, 0};
        const auto [target_y, image_y] =
            embedding.periodic_y ? wrap(y + stencil.dy, embedding.ny)
                                 : std::pair{y + stencil.dy, 0};
        int site_i = site_at(x, y, stencil.source_basis);
        int site_j = site_at(target_x, target_y, stencil.target_basis);
        if (site_i < 0 || site_j < 0) continue;
        Eigen::RowVector2d displacement = static_cast<double>(stencil.dx) * a1 +
                                          static_cast<double>(stencil.dy) * a2 +
                                          target_offset - source_offset;
        std::array<std::int64_t, 2> image_shift{};
        if (embedding.periodic_x) image_shift[0] = image_x;
        if (embedding.periodic_y) {
          image_shift[embedding.periodic_x ? 1 : 0] = image_y;
        }
        if (site_i > site_j || (site_i == site_j &&
                                (image_shift[0] < 0 || (image_shift[0] == 0 &&
                                                        image_shift[1] < 0)))) {
          std::swap(site_i, site_j);
          displacement = -displacement;
          image_shift[0] = -image_shift[0];
          image_shift[1] = -image_shift[1];
        }
        if (!seen.emplace(site_i, site_j, image_shift[0], image_shift[1])
                 .second) {
          continue;
        }
        result.push_back({static_cast<std::uint64_t>(site_i),
                          static_cast<std::uint64_t>(site_j),
                          {stencil.shell, stencil.orientation, stencil.axis},
                          displacement,
                          image_shift,
                          std::nullopt,
                          1.0});
      }
    }
  }
  std::sort(result.begin(), result.end(), connection_less);
  return result;
}

std::vector<NeighborConnection> LatticeGeometry::neighbor_connections(
    const std::vector<std::uint64_t>& shells, double tolerance) const {
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    throw std::invalid_argument(
        "Neighbor connection tolerance must be positive.");
  }
  std::set<std::uint64_t> requested_shells;
  for (std::uint64_t shell : shells) {
    if (shell == 0) {
      throw std::invalid_argument("Neighbor shell index must be > 0.");
    }
    requested_shells.insert(shell);
  }
  const std::uint64_t n = num_sites();
  if (requested_shells.empty() || n == 0) return {};
  if (_integer_embedding.has_value()) {
    return _integer_neighbor_connections(requested_shells, tolerance);
  }

  struct Candidate {
    double distance;
    std::uint64_t site_i;
    std::uint64_t site_j;
    std::array<std::int64_t, 2> image_shift;
    Eigen::RowVector2d displacement;
    std::uint64_t shell = 0;
  };
  const Eigen::Index num_periods = _periods.has_value() ? _periods->rows() : 0;
  Eigen::MatrixXd positions = _positions;
  Eigen::MatrixXd periods = _periods.value_or(Eigen::MatrixXd(0, 2));
  double geometry_scale = positions.cwiseAbs().maxCoeff();
  if (num_periods != 0) {
    geometry_scale = std::max(geometry_scale, periods.cwiseAbs().maxCoeff());
  }
  if (geometry_scale == 0.0) geometry_scale = 1.0;
  // Binary scaling avoids overflow without rounding nearby coordinates apart.
  // Subtracting a common origin can erase pairs far from that origin.
  geometry_scale = std::scalbn(1.0, std::ilogb(geometry_scale));
  positions /= geometry_scale;
  periods /= geometry_scale;

  double minimum_period_scale = std::numeric_limits<double>::infinity();
  if (num_periods == 1) {
    minimum_period_scale =
        blas::nrm2(2, periods.row(0).data(), periods.outerStride());
  } else if (num_periods == 2) {
    Eigen::Matrix2d basis;
    basis.col(0) = periods.row(0).transpose();
    basis.col(1) = periods.row(1).transpose();
    std::array<double, 2> singular_values;
    double unused = 0.0;
    const auto info = lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec, 2,
                                    2, basis.data(), 2, singular_values.data(),
                                    &unused, 1, &unused, 1);
    if (info != 0) {
      throw std::runtime_error(
          "Failed to compute the periodic-vector singular values.");
    }
    minimum_period_scale =
        *std::min_element(singular_values.begin(), singular_values.end());
  }
  if (num_periods != 0 &&
      (!std::isfinite(minimum_period_scale) || minimum_period_scale <= 0.0)) {
    throw std::overflow_error(
        "Periodic vectors are not numerically independent at the geometry "
        "scale.");
  }
  const Eigen::RowVector2d position_extents{
      positions.col(0).maxCoeff() - positions.col(0).minCoeff(),
      positions.col(1).maxCoeff() - positions.col(1).minCoeff()};
  const double position_span = blas::nrm2(2, position_extents.data(), 1);
  const std::uint64_t max_requested_shell = *requested_shells.rbegin();

  std::vector<Candidate> candidates;
  for (std::int64_t radius = 0;; ++radius) {
    candidates.clear();
    std::set<
        std::tuple<std::uint64_t, std::uint64_t, std::int64_t, std::int64_t>>
        seen;
    const std::int64_t lower_0 = num_periods >= 1 ? -radius : 0;
    const std::int64_t upper_0 = num_periods >= 1 ? radius : 0;
    const std::int64_t lower_1 = num_periods == 2 ? -radius : 0;
    const std::int64_t upper_1 = num_periods == 2 ? radius : 0;
    for (std::uint64_t source = 0; source < n; ++source) {
      for (std::uint64_t target = 0; target < n; ++target) {
        for (std::int64_t image_0 = lower_0; image_0 <= upper_0; ++image_0) {
          for (std::int64_t image_1 = lower_1; image_1 <= upper_1; ++image_1) {
            if (source == target && image_0 == 0 && image_1 == 0) continue;
            std::uint64_t site_i = source;
            std::uint64_t site_j = target;
            std::array<std::int64_t, 2> image_shift = {image_0, image_1};
            if (site_i > site_j ||
                (site_i == site_j &&
                 (image_shift[0] < 0 ||
                  (image_shift[0] == 0 && image_shift[1] < 0)))) {
              std::swap(site_i, site_j);
              image_shift[0] = -image_shift[0];
              image_shift[1] = -image_shift[1];
            }
            if (!seen.emplace(site_i, site_j, image_shift[0], image_shift[1])
                     .second) {
              continue;
            }
            Eigen::RowVector2d displacement =
                positions.row(static_cast<Eigen::Index>(site_j)) -
                positions.row(static_cast<Eigen::Index>(site_i));
            if (num_periods >= 1) {
              displacement +=
                  static_cast<double>(image_shift[0]) * periods.row(0);
            }
            if (num_periods == 2) {
              displacement +=
                  static_cast<double>(image_shift[1]) * periods.row(1);
            }
            const double distance = blas::nrm2(2, displacement.data(), 1);
            if (distance != 0.0) {
              candidates.push_back(
                  {distance, site_i, site_j, image_shift, displacement});
            }
          }
        }
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) {
                if (lhs.distance != rhs.distance) {
                  return lhs.distance < rhs.distance;
                }
                return std::tie(lhs.site_i, lhs.site_j, lhs.image_shift) <
                       std::tie(rhs.site_i, rhs.site_j, rhs.image_shift);
              });
    std::uint64_t shell = 0;
    double shell_distance = 0.0;
    double requested_shell_distance = 0.0;
    for (auto& candidate : candidates) {
      if (shell == 0 ||
          !same_distance(candidate.distance, shell_distance, tolerance)) {
        ++shell;
        shell_distance = candidate.distance;
      }
      candidate.shell = shell;
      if (shell == max_requested_shell) {
        requested_shell_distance = shell_distance;
      }
    }
    if (num_periods == 0 ||
        (shell >= max_requested_shell &&
         minimum_period_scale * static_cast<double>(radius + 1) -
                 position_span >
             requested_shell_distance * (1.0 + tolerance))) {
      break;
    }
    if (radius == std::numeric_limits<std::int64_t>::max() - 1) {
      throw std::overflow_error(
          "Neighbor connection image radius exceeds the supported range.");
    }
  }

  std::map<std::uint64_t, std::vector<Eigen::RowVector2d>> shell_axes;
  for (const auto& candidate : candidates) {
    if (!requested_shells.contains(candidate.shell)) continue;
    const Eigen::RowVector2d axis =
        canonical_axis(candidate.displacement, candidate.distance, tolerance);
    auto& axes = shell_axes[candidate.shell];
    if (std::none_of(axes.begin(), axes.end(), [&](const auto& existing) {
          const Eigen::RowVector2d difference = existing - axis;
          return blas::nrm2(2, difference.data(), 1) <= tolerance;
        })) {
      axes.push_back(axis);
    }
  }
  for (auto& [shell, axes] : shell_axes) {
    (void)shell;
    std::sort(axes.begin(), axes.end(), [](const auto& lhs, const auto& rhs) {
      return std::atan2(lhs.y(), lhs.x()) < std::atan2(rhs.y(), rhs.x());
    });
  }
  std::vector<NeighborConnection> result;
  for (const auto& candidate : candidates) {
    if (!requested_shells.contains(candidate.shell)) continue;
    const Eigen::RowVector2d axis =
        canonical_axis(candidate.displacement, candidate.distance, tolerance);
    const auto& axes = shell_axes.at(candidate.shell);
    const auto orientation = static_cast<std::uint32_t>(std::distance(
        axes.begin(),
        std::find_if(axes.begin(), axes.end(), [&](const auto& existing) {
          const Eigen::RowVector2d difference = existing - axis;
          return blas::nrm2(2, difference.data(), 1) <= tolerance;
        })));
    const Eigen::RowVector2d displacement =
        candidate.displacement * geometry_scale;
    if (!displacement.allFinite()) {
      throw std::overflow_error(
          "Neighbor connection displacement exceeds the supported range.");
    }
    result.push_back({candidate.site_i,
                      candidate.site_j,
                      {candidate.shell, orientation, axes[orientation]},
                      displacement,
                      candidate.image_shift,
                      std::nullopt,
                      1.0});
  }
  std::sort(result.begin(), result.end(), connection_less);
  return result;
}

LatticeGeometry LatticeGeometry::_bravais(std::uint64_t nx, std::uint64_t ny,
                                          const Eigen::RowVector2d& a1,
                                          const Eigen::RowVector2d& a2,
                                          Eigen::MatrixXd basis,
                                          bool periodic_x, bool periodic_y,
                                          bool remove_open_corners) {
  const auto limit =
      static_cast<std::uint64_t>(std::numeric_limits<int>::max());
  const auto basis_size = static_cast<std::uint64_t>(basis.rows());
  if (nx > limit || ny > limit || nx > limit / ny / basis_size) {
    throw std::overflow_error(
        "Lattice dimensions exceed the supported integer site range.");
  }
  const int Nx = static_cast<int>(nx);
  const int Ny = static_cast<int>(ny);
  const int B = static_cast<int>(basis_size);
  const int full_num_sites = Nx * Ny * B;
  const int n = full_num_sites - (remove_open_corners ? 2 : 0);
  std::vector<int> site_by_coordinate(full_num_sites, -1);
  Eigen::MatrixXd positions(n, 2);
  int site = 0;
  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      for (int s = 0; s < B; ++s) {
        const int coordinate = B * (y * Nx + x) + s;
        if (remove_open_corners &&
            (coordinate == 0 || coordinate == full_num_sites - 1)) {
          continue;
        }
        const Eigen::RowVector2d offset = basis.row(s);
        positions.row(site) = x * a1 + y * a2 + offset;
        site_by_coordinate[coordinate] = site++;
      }
    }
  }
  std::optional<Eigen::MatrixXd> periods;
  if (periodic_x || periodic_y) {
    periods.emplace(static_cast<int>(periodic_x) + periodic_y, 2);
    if (periodic_x) periods->row(0) = Nx * a1;
    if (periodic_y) periods->row(periodic_x ? 1 : 0) = Ny * a2;
  }
  Eigen::Matrix2d primitive_vectors;
  primitive_vectors.row(0) = a1;
  primitive_vectors.row(1) = a2;
  LatticeGeometry geometry(std::move(positions), std::move(periods));
  geometry._integer_embedding = IntegerEmbedding{Nx,
                                                 Ny,
                                                 primitive_vectors,
                                                 std::move(basis),
                                                 std::move(site_by_coordinate),
                                                 periodic_x,
                                                 periodic_y};
  return geometry;
}

LatticeGeometry LatticeGeometry::chain(std::uint64_t n, bool periodic) {
  if (n == 0) {
    throw std::invalid_argument("chain: n must be > 0.");
  }
  return _bravais(n, 1, {1.0, 0.0}, {0.0, 0.0}, Eigen::MatrixXd::Zero(1, 2),
                  periodic, false);
}

LatticeGeometry LatticeGeometry::square(std::uint64_t nx, std::uint64_t ny,
                                        bool periodic_x, bool periodic_y) {
  validate_dimensions("square", nx, ny, periodic_x, periodic_y);
  return _bravais(nx, ny, {1.0, 0.0}, {0.0, 1.0}, Eigen::MatrixXd::Zero(1, 2),
                  periodic_x, periodic_y);
}

LatticeGeometry LatticeGeometry::triangular(std::uint64_t nx, std::uint64_t ny,
                                            bool periodic_x, bool periodic_y) {
  validate_dimensions("triangular", nx, ny, periodic_x, periodic_y);
  // Guo and Franz, Phys. Rev. B 80, 113102 (2009): a2 = u2 - u1 matches
  // the upper-right diagonal convention of the lattice graph factory.
  return _bravais(nx, ny, {1.0, 0.0}, {-0.5, std::sqrt(3.0) / 2.0},
                  Eigen::MatrixXd::Zero(1, 2), periodic_x, periodic_y);
}

LatticeGeometry LatticeGeometry::honeycomb(std::uint64_t nx, std::uint64_t ny,
                                           bool periodic_x, bool periodic_y) {
  validate_dimensions("honeycomb", nx, ny, periodic_x, periodic_y);
  // Castro Neto et al., Rev. Mod. Phys. 81, 109 (2009), Eq. (1), in units
  // of the nearest-neighbor distance.
  Eigen::MatrixXd basis(2, 2);
  basis << 0.0, 0.0, 1.0, 0.0;
  return _bravais(nx, ny, {1.5, std::sqrt(3.0) / 2.0},
                  {1.5, -std::sqrt(3.0) / 2.0}, std::move(basis), periodic_x,
                  periodic_y);
}

LatticeGeometry LatticeGeometry::honeycomb_plaquettes(std::uint64_t nx,
                                                      std::uint64_t ny,
                                                      bool periodic_x,
                                                      bool periodic_y) {
  validate_dimensions("honeycomb_plaquettes", nx, ny, periodic_x, periodic_y);
  if ((!periodic_x && nx == std::numeric_limits<std::uint64_t>::max()) ||
      (!periodic_y && ny == std::numeric_limits<std::uint64_t>::max())) {
    throw std::overflow_error(
        "Honeycomb boundary extension exceeds the site range.");
  }
  Eigen::MatrixXd basis(2, 2);
  basis << 0.0, 0.0, 1.0, 0.0;
  return _bravais(nx + static_cast<std::uint64_t>(!periodic_x),
                  ny + static_cast<std::uint64_t>(!periodic_y),
                  {1.5, std::sqrt(3.0) / 2.0}, {1.5, -std::sqrt(3.0) / 2.0},
                  std::move(basis), periodic_x, periodic_y,
                  !periodic_x && !periodic_y);
}

LatticeGeometry LatticeGeometry::kagome(std::uint64_t nx, std::uint64_t ny,
                                        bool periodic_x, bool periodic_y) {
  validate_dimensions("kagome", nx, ny, periodic_x, periodic_y);
  // Guo and Franz, Phys. Rev. B 80, 113102 (2009), Fig. 1: the Bravais
  // periods are twice their unit directions.
  Eigen::MatrixXd basis(3, 2);
  basis << 0.0, 0.0, 1.0, 0.0, 0.5, std::sqrt(3.0) / 2.0;
  return _bravais(nx, ny, {2.0, 0.0}, {1.0, std::sqrt(3.0)}, std::move(basis),
                  periodic_x, periodic_y);
}

LatticeGeometry LatticeGeometry::permute(
    const LatticeGeometry& geometry, const std::vector<std::uint64_t>& path) {
  const std::uint64_t n = geometry.num_sites();
  if (path.size() != n) {
    throw std::invalid_argument("Permutation must contain every lattice site.");
  }
  std::vector<std::uint64_t> inverse(n, n);
  Eigen::MatrixXd positions(static_cast<Eigen::Index>(n), 2);
  for (std::uint64_t i = 0; i < n; ++i) {
    if (path[i] >= n || inverse[path[i]] != n) {
      throw std::invalid_argument(
          "Permutation must contain each lattice site exactly once.");
    }
    inverse[path[i]] = i;
    positions.row(static_cast<Eigen::Index>(i)) =
        geometry._positions.row(static_cast<Eigen::Index>(path[i]));
  }
  LatticeGeometry result(std::move(positions), geometry._periods);
  result._integer_embedding = geometry._integer_embedding;
  if (result._integer_embedding.has_value()) {
    for (int& site : result._integer_embedding->site_by_coordinate) {
      if (site >= 0) site = static_cast<int>(inverse[site]);
    }
  }
  return result;
}

std::string LatticeGeometry::get_summary() const {
  std::ostringstream summary;
  summary << "LatticeGeometry\n  Sites: " << num_sites()
          << "\n  Periodic vectors: " << (_periods ? _periods->rows() : 0);
  return summary.str();
}

void LatticeGeometry::to_file(const std::string& filename,
                              const std::string& type) const {
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unknown file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

nlohmann::json LatticeGeometry::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;
  j["positions"] = matrix_to_json(_positions);
  if (_periods.has_value()) j["periods"] = matrix_to_json(*_periods);
  return j;
}

void LatticeGeometry::to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }
  file << to_json().dump(2);
  if (file.fail()) {
    throw std::runtime_error("Error writing to file: " + filename);
  }
}

void LatticeGeometry::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    save_matrix_to_group(group, "positions", _positions);
    if (_periods.has_value()) save_matrix_to_group(group, "periods", *_periods);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in LatticeGeometry::to_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

void LatticeGeometry::to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    auto group = file.openGroup("/");
    to_hdf5(group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

LatticeGeometry LatticeGeometry::from_file(const std::string& filename,
                                           const std::string& type) {
  if (type == "json") return from_json_file(filename);
  if (type == "hdf5") return from_hdf5_file(filename);
  throw std::invalid_argument("Unknown file type: " + type +
                              ". Supported types are: json, hdf5");
}

LatticeGeometry LatticeGeometry::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  const auto read_matrix = [](const nlohmann::json& value) -> Eigen::MatrixXd {
    if (!value.is_array() || value.size() > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(
          "Geometry matrices require arrays of two-column rows.");
    }
    // The shared converter rejects empty arrays; geometry has a fixed width.
    if (value.empty()) return Eigen::MatrixXd(0, 2);
    for (const auto& row : value) {
      if (!row.is_array() || row.size() != 2) {
        throw std::invalid_argument("Geometry matrices must have two columns.");
      }
    }
    return json_to_matrix(value);
  };
  if (!j.is_object() || !j.contains("positions")) {
    throw std::runtime_error("JSON missing required 'positions' field.");
  }
  std::optional<Eigen::MatrixXd> periods;
  if (j.contains("periods")) periods = read_matrix(j.at("periods"));
  return LatticeGeometry(read_matrix(j.at("positions")), std::move(periods));
}

LatticeGeometry LatticeGeometry::from_json_file(const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open LatticeGeometry JSON file: " +
                             filename);
  }
  nlohmann::json j;
  file >> j;
  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }
  return from_json(j);
}

LatticeGeometry LatticeGeometry::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    const auto read_matrix = [](H5::Group& source, const std::string& name) {
      auto dataset = source.openDataSet(name);
      auto space = dataset.getSpace();
      if (space.getSimpleExtentNdims() != 2) {
        throw std::invalid_argument(
            "Geometry matrix dataset must have rank two: " + name);
      }
      hsize_t dimensions[2];
      space.getSimpleExtentDims(dimensions);
      if (dimensions[1] != 2 ||
          dimensions[0] > std::numeric_limits<int>::max() ||
          (dataset.getTypeClass() != H5T_FLOAT &&
           dataset.getTypeClass() != H5T_INTEGER)) {
        throw std::invalid_argument("Invalid geometry matrix shape or type: " +
                                    name);
      }
      return load_matrix_from_group(source, name);
    };
    std::optional<Eigen::MatrixXd> periods;
    if (group.nameExists("periods")) periods = read_matrix(group, "periods");
    return LatticeGeometry(read_matrix(group, "positions"), std::move(periods));
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in LatticeGeometry::from_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

LatticeGeometry LatticeGeometry::from_hdf5_file(const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    auto group = file.openGroup("/");
    return from_hdf5(group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to read LatticeGeometry HDF5 file '" +
                             filename + "': " + e.getCDetailMsg());
  }
}

void LatticeGeometry::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, _positions);
  hash_value(ctx, _periods.has_value());
  if (_periods.has_value()) hash_value(ctx, *_periods);
}

}  // namespace qdk::chemistry::data
