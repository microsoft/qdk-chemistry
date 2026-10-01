// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <blas.hh>
#include <cmath>
#include <fstream>
#include <limits>
#include <qdk/chemistry/data/lattice_geometry.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

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
                                 std::optional<Eigen::MatrixXd> periods,
                                 IntegerEmbedding embedding)
    : _positions(std::move(positions)),
      _periods(std::move(periods)),
      _embedding(std::move(embedding)) {}

const Eigen::MatrixXd& LatticeGeometry::positions() const { return _positions; }

const std::optional<Eigen::MatrixXd>& LatticeGeometry::periods() const {
  return _periods;
}

std::uint64_t LatticeGeometry::num_sites() const {
  return static_cast<std::uint64_t>(_positions.rows());
}

std::vector<LatticeGeometry::ShellBond> LatticeGeometry::_shell_bonds(
    const std::vector<std::uint64_t>& shells, double tolerance) const {
  std::set<std::uint64_t> requested_shells;
  for (std::uint64_t shell : shells) {
    if (shell == 0) {
      throw std::invalid_argument("Neighbor shell index must be > 0.");
    }
    requested_shells.insert(shell);
  }
  if (!std::isfinite(tolerance) || tolerance <= 0.0 || tolerance >= 1.0) {
    throw std::invalid_argument(
        "Neighbor connection tolerance must be positive and less than 1.");
  }
  if (requested_shells.empty()) return {};
  const auto& embedding = _embedding;
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
  };
  const auto enumerate = [&](int window) {
    std::vector<Stencil> stencils;
    const int min_dx = embedding.periodic_x ? -window : 1 - embedding.nx;
    const int max_dx = embedding.periodic_x ? window : embedding.nx - 1;
    const int min_dy = embedding.periodic_y ? -window : 1 - embedding.ny;
    const int max_dy = embedding.periodic_y ? window : embedding.ny - 1;
    for (int source_basis = 0; source_basis < basis_size; ++source_basis) {
      const Eigen::RowVector2d source_offset =
          embedding.basis.row(source_basis);
      for (int target_basis = 0; target_basis < basis_size; ++target_basis) {
        const Eigen::RowVector2d target_offset =
            embedding.basis.row(target_basis);
        for (int dy = min_dy; dy <= max_dy; ++dy) {
          for (int dx = min_dx; dx <= max_dx; ++dx) {
            if (dx > 0 ||
                (dx == 0 &&
                 (dy > 0 || (dy == 0 && source_basis >= target_basis)))) {
              continue;
            }

            // Finite shells count only displacements realized by actual
            // sites, including the missing corners of an open plaquette patch.
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

            const Eigen::RowVector2d displacement =
                static_cast<double>(dx) * a1 + static_cast<double>(dy) * a2 +
                target_offset - source_offset;
            const double distance = blas::nrm2(2, displacement.data(), 1);
            if (!std::isfinite(distance)) {
              throw std::overflow_error(
                  "Neighbor connection distance exceeds the supported range.");
            }
            if (distance == 0.0) continue;
            stencils.push_back(
                {source_basis, target_basis, dx, dy, distance, 0,
                 canonical_axis(displacement, distance, tolerance)});
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
    for (auto& stencil : stencils) {
      if (shell == 0 ||
          !same_distance(stencil.distance, shell_distance, tolerance)) {
        ++shell;
        shell_distance = stencil.distance;
      }
      stencil.shell = shell;
    }
    return stencils;
  };

  const std::uint64_t max_shell = *requested_shells.rbegin();
  std::vector<Stencil> stencils;
  if (!embedding.periodic_x && !embedding.periodic_y) {
    stencils = enumerate(0);
  } else {
    const auto limit = static_cast<std::uint64_t>(
        std::numeric_limits<int>::max() - std::max(embedding.nx, embedding.ny));
    const auto checked_window = [limit](double cells) {
      if (!(cells <= static_cast<double>(limit))) {
        throw std::overflow_error(
            "Neighbor connection stencil exceeds the supported integer range.");
      }
      return static_cast<int>(std::ceil(cells));
    };
    double spread = 0.0;
    for (int source = 0; source < basis_size; ++source) {
      for (int target = 0; target < basis_size; ++target) {
        spread = std::max(
            spread,
            (embedding.basis.row(target) - embedding.basis.row(source)).norm());
      }
    }
    const double smallest_singular_value =
        Eigen::JacobiSVD<Eigen::Matrix2d>(embedding.primitive_vectors)
            .singularValues()(1);
    // Cell offsets reaching distance r are at most (r + spread) / s_min.
    const auto offsets_within = [&](double radius) {
      const double lattice = (radius + spread) / smallest_singular_value;
      if (embedding.periodic_x && embedding.periodic_y) return lattice;
      // The open direction's offset is bounded by the patch instead.
      const double direct =
          embedding.periodic_x
              ? (radius + spread + (embedding.ny - 1) * a2.norm()) / a1.norm()
              : (radius + spread + (embedding.nx - 1) * a1.norm()) / a2.norm();
      return std::min(lattice, direct);
    };
    int window = checked_window(static_cast<double>(max_shell));
    while (true) {
      stencils = enumerate(window);
      const auto first = std::find_if(stencils.begin(), stencils.end(),
                                      [max_shell](const auto& stencil) {
                                        return stencil.shell == max_shell;
                                      });
      if (first == stencils.end()) {
        window = checked_window(2.0 * window);
        continue;
      }
      // Relative merging extends a shell to its first distance / (1 - tol).
      const int needed =
          checked_window(offsets_within(first->distance / (1.0 - tolerance)));
      if (needed <= window) break;
      window = needed;
    }
  }

  std::vector<ShellBond> result;
  for (const auto& stencil : stencils) {
    if (!requested_shells.contains(stencil.shell)) continue;
    const int y_begin = embedding.periodic_y ? 0 : std::max(0, -stencil.dy);
    const int y_end =
        embedding.ny - (embedding.periodic_y ? 0 : std::max(0, stencil.dy));
    const int x_begin = embedding.periodic_x ? 0 : std::max(0, -stencil.dx);
    const int x_end =
        embedding.nx - (embedding.periodic_x ? 0 : std::max(0, stencil.dx));
    for (int y = y_begin; y < y_end; ++y) {
      for (int x = x_begin; x < x_end; ++x) {
        const int target_x = embedding.periodic_x
                                 ? wrap(x + stencil.dx, embedding.nx).first
                                 : x + stencil.dx;
        const int target_y = embedding.periodic_y
                                 ? wrap(y + stencil.dy, embedding.ny).first
                                 : y + stencil.dy;
        int site_i = site_at(x, y, stencil.source_basis);
        int site_j = site_at(target_x, target_y, stencil.target_basis);
        if (site_i < 0 || site_j < 0) continue;
        if (site_i > site_j) std::swap(site_i, site_j);
        result.push_back({static_cast<std::uint64_t>(site_i),
                          static_cast<std::uint64_t>(site_j), stencil.shell,
                          stencil.axis});
      }
    }
  }
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
  const auto full_num_sites = static_cast<int>(nx * ny * basis_size);
  std::vector<int> site_by_coordinate(full_num_sites);
  int site = 0;
  for (int coordinate = 0; coordinate < full_num_sites; ++coordinate) {
    const bool removed = remove_open_corners &&
                         (coordinate == 0 || coordinate == full_num_sites - 1);
    site_by_coordinate[coordinate] = removed ? -1 : site++;
  }
  Eigen::MatrixXd primitive_vectors(2, 2);
  primitive_vectors << a1, a2;
  return _from_integer_embedding(
      static_cast<int>(nx), static_cast<int>(ny), primitive_vectors,
      std::move(basis), std::move(site_by_coordinate), periodic_x, periodic_y);
}

LatticeGeometry LatticeGeometry::_from_integer_embedding(
    int nx, int ny, const Eigen::MatrixXd& primitive_vectors,
    Eigen::MatrixXd basis, std::vector<int> site_by_coordinate, bool periodic_x,
    bool periodic_y) {
  const auto num_basis = static_cast<std::uint64_t>(basis.rows());
  const auto limit =
      static_cast<std::uint64_t>(std::numeric_limits<int>::max());
  if (nx < 1 || ny < 1 || num_basis == 0 || basis.cols() != 2 ||
      primitive_vectors.rows() != 2 || primitive_vectors.cols() != 2 ||
      static_cast<std::uint64_t>(nx) >
          limit / static_cast<std::uint64_t>(ny) / num_basis ||
      site_by_coordinate.size() != static_cast<std::uint64_t>(nx) *
                                       static_cast<std::uint64_t>(ny) *
                                       num_basis) {
    throw std::invalid_argument("Invalid lattice integer embedding.");
  }
  const auto n = static_cast<std::size_t>(
      std::count_if(site_by_coordinate.begin(), site_by_coordinate.end(),
                    [](int site) { return site != -1; }));
  if (n == 0) {
    throw std::invalid_argument(
        "Integer embedding must keep at least one site.");
  }
  const Eigen::RowVector2d a1 = primitive_vectors.row(0);
  const Eigen::RowVector2d a2 = primitive_vectors.row(1);
  Eigen::MatrixXd positions(static_cast<Eigen::Index>(n), 2);
  std::vector<bool> seen(n, false);
  std::size_t coordinate = 0;
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      for (Eigen::Index s = 0; s < basis.rows(); ++s, ++coordinate) {
        const int site = site_by_coordinate[coordinate];
        if (site == -1) continue;
        // The stencil search indexes positions through this map.
        if (site < 0 || static_cast<std::size_t>(site) >= n || seen[site]) {
          throw std::invalid_argument(
              "Integer embedding sites must form a permutation.");
        }
        seen[site] = true;
        const Eigen::RowVector2d offset = basis.row(s);
        positions.row(site) = x * a1 + y * a2 + offset;
      }
    }
  }
  std::optional<Eigen::MatrixXd> periods;
  if (periodic_x || periodic_y) {
    periods.emplace(static_cast<int>(periodic_x) + periodic_y, 2);
    if (periodic_x) periods->row(0) = nx * a1;
    if (periodic_y) periods->row(periodic_x ? 1 : 0) = ny * a2;
  }
  // Serialized layouts are external input; reject degenerate supercells.
  // Every direction the patch spans needs a nonzero, independent vector.
  const bool spans_x = periodic_x || nx > 1;
  const bool spans_y = periodic_y || ny > 1;
  if (!positions.allFinite() || (periods && !periods->allFinite()) ||
      (spans_x && a1.isZero(0.0)) || (spans_y && a2.isZero(0.0)) ||
      (spans_x && spans_y && a1.x() * a2.y() == a1.y() * a2.x())) {
    throw std::invalid_argument("Invalid lattice integer embedding.");
  }
  return LatticeGeometry(
      std::move(positions), std::move(periods),
      IntegerEmbedding{nx, ny, primitive_vectors, std::move(basis),
                       std::move(site_by_coordinate), periodic_x, periodic_y});
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
  j["version"] = SERIALIZATION_VERSION;
  j["integer_embedding"] = {
      {"nx", _embedding.nx},
      {"ny", _embedding.ny},
      {"primitive_vectors", matrix_to_json(_embedding.primitive_vectors)},
      {"basis", matrix_to_json(_embedding.basis)},
      {"site_by_coordinate", _embedding.site_by_coordinate},
      {"periodic_x", _embedding.periodic_x},
      {"periodic_y", _embedding.periodic_y}};
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
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    group.createAttribute("version", string_type, H5::DataSpace(H5S_SCALAR))
        .write(string_type, std::string(SERIALIZATION_VERSION));
    auto layout = group.createGroup("integer_embedding");
    save_stl_to_group(
        layout, "shape",
        std::vector<int>{_embedding.nx, _embedding.ny, _embedding.periodic_x,
                         _embedding.periodic_y});
    save_matrix_to_group(layout, "primitive_vectors",
                         _embedding.primitive_vectors);
    save_matrix_to_group(layout, "basis", _embedding.basis);
    save_stl_to_group(layout, "site_by_coordinate",
                      _embedding.site_by_coordinate);
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
  if (!j.contains("version")) {
    throw std::runtime_error("Invalid JSON: missing version field");
  }
  validate_serialization_version(SERIALIZATION_VERSION,
                                 j.at("version").get<std::string>());
  if (!j.contains("integer_embedding") ||
      !j.at("integer_embedding").is_object()) {
    throw std::invalid_argument("Invalid lattice integer embedding.");
  }
  const auto& embedding = j.at("integer_embedding");
  const auto& sites = embedding.at("site_by_coordinate");
  if (!sites.is_array()) {
    throw std::invalid_argument("Invalid lattice integer embedding.");
  }
  std::vector<int> site_by_coordinate;
  site_by_coordinate.reserve(sites.size());
  for (const auto& site : sites) {
    site_by_coordinate.push_back(detail::json_integer<int>(site));
  }
  return _from_integer_embedding(
      detail::json_integer<int>(embedding.at("nx")),
      detail::json_integer<int>(embedding.at("ny")),
      json_to_matrix(embedding.at("primitive_vectors")),
      json_to_matrix(embedding.at("basis")), std::move(site_by_coordinate),
      embedding.at("periodic_x").get<bool>(),
      embedding.at("periodic_y").get<bool>());
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
      if (dimensions[1] == 0 ||
          dimensions[1] > std::numeric_limits<int>::max() ||
          dimensions[0] > std::numeric_limits<int>::max() ||
          (dataset.getTypeClass() != H5T_FLOAT &&
           dataset.getTypeClass() != H5T_INTEGER)) {
        throw std::invalid_argument("Invalid geometry matrix shape or type: " +
                                    name);
      }
      return load_matrix_from_group(source, name);
    };
    const auto read_ints = [](H5::Group& source, const std::string& name) {
      // The shared loader reads one extent and converts any numeric type.
      const auto dataset = source.openDataSet(name);
      if (dataset.getSpace().getSimpleExtentNdims() != 1 ||
          dataset.getTypeClass() != H5T_INTEGER) {
        throw std::invalid_argument(
            "Integer embedding dataset must be a rank-one integer array: " +
            name);
      }
      return load_std_vector_from_group<int>(source, name);
    };
    if (!group.attrExists("version")) {
      throw std::runtime_error(
          "HDF5 group missing required 'version' attribute");
    }
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    std::string version;
    group.openAttribute("version").read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);
    if (!group.nameExists("integer_embedding")) {
      throw std::invalid_argument("Invalid lattice integer embedding.");
    }
    auto layout = group.openGroup("integer_embedding");
    const auto shape = read_ints(layout, "shape");
    if (shape.size() != 4 || (shape[2] != 0 && shape[2] != 1) ||
        (shape[3] != 0 && shape[3] != 1)) {
      throw std::invalid_argument("Invalid lattice integer embedding.");
    }
    return _from_integer_embedding(
        shape[0], shape[1], read_matrix(layout, "primitive_vectors"),
        read_matrix(layout, "basis"), read_ints(layout, "site_by_coordinate"),
        shape[2] == 1, shape[3] == 1);
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
