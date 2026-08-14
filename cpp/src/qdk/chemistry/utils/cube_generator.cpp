// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <gauxc/basisset.hpp>
#include <gauxc/external/cube.hpp>
#include <gauxc/molecule.hpp>
#include <gauxc/orbital_evaluator.hpp>
#include <gauxc/shell.hpp>
#include <limits>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/utils/cube_generator.hpp>
#include <stdexcept>

namespace qdk::chemistry::utils {

CubeGrid CubeGrid::from_basis_set(const data::BasisSet& basis_set,
                                  std::size_t nx, std::size_t ny,
                                  std::size_t nz, double margin) {
  if (nx == 0 || ny == 0 || nz == 0)
    throw std::invalid_argument("CubeGrid: dimensions must be positive.");
  if (margin < 0.0)
    throw std::invalid_argument("CubeGrid: margin cannot be negative.");
  const auto structure = basis_set.get_structure();
  if (!structure)
    throw std::runtime_error("CubeGrid: basis set has no structure.");
  const Eigen::MatrixXd& coords = structure->get_coordinates();
  if (coords.rows() == 0)
    throw std::runtime_error("CubeGrid: structure has no atoms.");

  Eigen::Vector3d lo = coords.colwise().minCoeff().transpose();
  Eigen::Vector3d hi = coords.colwise().maxCoeff().transpose();
  Eigen::Vector3d extent = (hi - lo).array() + 2.0 * margin;

  CubeGrid g;
  g.origin = lo.array() - margin;
  g.nx = nx;
  g.ny = ny;
  g.nz = nz;
  g.spacing[0] = nx > 1 ? extent[0] / double(nx - 1) : 0.0;
  g.spacing[1] = ny > 1 ? extent[1] / double(ny - 1) : 0.0;
  g.spacing[2] = nz > 1 ? extent[2] / double(nz - 1) : 0.0;
  return g;
}

std::size_t CubeGrid::num_points() const {
  if (nx == 0 || ny == 0 || nz == 0)
    throw std::invalid_argument("CubeGrid: dimensions must be positive.");
  // gauXC hands the point count to BLAS gemm as a leading dimension, and that
  // parameter is a plain int, so INT_MAX points is a hard backend limit rather
  // than a choice. Checking it here keeps the rejection ahead of the caller's
  // allocation, which would otherwise reserve eight bytes per point for a grid
  // that gauXC was always going to refuse.
  constexpr auto max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (nx > max / ny || nx * ny > max / nz)
    throw std::overflow_error(
        "CubeGrid: point count exceeds gauXC's limit of 2147483647 points.");
  return nx * ny * nz;
}

namespace {

const data::BasisSet& require_basis_set(
    const std::shared_ptr<data::BasisSet>& basis_set) {
  if (!basis_set)
    throw std::invalid_argument("CubeGenerator: basis set cannot be null.");
  if (!basis_set->get_structure())
    throw std::invalid_argument("CubeGenerator: basis set has no structure.");
  return *basis_set;
}

GauXC::BasisSet<double> to_gauxc_basis(const data::BasisSet& qdk) {
  using PA = GauXC::Shell<double>::prim_array;
  using CA = GauXC::Shell<double>::cart_array;

  const auto st = qdk.get_structure();
  const Eigen::MatrixXd& coords = st->get_coordinates();
  const bool sph = qdk.get_atomic_orbital_type() == data::AOType::Spherical;

  GauXC::BasisSet<double> basis;
  for (std::size_t ia = 0; ia < qdk.get_num_atoms(); ++ia) {
    CA center{coords(ia, 0), coords(ia, 1), coords(ia, 2)};
    for (const auto& sh : qdk.get_shells_for_atom(ia)) {
      // All-zero radial powers are plain r^0 Gaussians, which
      // from_basis_name uses to represent ordinary electron shells.
      if (sh.has_radial_powers() && (sh.rpowers.array() != 0).any())
        throw std::invalid_argument(
            "CubeGenerator: radial-power shells are unsupported.");
      const int l = sh.get_angular_momentum();
      if (l < 0)
        throw std::invalid_argument(
            "CubeGenerator: ECP potential shells are unsupported.");
      const auto np = static_cast<int32_t>(sh.exponents.size());
      PA alpha{}, coeff{};
      if (static_cast<std::size_t>(np) > alpha.size())
        throw std::invalid_argument(
            "CubeGenerator: shell exceeds gauXC's primitive limit.");
      for (int i = 0; i < np; ++i) {
        alpha.at(i) = sh.exponents[i];
        coeff.at(i) = sh.coefficients[i];
      }
      basis.emplace_back(GauXC::PrimSize(np), GauXC::AngularMomentum(l),
                         GauXC::SphericalType(l > 1 && sph), alpha, coeff,
                         center, true);
    }
  }
  return basis;
}

GauXC::Molecule to_gauxc_mol(const data::Structure& st) {
  const auto& coords = st.get_coordinates();
  const auto& elems = st.get_elements();
  GauXC::Molecule mol;
  for (Eigen::Index i = 0; i < coords.rows(); ++i)
    mol.push_back({GauXC::AtomicNumber(int64_t(elems[i])), coords(i, 0),
                   coords(i, 1), coords(i, 2)});
  return mol;
}

GauXC::CubeGrid to_gauxc_grid(const CubeGrid& g) {
  g.num_points();
  return {{g.origin[0], g.origin[1], g.origin[2]},
          {g.spacing[0], g.spacing[1], g.spacing[2]},
          int64_t(g.nx),
          int64_t(g.ny),
          int64_t(g.nz)};
}

// Records ECP provenance in the cube comment so that a valence-only field is
// self-describing once the file leaves this process. Fields evaluated from an
// ECP basis omit core density by construction.
std::string annotate_comment(const data::BasisSet& basis_set,
                             const std::string& comment) {
  if (!basis_set.has_ecp_electrons()) return comment;
  std::size_t core_electrons = 0;
  for (std::size_t n : basis_set.get_ecp_electrons()) core_electrons += n;
  if (core_electrons == 0) return comment;
  const std::string note = "valence-only: ECP replaces " +
                           std::to_string(core_electrons) + " core electrons";
  return comment.empty() ? note : comment + " [" + note + "]";
}

}  // namespace

struct CubeGenerator::Impl {
  std::shared_ptr<data::BasisSet> basis_set;
  GauXC::BasisSet<double> gauxc_basis;
  GauXC::Molecule gauxc_mol;
  GauXC::OrbitalEvaluator evaluator;
  int32_t nbf;

  explicit Impl(std::shared_ptr<data::BasisSet> bs)
      : basis_set(std::move(bs)),
        gauxc_basis(to_gauxc_basis(require_basis_set(basis_set))),
        gauxc_mol(to_gauxc_mol(*basis_set->get_structure())),
        evaluator(GauXC::OrbitalEvaluatorFactory::make_orbital_evaluator(
            GauXC::ExecutionSpace::Host, gauxc_basis)),
        nbf(gauxc_basis.nbf()) {}
};

CubeGenerator::CubeGenerator(std::shared_ptr<data::BasisSet> bs)
    : _impl(std::make_unique<Impl>(std::move(bs))) {}
CubeGenerator::~CubeGenerator() noexcept = default;
CubeGenerator::CubeGenerator(CubeGenerator&&) noexcept = default;
CubeGenerator& CubeGenerator::operator=(CubeGenerator&&) noexcept = default;

CubeField CubeGenerator::orbital(const Eigen::VectorXd& C,
                                 const std::string& outfile,
                                 const CubeGrid& grid,
                                 const std::string& comment) const {
  if (C.size() != _impl->nbf)
    throw std::invalid_argument("orbital: mo_coeff length mismatch.");
  auto g = to_gauxc_grid(grid);
  CubeField field(g.num_points());
  _impl->evaluator.eval_orbital(g, C.data(), field.data());
  if (!outfile.empty())
    GauXC::write_cube(outfile, _impl->gauxc_mol, g, field.data(),
                      annotate_comment(*_impl->basis_set, comment));
  return field;
}

CubeField CubeGenerator::density(const Eigen::MatrixXd& D,
                                 const std::string& outfile,
                                 const CubeGrid& grid,
                                 const std::string& comment) const {
  if (D.rows() != _impl->nbf || D.cols() != _impl->nbf)
    throw std::invalid_argument("density: matrix shape mismatch.");
  auto g = to_gauxc_grid(grid);
  CubeField field(g.num_points());
  _impl->evaluator.eval_density(g, D.data(), _impl->nbf, field.data());
  if (!outfile.empty())
    GauXC::write_cube(outfile, _impl->gauxc_mol, g, field.data(),
                      annotate_comment(*_impl->basis_set, comment));
  return field;
}

std::vector<std::string> generate_orbital_cubes(
    const data::Wavefunction& wfn, const std::vector<std::size_t>& indices,
    const std::string& output_dir, const CubeGrid& grid,
    const std::string& prefix) {
  const auto orbitals = wfn.get_orbitals();
  CubeGenerator gen(orbitals->get_basis_set());
  const auto coeffs = orbitals->coefficients();
  const auto& C_a = coeffs->block({data::axes::alpha(), data::axes::alpha()});
  const bool restricted = orbitals->is_restricted();
  std::filesystem::create_directories(output_dir);

  // Zero-based orbital numbering, consistent with the 0-based `indices`
  // argument and the rest of qdk-chemistry (cf. coupled_cluster frozen-orbital
  // indices and data/symmetry spin_channel_indices). Note this differs from
  // the Python `cubegen.py` layer, which labels files 1-based.
  auto index_string = [](std::size_t p) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04zu", p);
    return std::string(buf);
  };

  std::vector<std::string> paths;
  paths.reserve(indices.size() * (restricted ? 1u : 2u));

  auto emit = [&](const Eigen::VectorXd& coeff, const std::string& filename,
                  const std::string& comment) {
    auto path = (std::filesystem::path(output_dir) / filename).string();
    gen.orbital(coeff, path, grid, comment);
    paths.push_back(std::move(path));
  };

  for (auto p : indices) {
    if (std::size_t(C_a.cols()) <= p)
      throw std::out_of_range("generate_orbital_cubes: index OOB.");
    const auto stem = prefix + index_string(p);
    // Restricted: a single spatial cube with no spin suffix. Unrestricted:
    // separate alpha (`_a`) and beta (`_b`) cubes, mirroring `cubegen.py`.
    if (restricted) {
      emit(C_a.col(p), stem + ".cube",
           "Orbital " + std::to_string(p) + " (alpha)");
    } else {
      const auto& C_b = coeffs->block({data::axes::beta(), data::axes::beta()});
      if (std::size_t(C_b.cols()) <= p)
        throw std::out_of_range("generate_orbital_cubes: index OOB.");
      emit(C_a.col(p), stem + "_a.cube",
           "Orbital " + std::to_string(p) + " (alpha)");
      emit(C_b.col(p), stem + "_b.cube",
           "Orbital " + std::to_string(p) + " (beta)");
    }
  }
  return paths;
}

}  // namespace qdk::chemistry::utils
