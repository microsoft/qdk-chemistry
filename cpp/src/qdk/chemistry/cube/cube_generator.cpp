// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <cstdint>
#include <filesystem>
#include <gauxc/basisset.hpp>
#include <gauxc/external/cube.hpp>
#include <gauxc/molecule.hpp>
#include <gauxc/orbital_evaluator.hpp>
#include <gauxc/shell.hpp>
#include <qdk/chemistry/cube/cube_generator.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <stdexcept>

namespace qdk::chemistry::cube {

CubeGrid CubeGrid::from_basis_set(const data::BasisSet& basis_set,
                                  std::size_t nx, std::size_t ny,
                                  std::size_t nz, double margin) {
  const auto structure = basis_set.get_structure();
  if (!structure)
    throw std::runtime_error("CubeGrid: basis set has no structure.");
  const Eigen::MatrixXd& coords = structure->get_coordinates();
  if (coords.rows() == 0)
    throw std::runtime_error("CubeGrid: structure has no atoms.");

  Eigen::Vector3d lo = coords.colwise().minCoeff();
  Eigen::Vector3d extent =
      (coords.colwise().maxCoeff() - lo).array() + 2.0 * margin;

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

namespace {

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
      const int l = sh.get_angular_momentum();
      const auto np = static_cast<int32_t>(sh.exponents.size());
      PA alpha{}, coeff{};
      for (int i = 0; i < np; ++i) {
        alpha[i] = sh.exponents[i];
        coeff[i] = sh.coefficients[i];
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
  return {{g.origin[0], g.origin[1], g.origin[2]},
          {g.spacing[0], g.spacing[1], g.spacing[2]},
          int64_t(g.nx),
          int64_t(g.ny),
          int64_t(g.nz)};
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
        gauxc_basis(to_gauxc_basis(*basis_set)),
        gauxc_mol(to_gauxc_mol(*basis_set->get_structure())),
        evaluator(gauxc_basis),
        nbf(gauxc_basis.nbf()) {
    for (auto& sh : gauxc_basis) sh.set_shell_tolerance(1e-12);
  }
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
    GauXC::write_cube(outfile, _impl->gauxc_mol, g, field.data(), comment);
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
    GauXC::write_cube(outfile, _impl->gauxc_mol, g, field.data(), comment);
  return field;
}

std::vector<std::string> generate_orbital_cubes(
    const data::Wavefunction& wfn, const std::vector<std::size_t>& indices,
    const std::string& output_dir, const CubeGrid& grid,
    const std::string& prefix) {
  CubeGenerator gen(wfn.get_orbitals()->get_basis_set());
  auto [C_a, C_b] = wfn.get_orbitals()->get_coefficients();
  std::filesystem::create_directories(output_dir);

  std::vector<std::string> paths;
  paths.reserve(indices.size());
  for (auto p : indices) {
    if (std::size_t(C_a.cols()) <= p)
      throw std::out_of_range("generate_orbital_cubes: index OOB.");
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s%04zu.cube", prefix.c_str(), p);
    auto path = (std::filesystem::path(output_dir) / buf).string();
    gen.orbital(C_a.col(p), path, grid,
                "Orbital " + std::to_string(p) + " (alpha)");
    paths.push_back(std::move(path));
  }
  return paths;
}

}  // namespace qdk::chemistry::cube
