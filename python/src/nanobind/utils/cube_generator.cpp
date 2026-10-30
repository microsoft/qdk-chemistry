// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <memory>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/utils/cube_generator.hpp>
#include <string>
#include <utility>
#include <vector>

#include "eigen.hpp"
#include "stl.hpp"

namespace nb = nanobind;

using qdk::chemistry::data::BasisSet;
using qdk::chemistry::data::Orbitals;
using qdk::chemistry::utils::CubeField;
using qdk::chemistry::utils::CubeGenerator;
using qdk::chemistry::utils::CubeGrid;

namespace {

// Hands the evaluated field to NumPy without copying it.
//
// The field is moved onto the heap and its lifetime is tied to a capsule, so
// the returned array owns the buffer. gauXC documents the layout as row-major
// (ix, iy, iz) with iz varying fastest, which is exactly C order for shape
// (nx, ny, nz), so the flat buffer can be reshaped without any permutation.
nb::ndarray<nb::numpy, double, nb::ndim<3>> field_to_array(
    CubeField&& field, const CubeGrid& grid) {
  auto* owned = new CubeField(std::move(field));
  nb::capsule owner(
      owned, [](void* p) noexcept { delete static_cast<CubeField*>(p); });
  return nb::ndarray<nb::numpy, double, nb::ndim<3>>(
      owned->data(), {grid.nx, grid.ny, grid.nz}, owner);
}

}  // namespace

void bind_cube_generator(nb::module_& m) {
  nb::class_<CubeGrid>(m, "CubeGrid",
                       R"(
Regular Cartesian grid on which orbitals and densities are evaluated.

All lengths are in Bohr, matching the Gaussian cube-file convention.
)")
      .def(nb::init<>())
      .def(
          "__init__",
          [](CubeGrid* self, const Eigen::Vector3d& origin,
             const Eigen::Vector3d& spacing, std::size_t nx, std::size_t ny,
             std::size_t nz) {
            CubeGrid grid;
            grid.origin = origin;
            grid.spacing = spacing;
            grid.nx = nx;
            grid.ny = ny;
            grid.nz = nz;
            new (self) CubeGrid(std::move(grid));
          },
          nb::arg("origin"), nb::arg("spacing"), nb::arg("nx") = 80,
          nb::arg("ny") = 80, nb::arg("nz") = 80)
      .def_static("from_basis_set", &CubeGrid::from_basis_set,
                  R"(
Build a grid that encloses the molecule with a uniform margin.

Args:
    basis_set: Basis set carrying the molecular structure.
    nx, ny, nz: Number of grid points along each axis.
    margin: Padding in Bohr added around the nuclear bounding box.

Returns:
    CubeGrid: Grid spanning the padded bounding box.
)",
                  nb::arg("basis_set"), nb::arg("nx") = 80, nb::arg("ny") = 80,
                  nb::arg("nz") = 80, nb::arg("margin") = 3.0)
      .def("num_points", &CubeGrid::num_points,
           "Total number of grid points (nx * ny * nz).")
      .def_rw("origin", &CubeGrid::origin,
              "Cartesian position of the first grid point, in Bohr.")
      .def_rw("spacing", &CubeGrid::spacing,
              "Grid step along each axis, in Bohr.")
      .def_rw("nx", &CubeGrid::nx, "Number of points along x.")
      .def_rw("ny", &CubeGrid::ny, "Number of points along y.")
      .def_rw("nz", &CubeGrid::nz, "Number of points along z.")
      .def("__repr__", [](const CubeGrid& grid) {
        return "<CubeGrid shape=(" + std::to_string(grid.nx) + ", " +
               std::to_string(grid.ny) + ", " + std::to_string(grid.nz) + ")>";
      });

  nb::class_<CubeGenerator>(m, "CubeGenerator",
                            R"(
Evaluate molecular orbitals and densities on a grid using the native backend.

Atomic orbital ordering
    Coefficient and density-matrix indices follow the canonical qdk-chemistry
    atomic orbital ordering: atom-major, and within an atom by ascending
    angular momentum and then descending leading exponent. s and p shells are
    always Cartesian, so p functions appear as x, y, z. Shells with angular
    momentum of 2 or more follow gauXC's component convention.

Effective core potentials
    ECP projector shells are not basis functions and take no part in
    evaluation, so a field built from an ECP basis is valence-only: it omits
    the core density that the potential replaces. Cube files written from such
    a basis record this in their comment line.
)")
      .def(nb::init<std::shared_ptr<BasisSet>>(), nb::arg("basis_set"),
           R"(
Args:
    basis_set: Basis set defining the atomic orbitals and the molecule.
)")
      .def(
          "orbital",
          [](const CubeGenerator& self, const Eigen::VectorXd& mo_coeff,
             const CubeGrid& grid, const std::string& outfile,
             const std::string& comment) {
            CubeField field;
            {
              // The backend never calls into Python, so the interpreter is
              // free to run other threads while the grid is evaluated.
              nb::gil_scoped_release release;
              field = self.orbital(mo_coeff, outfile, grid, comment);
            }
            return field_to_array(std::move(field), grid);
          },
          R"(
Evaluate a single molecular orbital on a grid.

Args:
    mo_coeff: One coefficient per atomic orbital, in the canonical ordering
        described on this class. Must have exactly ``nbf`` entries.
    grid: Grid to evaluate on.
    outfile: Path of the cube file to write. If empty, nothing is written and
        the field is only returned.
    comment: First comment line of the cube file.

Returns:
    numpy.ndarray: Field of shape ``(grid.nx, grid.ny, grid.nz)``.

Examples:
    >>> generator = CubeGenerator(basis_set)
    >>> grid = CubeGrid.from_basis_set(basis_set)
    >>> field = generator.orbital(coeff, grid, outfile="homo.cube")
)",
          nb::arg("mo_coeff"), nb::arg("grid"), nb::arg("outfile") = "",
          nb::arg("comment") = "")
      .def(
          "density",
          [](const CubeGenerator& self, const Eigen::MatrixXd& density_matrix,
             const CubeGrid& grid, const std::string& outfile,
             const std::string& comment) {
            CubeField field;
            {
              nb::gil_scoped_release release;
              field = self.density(density_matrix, outfile, grid, comment);
            }
            return field_to_array(std::move(field), grid);
          },
          R"(
Evaluate a density on a grid as ``sum_uv D_uv phi_u(r) phi_v(r)``.

The matrix is used exactly as supplied. It is not scaled, symmetrised, or
spin-summed, so the caller decides which physical quantity results: pass
``Da + Db`` for the total electron density, or a single spin block for that
spin density. A restricted calculation that stores a spatial density matrix
without its factor of two yields a field that is uniformly half the total
density. Both matrices have identical shapes, so this cannot be detected
automatically.

Args:
    density_matrix: ``nbf`` by ``nbf`` matrix indexed by atomic orbital in the
        canonical ordering described on this class.
    grid: Grid to evaluate on.
    outfile: Path of the cube file to write. If empty, nothing is written and
        the field is only returned.
    comment: First comment line of the cube file.

Returns:
    numpy.ndarray: Field of shape ``(grid.nx, grid.ny, grid.nz)``.
)",
          nb::arg("density_matrix"), nb::arg("grid"), nb::arg("outfile") = "",
          nb::arg("comment") = "");

  m.def("generate_orbital_cubes",
        &qdk::chemistry::utils::generate_orbital_cubes,
        R"(
Write one cube file per requested orbital.

Orbital indices are zero-based, matching the numbering used throughout
qdk-chemistry, and the emitted file names embed that same zero-based index as
``<label_prefix>%04d``. Restricted orbitals produce a single spatial cube per
index with no spin suffix. Unrestricted orbitals produce an ``_a`` and a ``_b``
cube per index.

Args:
    orbitals: Orbitals supplying the basis set and coefficients.
    indices: Zero-based orbital indices to write.
    output_dir: Existing directory to write the cube files into.
    grid: Grid to evaluate on.
    label_prefix: Prefix of the generated file names.

Returns:
    list[str]: Paths of the cube files written, in the order written.

Examples:
    >>> paths = generate_orbital_cubes(orbitals, [0, 1], "cubes", grid)
)",
        nb::arg("orbitals"), nb::arg("indices"), nb::arg("output_dir"),
        nb::arg("grid"), nb::arg("label_prefix") = "orbital_",
        nb::call_guard<nb::gil_scoped_release>());
}
