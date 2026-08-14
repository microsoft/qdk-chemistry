// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace qdk::chemistry::data {
class BasisSet;
class Wavefunction;
}  // namespace qdk::chemistry::data

namespace qdk::chemistry::utils {

/**
 * @brief Scalar field sampled on a `CubeGrid`
 *
 * Values are stored in the order gauXC emits them, which is row-major in
 * (ix, iy, iz) with `iz` varying fastest. This is the same order the Gaussian
 * cube format expects, so a field can be written out without permutation.
 */
using CubeField = std::vector<double>;

/**
 * @struct CubeGrid
 * @brief Regular axis-aligned grid on which orbitals and densities are sampled
 *
 * The grid is defined by a corner `origin`, a per-axis `spacing`, and a point
 * count along each axis. Point (ix, iy, iz) sits at
 * `origin + (ix * spacing.x, iy * spacing.y, iz * spacing.z)`. The axes are
 * always aligned with the Cartesian axes; rotated or sheared grids are not
 * representable. All lengths are in Bohr.
 */
struct CubeGrid {
  /// Position of the (0, 0, 0) corner point, in Bohr.
  Eigen::Vector3d origin{0.0, 0.0, 0.0};
  /// Step between adjacent points along each axis, in Bohr.
  Eigen::Vector3d spacing{0.2, 0.2, 0.2};
  /// Number of points along x, y and z respectively.
  std::size_t nx = 80, ny = 80, nz = 80;

  /**
   * @brief Builds a grid that encloses the molecule with a uniform margin
   *
   * The origin is the corner of the nuclear bounding box less `margin`, and
   * the step along each axis is the padded extent divided by `n - 1`, so the
   * first and last points lie exactly on the padded box. An axis requesting a
   * single point is given zero spacing. This is the same convention PySCF's
   * `cubegen` uses, so a grid built here and a grid built by PySCF for the
   * same molecule and margin place every point identically.
   *
   * @param basis_set Basis set whose structure supplies the nuclear positions
   * @param nx Number of grid points along x
   * @param ny Number of grid points along y
   * @param nz Number of grid points along z
   * @param margin Padding added on every side of the nuclear bounding box, in
   *               Bohr
   *
   * @return The enclosing grid
   *
   * @throws std::invalid_argument if any dimension is zero or `margin` is
   *         negative
   * @throws std::runtime_error if the basis set carries no structure, or the
   *         structure has no atoms
   */
  static CubeGrid from_basis_set(const data::BasisSet& basis_set,
                                 std::size_t nx = 80, std::size_t ny = 80,
                                 std::size_t nz = 80, double margin = 3.0);

  /**
   * @brief Total number of grid points, `nx * ny * nz`
   *
   * @return The point count
   *
   * @throws std::invalid_argument if any dimension is zero
   * @throws std::overflow_error if the product exceeds `INT_MAX`, which is a
   *         hard limit of the gauXC evaluation backend rather than a choice
   *         made here
   */
  std::size_t num_points() const;
};

/**
 * @class CubeGenerator
 * @brief Evaluates molecular orbitals and densities on a `CubeGrid` and
 *        optionally writes Gaussian cube files
 *
 * Atomic orbital ordering
 * -----------------------
 * Coefficients and density matrices passed to this class are indexed in the
 * canonical ordering of the `BasisSet` they were built from, which is the
 * ordering `BasisSet::get_shell` exposes. Shells are grouped by atom in
 * structure order, and within an atom they follow the order established by the
 * basis set, namely ascending angular momentum and then descending leading
 * exponent. No reordering is applied here, so a matrix produced against a
 * different convention must be permuted by the caller.
 *
 * Within a shell, s and p shells are always Cartesian, so a p shell always
 * contributes three components in x, y, z order regardless of the basis set's
 * AOType. Only shells with angular momentum above p honour AOType, matching
 * the convention used elsewhere in qdk-chemistry. Component ordering for those
 * higher shells follows gauXC's convention, which for spherical shells is the
 * same ordering PySCF uses. This whole convention, ordering and normalisation
 * together, is pinned by a test that reproduces PySCF's own atomic orbital
 * evaluation to machine precision for a basis containing d functions; see
 * `test_matches_pyscf_atomic_orbital_evaluation` in
 * `python/tests/test_utils_cube_generator.py`.
 *
 * Effective core potentials
 * -------------------------
 * Basis sets carrying an ECP are fully supported. The ECP is an operator in
 * the Hamiltonian, so it influences the SCF coefficients but never enters the
 * evaluation of a basis function at a grid point; the valence shells are
 * ordinary contracted Gaussians that are evaluated exactly. The consequence is
 * one of interpretation rather than accuracy: the resulting field is a
 * *valence-only* quantity. Core density is absent near ECP-carrying nuclei,
 * the nuclear cusp there is smoothed, and a density cube integrates to the
 * valence electron count rather than the total. This matches the behaviour of
 * PySCF's cubegen. When the basis reports ECP electrons, the generated cube
 * files record this in their comment line.
 */
class CubeGenerator {
 public:
  /**
   * @brief Constructs a generator bound to a basis set
   *
   * The basis set is translated to the gauXC representation once here and
   * reused by every subsequent evaluation, so constructing one generator and
   * calling it repeatedly is cheaper than constructing one per orbital.
   *
   * @param basis_set Basis set defining the atomic orbitals and the molecular
   *                  geometry
   *
   * @throws std::invalid_argument if `basis_set` is null, carries no
   *         structure, or contains a shell gauXC cannot represent
   */
  explicit CubeGenerator(std::shared_ptr<data::BasisSet> basis_set);
  ~CubeGenerator() noexcept;
  CubeGenerator(CubeGenerator&&) noexcept;
  CubeGenerator& operator=(CubeGenerator&&) noexcept;

  /**
   * @brief Evaluates a single molecular orbital on `grid`
   *
   * @param mo_coeff One coefficient per atomic orbital, in the canonical
   *                 ordering described in the class documentation. Must have
   *                 exactly `nbf` entries.
   * @param outfile Path of the cube file to write. Pass an empty string to
   *                evaluate without writing anything to disk.
   * @param grid Grid on which to sample the orbital
   * @param comment Free-form text placed on the cube file comment line. When
   *                the basis carries an ECP, a valence-only note is appended
   *                to it.
   *
   * @return The orbital sampled at every grid point
   *
   * @throws std::invalid_argument if `mo_coeff` does not have `nbf` entries
   * @throws std::overflow_error if the grid exceeds the backend point limit
   */
  CubeField orbital(const Eigen::VectorXd& mo_coeff, const std::string& outfile,
                    const CubeGrid& grid,
                    const std::string& comment = "") const;

  /**
   * @brief Evaluates a density on `grid` as sum_{uv} D_uv phi_u(r) phi_v(r)
   *
   * The matrix is applied exactly as supplied. It is not scaled, symmetrised,
   * or spin-summed, so the caller decides which physical quantity results:
   * pass Da + Db for the total electron density, or a single spin block for
   * that spin density. Note that a restricted calculation storing a spatial
   * density matrix without its factor of two yields a field that is uniformly
   * half the total density. Both matrices have identical dimensions, so no
   * shape check can detect the difference.
   *
   * @param density_matrix An `nbf` by `nbf` matrix indexed by atomic orbital
   *                       in the canonical ordering described in the class
   *                       documentation
   * @param outfile Path of the cube file to write. Pass an empty string to
   *                evaluate without writing anything to disk.
   * @param grid Grid on which to sample the density
   * @param comment Free-form text placed on the cube file comment line. When
   *                the basis carries an ECP, a valence-only note is appended
   *                to it.
   *
   * @return The density sampled at every grid point
   *
   * @throws std::invalid_argument if `density_matrix` is not `nbf` by `nbf`
   * @throws std::overflow_error if the grid exceeds the backend point limit
   */
  CubeField density(const Eigen::MatrixXd& density_matrix,
                    const std::string& outfile, const CubeGrid& grid,
                    const std::string& comment = "") const;

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

/**
 * @brief Writes one cube file per requested orbital into `output_dir`
 *
 * Orbital `indices` are zero-based, matching the numbering used throughout
 * qdk-chemistry, and the emitted file names embed that same zero-based index
 * (`<label_prefix>%04zu`). For restricted wavefunctions a single spatial cube
 * is written per index, with no spin suffix; for unrestricted wavefunctions
 * the alpha and beta channels are written as `_a` and `_b` cubes.
 *
 * @param wavefunction Wavefunction supplying the orbitals and their basis set
 * @param indices Zero-based orbital indices to write
 * @param output_dir Directory to write into. Created if it does not exist.
 * @param grid Grid on which to sample each orbital
 * @param label_prefix Stem placed before the zero-padded index in each file
 *                     name
 *
 * @return The paths written, in the order they were written
 *
 * @throws std::out_of_range if any index is beyond the available orbitals
 */
std::vector<std::string> generate_orbital_cubes(
    const data::Wavefunction& wavefunction,
    const std::vector<std::size_t>& indices, const std::string& output_dir,
    const CubeGrid& grid, const std::string& label_prefix = "orbital_");

}  // namespace qdk::chemistry::utils
