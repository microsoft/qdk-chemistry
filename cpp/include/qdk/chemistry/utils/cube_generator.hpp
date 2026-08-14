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

using CubeField = std::vector<double>;

struct CubeGrid {
  Eigen::Vector3d origin{0.0, 0.0, 0.0};
  Eigen::Vector3d spacing{0.2, 0.2, 0.2};
  std::size_t nx = 80, ny = 80, nz = 80;

  static CubeGrid from_basis_set(const data::BasisSet&, std::size_t nx = 80,
                                 std::size_t ny = 80, std::size_t nz = 80,
                                 double margin = 3.0);
  std::size_t num_points() const;
};

// Evaluates molecular orbitals and densities on a `CubeGrid` and optionally
// writes Gaussian cube files.
//
// Atomic orbital ordering: coefficients and density matrices passed to this
// class are indexed in the canonical ordering of the `BasisSet` they were
// built from, which is the ordering `BasisSet::get_shell` exposes. Shells are
// grouped by atom in structure order, and within an atom they follow the order
// established by the basis set, namely ascending angular momentum and then
// descending leading exponent. No reordering is applied here, so a matrix
// produced against a different convention must be permuted by the caller.
//
// Within a shell, s and p shells are always Cartesian, so a p shell always
// contributes three components in x, y, z order regardless of the basis set's
// AOType. Only shells with angular momentum above p honour AOType, matching the
// convention used elsewhere in qdk-chemistry. Component ordering for those
// higher shells follows gauXC's convention, which for spherical shells is the
// same ordering PySCF uses. This whole convention, ordering and normalisation
// together, is pinned by a test that reproduces PySCF's own atomic orbital
// evaluation to machine precision for a basis containing d functions; see
// `test_matches_pyscf_atomic_orbital_evaluation` in
// `python/tests/test_utils_cube_generator.py`.
//
// Effective core potentials: basis sets carrying an ECP are fully supported.
// The ECP is an operator in the Hamiltonian, so it influences the SCF
// coefficients but never enters the evaluation of a basis function at a grid
// point; the valence shells are ordinary contracted Gaussians that are
// evaluated exactly. The consequence is one of interpretation rather than
// accuracy: the resulting field is a *valence-only* quantity. Core density is
// absent near ECP-carrying nuclei, the nuclear cusp there is smoothed, and a
// density cube integrates to the valence electron count rather than the total.
// This matches the behaviour of PySCF's cubegen. When the basis reports ECP
// electrons, the generated cube files record this in their comment line.
class CubeGenerator {
 public:
  explicit CubeGenerator(std::shared_ptr<data::BasisSet> basis_set);
  ~CubeGenerator() noexcept;
  CubeGenerator(CubeGenerator&&) noexcept;
  CubeGenerator& operator=(CubeGenerator&&) noexcept;

  // Evaluates a single molecular orbital on `grid`.
  //
  // `mo_coeff` holds one coefficient per atomic orbital, in the canonical
  // ordering described above, and must have exactly `nbf` entries.
  CubeField orbital(const Eigen::VectorXd& mo_coeff, const std::string& outfile,
                    const CubeGrid& grid,
                    const std::string& comment = "") const;

  // Evaluates a density on `grid` as sum_{uv} D_uv phi_u(r) phi_v(r).
  //
  // `density_matrix` is `nbf` by `nbf`, indexed by atomic orbital in the
  // canonical ordering described above.
  //
  // The matrix is applied exactly as supplied. It is not scaled, symmetrised,
  // or spin-summed, so the caller decides which physical quantity results:
  // pass Da + Db for the total electron density, or a single spin block for
  // that spin density. Note that a restricted calculation storing a spatial
  // density matrix without its factor of two yields a field that is uniformly
  // half the total density. Both matrices have identical dimensions, so no
  // shape check can detect the difference.
  CubeField density(const Eigen::MatrixXd& density_matrix,
                    const std::string& outfile, const CubeGrid& grid,
                    const std::string& comment = "") const;

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

// Writes one cube file per requested orbital into `output_dir`, returning the
// paths written. Orbital `indices` are zero-based, matching the numbering used
// throughout qdk-chemistry, and the emitted file names embed that same
// zero-based index (`<label_prefix>%04zu`). For restricted wavefunctions a
// single spatial cube is written per index (no spin suffix); for unrestricted
// wavefunctions the alpha and beta channels are written as `_a`/`_b` cubes.
std::vector<std::string> generate_orbital_cubes(
    const data::Wavefunction&, const std::vector<std::size_t>& indices,
    const std::string& output_dir, const CubeGrid& grid,
    const std::string& label_prefix = "orbital_");

}  // namespace qdk::chemistry::utils
