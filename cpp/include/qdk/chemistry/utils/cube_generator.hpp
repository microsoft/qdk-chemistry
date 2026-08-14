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

  CubeField orbital(const Eigen::VectorXd& mo_coeff, const std::string& outfile,
                    const CubeGrid& grid,
                    const std::string& comment = "") const;

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
