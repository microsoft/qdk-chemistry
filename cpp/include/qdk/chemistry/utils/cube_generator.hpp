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
