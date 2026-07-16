// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_scalar.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

enum class MPSCanonicalForm {
  Unspecified,
  LeftNormalized,
  RightNormalized,
  Mixed
};

struct MPSMetadata {
  MPSCanonicalForm canonical_form = MPSCanonicalForm::Unspecified;
  std::optional<std::size_t> canonical_center;
  double discarded_weight = 0.0;
  std::vector<std::string> physical_basis;
};

/**
 * @brief One block-sparse MPS site represented by physical-state slices.
 *
 * Slice @c p stores the matrix @f$A^p[l,r]@f$. The number and meaning of the
 * slices are not fixed by this class; @ref MPSMetadata::physical_basis may
 * describe their ordering. Bond-sector order is explicit so dense conversion
 * never depends on unordered block-map iteration.
 */
class MPSSite {
 public:
  using PhysicalSlice = SymmetryBlockedTensorVariant<2>;
  using PhysicalSlicePtr = std::shared_ptr<const PhysicalSlice>;
  using DenseMatrixVariant = std::variant<Eigen::MatrixXd, Eigen::MatrixXcd>;

  MPSSite(std::vector<PhysicalSlicePtr> physical_slices,
          std::vector<SymmetryLabel> left_sector_order,
          std::vector<SymmetryLabel> right_sector_order);

  const std::vector<PhysicalSlicePtr>& physical_slices() const {
    return _physical_slices;
  }
  const std::vector<SymmetryLabel>& left_sector_order() const {
    return _left_sector_order;
  }
  const std::vector<SymmetryLabel>& right_sector_order() const {
    return _right_sector_order;
  }

  std::size_t physical_dimension() const { return _physical_slices.size(); }
  std::size_t left_bond_dimension() const;
  std::size_t right_bond_dimension() const;
  bool is_complex() const;

  /**
   * @brief Materialize this site as a matrix packed as
   * @c (left * physical, right).
   *
   * Row @c (l * physical_dimension() + p) stores @f$A^p[l,r]@f$.
   */
  DenseMatrixVariant to_dense() const;

 private:
  void _validate() const;

  std::vector<PhysicalSlicePtr> _physical_slices;
  std::vector<SymmetryLabel> _left_sector_order;
  std::vector<SymmetryLabel> _right_sector_order;
};

/**
 * @brief Immutable chemistry wavefunction represented by a block-sparse MPS.
 *
 * Each site stores an ordered collection of sparse matrices, one per local
 * physical basis state. The representation is independent of algorithms that
 * consume the MPS, including DMRG and quantum state preparation.
 */
class MPSWavefunction : public WavefunctionContainer {
 public:
  using SitePtr = std::shared_ptr<const MPSSite>;

  MPSWavefunction(std::vector<SitePtr> sites,
                  std::shared_ptr<Orbitals> orbitals,
                  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
                      total_num_particles = nullptr,
                  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
                      active_num_particles = nullptr,
                  MPSMetadata metadata = {});

  const std::vector<SitePtr>& sites() const { return _sites; }
  std::shared_ptr<const Orbitals> orbitals() const { return _orbitals; }

  std::unique_ptr<WavefunctionContainer> clone() const override;
  ScalarVariant overlap(const WavefunctionContainer& other) const override;
  double norm() const override;
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
  total_num_particles() const override {
    return _total_num_particles;
  }
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
  active_num_particles() const override {
    return _active_num_particles;
  }
  std::shared_ptr<const SymmetryBlockedTensor<1>> total_orbital_occupations()
      const override;
  std::shared_ptr<const SymmetryBlockedTensor<1>> active_orbital_occupations()
      const override;
  void clear_caches() const override;
  nlohmann::json to_json() const override;
  std::string get_container_type() const override;
  std::shared_ptr<Orbitals> get_orbitals() const override { return _orbitals; }
  std::vector<std::string> sectors() const override;
  std::shared_ptr<const Orbitals> sector_basis(
      const std::string& name) const override;
  const MPSMetadata& metadata() const { return _metadata; }

  std::size_t num_sites() const { return _sites.size(); }
  std::size_t max_bond_dimension() const;
  bool is_complex() const;

 private:
  void _validate() const;

  std::vector<SitePtr> _sites;
  std::shared_ptr<Orbitals> _orbitals;
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
      _total_num_particles;
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
      _active_num_particles;
  MPSMetadata _metadata;
};

}  // namespace qdk::chemistry::data