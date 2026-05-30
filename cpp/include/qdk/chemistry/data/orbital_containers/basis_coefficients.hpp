// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <array>
#include <memory>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <string>
#include <unordered_map>

namespace qdk::chemistry::data {

/**
 * @class BasisCoefficients
 * @brief Symmetry-blocked single-particle basis coefficients (MO coefficients).
 *
 * Thin, immutable semantic wrapper around a rank-2 @ref SymmetryBlockedTensor
 * whose two axes are @c [ao_symmetries, mo_symmetries]. Block @c (ao_label,
 * mo_label) is the matrix expressing the modes of @c mo_label in terms of the
 * atomic orbitals of @c ao_label.
 *
 * Aliasing follows the wrapped tensor: for RHF/ROHF the producer supplies the
 * @c (alpha, alpha) and @c (beta, beta) blocks as the same @c shared_ptr (so
 * @ref is_restricted is true); for UHF the blocks are distinct.
 */
class BasisCoefficients : public DataClass {
 public:
  using Sbt = SymmetryBlockedTensor<2, double>;
  using Labels = Sbt::Labels;

  /**
   * @brief Construct from a rank-2 symmetry-blocked tensor of coefficients.
   * @param coefficients Non-null rank-2 tensor with axes [ao, mo]
   * @throws std::invalid_argument if @p coefficients is null
   */
  explicit BasisCoefficients(std::shared_ptr<const Sbt> coefficients);

  /** @brief The underlying rank-2 symmetry-blocked tensor. */
  const std::shared_ptr<const Sbt>& tensor() const { return _coefficients; }

  /** @brief Symmetry vocabulary of the atomic-orbital (row) axis. */
  std::shared_ptr<const Symmetries> ao_symmetries() const {
    return _coefficients->symmetries()[0];
  }

  /** @brief Symmetry vocabulary of the molecular-orbital (column) axis. */
  std::shared_ptr<const Symmetries> mo_symmetries() const {
    return _coefficients->symmetries()[1];
  }

  /** @brief Per-label AO extents. */
  std::unordered_map<SymmetryLabel, std::size_t> ao_extents() const {
    return _coefficients->extents()[0];
  }

  /** @brief Per-label MO extents. */
  std::unordered_map<SymmetryLabel, std::size_t> mo_extents() const {
    return _coefficients->extents()[1];
  }

  /** @brief True iff the spin blocks alias (restricted coefficients). */
  bool is_restricted() const { return _coefficients->is_restricted(); }

  /** @brief True iff a coefficient block is stored for @p ao_label/@p mo_label.
   */
  bool has_block(const SymmetryLabel& ao_label,
                 const SymmetryLabel& mo_label) const {
    return _coefficients->has_block(Labels{ao_label, mo_label});
  }

  /**
   * @brief Coefficient matrix for the @p ao_label rows and @p mo_label columns.
   * @throws BlockLabelInvalidError if no such block exists
   */
  const Tensor<2, double>& block(const SymmetryLabel& ao_label,
                                 const SymmetryLabel& mo_label) const {
    return _coefficients->block(Labels{ao_label, mo_label});
  }

  // ---- DataClass interface ------------------------------------------------

  std::string get_data_type_name() const override {
    return "basis_coefficients";
  }
  std::string get_summary() const override;
  void to_file(const std::string& filename,
               const std::string& type) const override;
  nlohmann::json to_json() const override;
  void to_json_file(const std::string& filename) const override;
  void to_hdf5(H5::Group& group) const override;
  void to_hdf5_file(const std::string& filename) const override;

  static std::shared_ptr<BasisCoefficients> from_json(const nlohmann::json& j);
  static std::shared_ptr<BasisCoefficients> from_json_file(
      const std::string& filename);
  static std::shared_ptr<BasisCoefficients> from_hdf5(H5::Group& group);
  static std::shared_ptr<BasisCoefficients> from_hdf5_file(
      const std::string& filename);
  static std::shared_ptr<BasisCoefficients> from_file(
      const std::string& filename, const std::string& type);

 private:
  std::shared_ptr<const Sbt> _coefficients;
};

}  // namespace qdk::chemistry::data
