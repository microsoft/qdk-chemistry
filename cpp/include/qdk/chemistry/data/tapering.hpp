// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <H5Cpp.h>

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

class TaperingSpecification : public DataClass {
 public:
  TaperingSpecification(std::vector<std::size_t> qubit_indices,
                        std::vector<int> eigenvalues);

  const std::vector<std::size_t>& qubit_indices() const {
    return qubit_indices_;
  }

  const std::vector<int>& eigenvalues() const { return eigenvalues_; }

  std::size_t num_tapered() const { return qubit_indices_.size(); }

  static TaperingSpecification symmetry_conserving_bravyi_kitaev(
      std::size_t num_modes, std::size_t n_alpha, std::size_t n_beta);

  static TaperingSpecification parity_two_qubit_reduction(std::size_t num_modes,
                                                          std::size_t n_alpha,
                                                          std::size_t n_beta);

  std::string get_data_type_name() const override {
    return "tapering_specification";
  }

  std::string get_summary() const override;

  void to_file(const std::string& filename,
               const std::string& type) const override;

  nlohmann::json to_json() const override;

  static TaperingSpecification from_json(const nlohmann::json& data);

  void to_json_file(const std::string& filename) const override;

  static TaperingSpecification from_json_file(const std::string& filename);

  void to_hdf5(H5::Group& group) const override;

  static TaperingSpecification from_hdf5(H5::Group& group);

  void to_hdf5_file(const std::string& filename) const override;

  static TaperingSpecification from_hdf5_file(const std::string& filename);

  static TaperingSpecification from_file(const std::string& filename,
                                         const std::string& type);

  bool operator==(const TaperingSpecification& other) const;

 private:
  std::vector<std::size_t> qubit_indices_;
  std::vector<int> eigenvalues_;

  /// Serialization schema version.
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

}  // namespace qdk::chemistry::data
