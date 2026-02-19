// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @class ModelHamiltonianContainer
 * @brief Hamiltonian container for lattice model Hamiltonians (Hückel,
 * Hubbard, PPP, etc.) with sparse internal storage.
 *
 * Stores one-body integrals as a sparse matrix and two-body integrals as a
 * sparse map of (p,q,r,s) indices. Provides lazy materialization of dense
 * two-body integrals for the base class interface.
 *
 * Uses ModelOrbitals internally so that the full HamiltonianContainer base
 * class interface works.
 *
 * In addition to the standard interface, this container exposes sparse-specific
 * accessors (sparse_one_body_integrals, sparse_two_body_integrals,
 * one_body_element, two_body_element) that are only available when working
 * directly with the concrete container type.
 */
class ModelHamiltonianContainer : public HamiltonianContainer {
 public:
  /// Sparse index type for two-body integrals (p,q,r,s).
  using TwoBodyIndex = std::tuple<int, int, int, int>;

  /// Sparse two-body integral storage: maps (p,q,r,s) to value.
  using TwoBodyMap = std::map<TwoBodyIndex, double>;

  /**
   * @brief Construct from sparse one-body integrals and sparse two-body map.
   *
   * @param one_body_integrals Sparse one-body integral matrix [n x n]
   * @param two_body_integrals Sparse two-body integral map
   * @param core_energy Scalar energy offset (default 0.0)
   * @param type Hamiltonian type (default Hermitian)
   */
  ModelHamiltonianContainer(Eigen::SparseMatrix<double> one_body_integrals,
                            TwoBodyMap two_body_integrals,
                            double core_energy = 0.0,
                            HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Construct with sparse one-body integrals only (no two-body).
   *
   * @param one_body_integrals Sparse one-body integral matrix [n x n]
   * @param core_energy Scalar energy offset (default 0.0)
   * @param type Hamiltonian type (default Hermitian)
   */
  explicit ModelHamiltonianContainer(
      Eigen::SparseMatrix<double> one_body_integrals, double core_energy = 0.0,
      HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Construct from dense one-body and dense two-body integrals.
   *
   * @param one_body_integrals Dense one-body integral matrix [n x n]
   * @param two_body_integrals Dense two-body integrals [n^4]
   * @param core_energy Scalar energy offset (default 0.0)
   * @param type Hamiltonian type (default Hermitian)
   */
  ModelHamiltonianContainer(const Eigen::MatrixXd& one_body_integrals,
                            const Eigen::VectorXd& two_body_integrals,
                            double core_energy = 0.0,
                            HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Construct from dense one-body integrals only.
   *
   * @param one_body_integrals Dense one-body integral matrix [n x n]
   * @param core_energy Scalar energy offset (default 0.0)
   * @param type Hamiltonian type (default Hermitian)
   */
  explicit ModelHamiltonianContainer(
      const Eigen::MatrixXd& one_body_integrals, double core_energy = 0.0,
      HamiltonianType type = HamiltonianType::Hermitian);

  /**
   * @brief Destructor
   */
  ~ModelHamiltonianContainer() = default;

  /**
   * @brief Create a deep copy of this container
   * @return Unique pointer to a cloned container
   */
  std::unique_ptr<HamiltonianContainer> clone() const override;

  /**
   * @brief Get the type of the underlying container
   * @return "model"
   */
  std::string get_container_type() const override;

  /**
   * @brief Get two-electron integrals as dense vectors for all spin channels.
   *
   * Materializes the sparse two-body map into a dense n^4 vector on first
   * call (cached). Model Hamiltonians are restricted, so all three channels
   * reference the same vector.
   *
   * @return Tuple of references to (aaaa, aabb, bbbb) two-electron integrals
   * @throws std::runtime_error if no two-body integrals are stored
   */
  std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
             const Eigen::VectorXd&>
  get_two_body_integrals() const override final;

  /**
   * @brief Get a specific two-electron integral element
   * @param i First orbital index
   * @param j Second orbital index
   * @param k Third orbital index
   * @param l Fourth orbital index
   * @param channel Spin channel (ignored; model Hamiltonians are restricted)
   * @return Two-electron integral (ij|kl), or 0 if not stored
   */
  double get_two_body_element(
      unsigned i, unsigned j, unsigned k, unsigned l,
      SpinChannel channel = SpinChannel::aaaa) const override final;

  /**
   * @brief Check if two-body integrals are available
   * @return True if the two-body map is non-empty
   */
  bool has_two_body_integrals() const override final;

  /**
   * @brief Model Hamiltonians are always restricted
   * @return true
   */
  bool is_restricted() const override final;

  /**
   * @brief Check if the container data is consistent
   * @return True if the one-body matrix is square and non-empty
   */
  bool is_valid() const override final;

  /**
   * @brief Convert Hamiltonian to JSON
   * @return JSON object containing Hamiltonian data
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Serialize Hamiltonian data to HDF5 group
   * @param group HDF5 group to write data to
   * @throws std::runtime_error (not yet implemented)
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Deserialize Hamiltonian data from HDF5 group
   * @param group HDF5 group to read data from
   * @return Unique pointer to Hamiltonian loaded from group
   * @throws std::runtime_error (not yet implemented)
   */
  static std::unique_ptr<ModelHamiltonianContainer> from_hdf5(H5::Group& group);

  /**
   * @brief Load Hamiltonian from JSON
   * @param j JSON object containing Hamiltonian data
   * @return Unique pointer to Hamiltonian loaded from JSON
   * @throws std::runtime_error (not yet implemented)
   */
  static std::unique_ptr<ModelHamiltonianContainer> from_json(
      const nlohmann::json& j);

  /**
   * @brief Save Hamiltonian to an FCIDUMP file
   * @param filename Path to FCIDUMP file to create/overwrite
   * @param nalpha Number of alpha electrons
   * @param nbeta Number of beta electrons
   * @throws std::runtime_error (not yet implemented)
   */
  void to_fcidump_file(const std::string& filename, size_t nalpha,
                       size_t nbeta) const override;

  /**
   * @brief Direct access to the sparse one-body integral matrix.
   * @return Const reference to the internal sparse matrix
   */
  const Eigen::SparseMatrix<double>& sparse_one_body_integrals() const;

  /**
   * @brief Direct access to the sparse two-body integral map.
   * @return Const reference to the internal two-body map
   */
  const TwoBodyMap& sparse_two_body_integrals() const;

  /**
   * @brief Access a single one-body integral element.
   * @param i Row index
   * @param j Column index
   * @return Value of one-body integral (i,j)
   */
  double one_body_element(int i, int j) const;

 private:
  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /// Sparse storage of one-body integrals
  Eigen::SparseMatrix<double> _one_body_sparse;
  /// Sparse storage of two-body integrals (p,q,r,s) -> value
  TwoBodyMap _two_body_map;

  /// Lazy-materialized dense two-body vector for base class interface
  mutable Eigen::VectorXd _two_body_dense_cache;
  /// Flag indicating whether the dense two-body cache is valid
  mutable bool _two_body_dense_valid = false;

  /**
   * @brief Materialize the dense two-body vector from the sparse map.
   */
  void _materialize_dense_two_body() const;

  /// Create a ModelOrbitals with all n orbitals active.
  static std::shared_ptr<ModelOrbitals> _make_orbitals(int n);

  /// Convert a dense matrix to compressed sparse form.
  static Eigen::SparseMatrix<double> _to_sparse(const Eigen::MatrixXd& m);

  /// Convert a dense two-body vector to sparse map.
  static TwoBodyMap _to_map(const Eigen::VectorXd& v, int n);
};

}  // namespace qdk::chemistry::data
