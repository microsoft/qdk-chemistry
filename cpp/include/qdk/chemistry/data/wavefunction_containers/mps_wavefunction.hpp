// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief One explicit, optionally symmetry-blocked MPS site.
 *
 * The tensor slots are (left bond, physical state, right bond). Each rank-3
 * block is a matrix with rows (local_left * physical_extent + local_physical)
 * and columns local_right. Missing blocks represent zero.
 *
 * Physical states use Configuration's binary or spin-half encoding. Symmetry
 * labels describe explicit basis states.
 */
class MPSSite {
 public:
  /** @brief Real or complex rank-3 symmetry-blocked storage. */
  using TensorVariant = SymmetryBlockedTensorVariant<3>;
  /** @brief Shared immutable site tensor. */
  using TensorPtr = std::shared_ptr<const TensorVariant>;
  /**
   * @brief Sector packing orders for the left, physical, and right slots.
   *
   * Each order lists that slot's sector labels exactly once. A sector occupies
   * a contiguous range of basis indices, starting after the extents of all
   * preceding sectors in the order.
   */
  using SectorOrders = std::array<std::vector<SymmetryLabel>, 3>;
  /** @brief Validated flattened dimensions and sector offsets of a site. */
  struct TensorLayout {
    std::array<Eigen::Index, 3> dimensions{};
    std::array<std::unordered_map<SymmetryLabel, Eigen::Index>, 3> offsets;
  };
  /** @brief Real or complex matrix used for packed dense export. */
  using DenseMatrixVariant = ContainerTypes::MatrixVariant;

  /**
   * @brief Construct a site from blocks and index spaces.
   * @param tensor Rank-3 blocks in the packing convention above.
   * @param sector_orders Packing order for each of the three tensor slots;
   * each list must contain all of that slot's sector labels exactly once.
   * @param physical_basis One-mode configurations in flattened physical order.
   * Defaults to (0,1) for dimension two or (0,u,d,2) for dimension four.
   * Other dimensions require an explicit basis.
   * @throws std::invalid_argument for incompatible shapes, orders or basis
   * states, nonfinite entries, or equivalent-sector axes.
   */
  MPSSite(TensorPtr tensor, SectorOrders sector_orders,
          std::vector<Configuration> physical_basis = {});

  /** @brief Immutable symmetry-blocked site tensor. */
  const TensorVariant& tensor() const { return *_tensor; }
  /** @brief Packing orders for (left, physical, right). */
  const SectorOrders& sector_orders() const { return _sector_orders; }
  /** @brief Immutable packing layout computed at construction. */
  const TensorLayout& tensor_layout() const { return _tensor_layout; }
  /** @brief Packing order of left-bond sectors. */
  const std::vector<SymmetryLabel>& left_sector_order() const {
    return _sector_orders[0];
  }
  /** @brief Packing order of physical sectors. */
  const std::vector<SymmetryLabel>& physical_sector_order() const {
    return _sector_orders[1];
  }
  /** @brief Packing order of right-bond sectors. */
  const std::vector<SymmetryLabel>& right_sector_order() const {
    return _sector_orders[2];
  }
  /** @brief Local configurations in flattened physical-index order. */
  const std::vector<Configuration>& physical_basis() const {
    return _physical_basis;
  }
  /** @brief Total dimension of the physical space. */
  std::size_t physical_dimension() const;
  /** @brief Total dimension of the left bond. */
  std::size_t left_bond_dimension() const;
  /** @brief Total dimension of the right bond. */
  std::size_t right_bond_dimension() const;
  /** @brief Whether the tensor uses complex coefficients. */
  bool is_complex() const;

  /**
   * @brief Materialize the site for dense interoperability.
   * @return Matrix packed as (left * physical, right), with absent blocks zero.
   */
  DenseMatrixVariant to_dense() const;

 private:
  /** @brief Check tensor packing, sector orders, local basis, and finiteness.
   */
  void _validate();

  TensorPtr _tensor;
  SectorOrders _sector_orders;
  TensorLayout _tensor_layout;
  std::vector<Configuration> _physical_basis;
};

/**
 * @brief Container to store MPS over an orbital or model-mode basis.
 *
 * Each site describes its own local basis. Particle counts, the orthogonality
 * center, and supplied RDMs are optional. The site-to-orbital order defaults to
 * identity. RDM indices use active-orbital order, not MPS chain order.
 */
class MPSContainer : public WavefunctionContainer {
 public:
  /** @brief Shared immutable site in the chain. */
  using SitePtr = std::shared_ptr<const MPSSite>;
  /** @brief Symmetry-blocked occupation-count metadata. */
  using ParticleCount = SymmetryBlockedScalar<std::size_t>;

  /**
   * @brief Construct an MPS with optional precomputed RDMs.
   * @param sites Sites in chain order.
   * @param orbitals Orbital or model-mode basis for the active sites.
   * @param total_num_particles Optional total number of particles.
   * @param active_num_particles Optional active number of particles.
   * @param orthogonality_center Optional, unverified orthogonality center.
   * @param site_to_orbital_order Permutation of active-mode slots, not full
   * orbital indices. An empty vector selects identity.
   * @param one_rdm_spin_traced Optional spin-traced active-space 1-RDM.
   * @param one_rdm_aa Optional alpha-alpha 1-RDM block.
   * @param one_rdm_bb Optional beta-beta 1-RDM block.
   * @param two_rdm_spin_traced Optional flattened spin-traced active-space
   * 2-RDM.
   * @param two_rdm_aaaa Optional flattened alpha-alpha-alpha-alpha 2-RDM block.
   * @param two_rdm_aabb Optional flattened alpha-alpha-beta-beta 2-RDM block.
   * @param two_rdm_bbbb Optional flattened beta-beta-beta-beta 2-RDM block.
   * RDMs use the same conventions and real/complex storage as
   * StateVectorContainer. They are supplied data, not computed from the sites.
   * @throws std::invalid_argument for null sites/basis, incompatible bonds or
   * scalar types, or invalid site counts, permutations or center indices.
   */
  MPSContainer(
      std::vector<SitePtr> sites, std::shared_ptr<Orbitals> orbitals,
      std::shared_ptr<const ParticleCount> total_num_particles = nullptr,
      std::shared_ptr<const ParticleCount> active_num_particles = nullptr,
      std::optional<std::size_t> orthogonality_center = std::nullopt,
      std::vector<std::size_t> site_to_orbital_order = {},
      const std::optional<MatrixVariant>& one_rdm_spin_traced = std::nullopt,
      const std::optional<MatrixVariant>& one_rdm_aa = std::nullopt,
      const std::optional<MatrixVariant>& one_rdm_bb = std::nullopt,
      const std::optional<VectorVariant>& two_rdm_spin_traced = std::nullopt,
      const std::optional<VectorVariant>& two_rdm_aaaa = std::nullopt,
      const std::optional<VectorVariant>& two_rdm_aabb = std::nullopt,
      const std::optional<VectorVariant>& two_rdm_bbbb = std::nullopt);

  /** @brief Immutable sites in chain order. */
  const std::vector<SitePtr>& sites() const { return _sites; }
  /**
   * @brief Validate an open-boundary chain independently of orbital metadata.
   * @param sites Nonempty chain of nonnull sites with one scalar type.
   * @throws std::invalid_argument for incompatible adjacent bond spaces or
   * outer bond dimensions other than one.
   */
  static void validate_sites(const std::vector<SitePtr>& sites);
  /** @brief Number of sites. */
  std::size_t num_sites() const { return _sites.size(); }
  /** @brief Largest total bond dimension in the stored chain. */
  std::size_t max_bond_dimension() const;
  /** @brief Whether the site tensors use complex coefficients. */
  bool is_complex() const override;
  /**
   * @brief Copy the container, sharing its immutable sites and stored RDMs.
   * @return Independent container with the same wavefunction and metadata.
   */
  std::unique_ptr<WavefunctionContainer> clone() const override;
  /** @brief Return the serialization identifier "mps". */
  std::string get_container_type() const override;
  /** @brief Return the associated orbital or model-mode basis. */
  std::shared_ptr<Orbitals> get_orbitals() const override { return _orbitals; }

  /** @brief Whether total particle-count metadata was supplied. */
  bool has_total_num_particles() const {
    return _total_num_particles != nullptr;
  }
  /** @brief Whether active particle-count metadata was supplied. */
  bool has_active_num_particles() const {
    return _active_num_particles != nullptr;
  }
  /**
   * @brief Get the supplied total particle count.
   * @return Symmetry-blocked total occupation-count metadata.
   * @throws std::runtime_error if no count was supplied.
   */
  std::shared_ptr<const ParticleCount> total_num_particles() const override;
  /**
   * @brief Get the supplied active-space particle count.
   * @return Symmetry-blocked active occupation-count metadata.
   * @throws std::runtime_error if no count was supplied.
   */
  std::shared_ptr<const ParticleCount> active_num_particles() const override;

  /** @brief Producer-supplied canonical center, or unspecified. */
  std::optional<std::size_t> orthogonality_center() const {
    return _orthogonality_center;
  }
  /** @brief Active-mode slots in chain order. */
  const std::vector<std::size_t>& site_to_orbital_order() const {
    return _site_to_orbital_order;
  }
  /** @brief Return the single default sector spanned by this container. */
  std::vector<std::string> sectors() const override;
  /**
   * @brief Resolve the basis for a single-particle sector.
   * @param name Sector name; only Wavefunction::DEFAULT_SECTOR is supported.
   * @return Associated orbital or model-mode basis.
   * @throws std::out_of_range if the requested sector is unknown.
   */
  std::shared_ptr<const Orbitals> sector_basis(
      const std::string& name) const override;

  /**
   * @brief Unsupported MPS contraction.
   * @param other Wavefunction that would be used in the overlap.
   * @throws std::runtime_error Always.
   */
  ScalarVariant overlap(const WavefunctionContainer& other) const override;
  /** @brief Unsupported MPS norm; throws std::runtime_error. */
  double norm() const override;
  /** @brief Unsupported total orbital occupations; throws std::runtime_error.
   */
  std::shared_ptr<const SymmetryBlockedTensor<1>> total_orbital_occupations()
      const override;
  /** @brief Unsupported active orbital occupations; throws std::runtime_error.
   */
  std::shared_ptr<const SymmetryBlockedTensor<1>> active_orbital_occupations()
      const override;
  /** @brief No-op: retain supplied RDMs, which cannot be regenerated from
   * sites. */
  void clear_caches() const override;

  /** @brief Serialize sites, metadata, orbitals, and stored RDMs to JSON. */
  nlohmann::json to_json() const override;
  /**
   * @brief Reconstruct an MPS from its versioned JSON payload.
   * @param json Payload emitted by to_json().
   * @return Reconstructed container with unchanged tensor and RDM data.
   * @throws std::exception if metadata is incompatible or storage is malformed.
   */
  static std::unique_ptr<MPSContainer> from_json(const nlohmann::json& json);
  /**
   * @brief Serialize metadata and binary site/RDM payloads to HDF5.
   * @param group Empty destination group.
   * @throws std::runtime_error if HDF5 serialization fails.
   */
  void to_hdf5(H5::Group& group) const override;
  /**
   * @brief Reconstruct an MPS from HDF5.
   * @param group Group written by to_hdf5().
   * @return Reconstructed container.
   * @throws std::exception if the data is incompatible, malformed, or
   * unreadable.
   */
  static std::unique_ptr<MPSContainer> from_hdf5(H5::Group& group);

 protected:
  /**
   * @brief Hash defining tensor data and metadata, excluding cached RDMs.
   * @param ctx Content-hash context to update.
   */
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;
  /**
   * @brief Select general spin handling for supplied MPS RDMs.
   * @return Always false: equal particle counts do not establish closed-shell
   * symmetry of the stored state.
   */
  bool _is_restricted_closed_shell() const override;

 private:
  /** @brief Validate site/basis counts, adjacent bonds, boundaries, and
   * ordering. */
  void _validate() const;
  /** @brief Serialize chain metadata without binary tensors, orbitals, or RDMs.
   */
  nlohmann::json _metadata_to_json() const;

  std::vector<SitePtr> _sites;
  std::shared_ptr<Orbitals> _orbitals;
  std::shared_ptr<const ParticleCount> _total_num_particles;
  std::shared_ptr<const ParticleCount> _active_num_particles;
  std::optional<std::size_t> _orthogonality_center;
  std::vector<std::size_t> _site_to_orbital_order;
  /** @brief Version of the MPS payload, independent of the wavefunction
   * envelope. */
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

}  // namespace qdk::chemistry::data
