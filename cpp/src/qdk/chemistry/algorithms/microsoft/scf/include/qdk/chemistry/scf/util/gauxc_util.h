// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/config.h>

#include <functional>
#include <gauxc/enums.hpp>
#include <gauxc/grid_factory.hpp>
#include <string>

namespace qdk::chemistry::scf {
namespace gauxc_util {

/**
 * @brief Convert string to GauXC atomic grid size enumeration
 *
 * Parses grid size specification strings (e.g., "UltraFineGrid", "FineGrid")
 * and returns the corresponding GauXC enumeration value.
 *
 * @param str Grid size specification string
 * @return GauXC::AtomicGridSizeDefault Enumeration value for the grid size
 * @throws std::runtime_error if string is not recognized
 */
GauXC::AtomicGridSizeDefault grid_size_from_string(const std::string& str);

/**
 * @brief Convert GauXC atomic grid size enumeration to string
 *
 * Converts a GauXC grid size enumeration to its string representation
 * for output and configuration purposes.
 *
 * @param type Grid size enumeration value
 * @return std::string String representation of the grid size
 */
std::string to_string(GauXC::AtomicGridSizeDefault type);

/**
 * @brief Convert string to GauXC radial quadrature enumeration
 *
 * Parses radial quadrature scheme strings (e.g., "MuraKnowles", "Treutler")
 * and returns the corresponding GauXC enumeration value.
 *
 * @param str Radial quadrature scheme string
 * @return GauXC::RadialQuad Enumeration value for the radial quadrature
 * @throws std::runtime_error if string is not recognized
 */
GauXC::RadialQuad radial_quad_from_string(const std::string& str);

/**
 * @brief Convert GauXC radial quadrature enumeration to string
 *
 * Converts a GauXC radial quadrature enumeration to its string representation.
 *
 * @param type Radial quadrature enumeration value
 * @return std::string String representation of the radial quadrature scheme
 */
std::string to_string(GauXC::RadialQuad type);

/**
 * @brief Convert string to GauXC pruning scheme enumeration
 *
 * Parses grid pruning scheme strings (e.g., "Unpruned", "Robust", "Treutler")
 * and returns the corresponding GauXC enumeration value. Pruning reduces the
 * number of grid points in regions where basis functions have small overlap.
 *
 * @param str Pruning scheme string
 * @return GauXC::PruningScheme Enumeration value for the pruning scheme
 * @throws std::runtime_error if string is not recognized
 */
GauXC::PruningScheme prune_method_from_string(const std::string& str);

/**
 * @brief Convert GauXC pruning scheme enumeration to string
 *
 * Converts a GauXC pruning scheme enumeration to its string representation.
 *
 * @param type Pruning scheme enumeration value
 * @return std::string String representation of the pruning scheme
 */
std::string to_string(GauXC::PruningScheme type);

/**
 * @brief Convert string to GauXC execution space enumeration
 *
 * Parses execution space strings (e.g., "Host", "Device") and returns the
 * corresponding GauXC enumeration value. This determines whether operations
 * run on CPU (Host) or GPU (Device).
 *
 * @param str Execution space string
 * @return GauXC::ExecutionSpace Enumeration value for the execution space
 * @throws std::runtime_error if string is not recognized
 */
GauXC::ExecutionSpace execution_space_from_string(const std::string& str);

/**
 * @brief Convert GauXC execution space enumeration to string
 *
 * Converts a GauXC execution space enumeration to its string representation.
 *
 * @param type Execution space enumeration value
 * @return std::string String representation of the execution space
 */
std::string to_string(GauXC::ExecutionSpace type);
}  // namespace gauxc_util

/**
 * @brief Configuration parameters for GauXC DFT integration
 *
 * Contains all settings for configuring GauXC's GPU-accelerated DFT grid
 * integration, including grid specifications, quadrature schemes, kernel
 * selections, and execution spaces (CPU vs GPU).
 *
 * GPU acceleration (Device execution space) is automatically enabled when
 * both GAUXC_HAS_DEVICE and QDK_CHEMISTRY_ENABLE_GPU are defined at compile
 * time.
 */
struct GAUXCInput {
  GauXC::AtomicGridSizeDefault grid_spec =
      GauXC::AtomicGridSizeDefault::UltraFineGrid;  ///< Atomic grid density

  GauXC::RadialQuad rad_quad_spec =
      GauXC::RadialQuad::MuraKnowles;  ///< Radial quadrature scheme

  GauXC::PruningScheme prune_spec =
      GauXC::PruningScheme::Unpruned;  ///< Grid pruning scheme

  double basis_tol = 1e-10;  ///< Threshold for screening negligible basis
                             ///< function contributions

  double batch_size = 10000;  ///< Number of grid points processed per batch

  std::string integrator_kernel =
      "Default";  ///< Kernel selection for XC integration (See GauXC
                  ///< Documentation)

  std::string reduction_kernel =
      "Default";  ///< Kernel for reduction operations in parallel integration
                  ///< (See GauXC Documentation)

#if defined(GAUXC_HAS_DEVICE) && defined(QDK_CHEMISTRY_ENABLE_GPU)
  std::string lwd_kernel = "Scheme1-CUTLASS";  ///< Local work driver kernel
                                               ///< (See GauXC Documentation)

  GauXC::ExecutionSpace integrator_ex =
      GauXC::ExecutionSpace::Device;  ///< Run integrator on GPU

  GauXC::ExecutionSpace loadbalancer_ex =
      GauXC::ExecutionSpace::Device;  ///< Run load balancer on GPU

  GauXC::ExecutionSpace weights_ex =
      GauXC::ExecutionSpace::Device;  ///< Compute grid weights on GPU
#else
  std::string lwd_kernel =
      "Default";  ///< Local work driver kernel (See GauXC Documentation)

  GauXC::ExecutionSpace integrator_ex =
      GauXC::ExecutionSpace::Host;  ///< Run integrator on CPU

  GauXC::ExecutionSpace loadbalancer_ex =
      GauXC::ExecutionSpace::Host;  ///< Run load balancer on CPU

  GauXC::ExecutionSpace weights_ex =
      GauXC::ExecutionSpace::Host;  ///< Compute grid weights on CPU
#endif
};
}  // namespace qdk::chemistry::scf

/**
 * @brief Hash specialization for GAUXCInput to enable use in hash maps
 *
 * Provides std::hash specialization for GAUXCInput, allowing it to be used
 * as a key in std::unordered_map, std::unordered_set, and other hash-based
 * containers. This is useful for caching GauXC integrators with specific
 * configurations.
 *
 * The hash function combines all GAUXCInput fields using the golden ratio
 * hash constant (0x9e3779b9) to minimize hash collisions.
 */
namespace std {
template <>
struct hash<qdk::chemistry::scf::GAUXCInput> {
  /**
   * @brief Compute hash value for GAUXCInput
   *
   * Combines hashes of all GAUXCInput members using the golden ratio method
   * from Boost. Handles both enum and non-enum types appropriately.
   *
   * @param input GAUXCInput instance to hash
   * @return size_t Hash value combining all configuration parameters
   */
  size_t operator()(
      const qdk::chemistry::scf::GAUXCInput& input) const noexcept {
    // 0x9e3779b9 is the golden ratio hash constant from Boost
    constexpr size_t HASH_CONSTANT = 0x9e3779b9;

    // Lambda to handle hash combining with a unified approach
    auto hash_combine = [&](size_t& seed, const auto& value) {
      using ValueType = std::decay_t<decltype(value)>;
      if constexpr (std::is_enum_v<ValueType>) {
        seed ^= std::hash<int>{}(static_cast<int>(value)) + HASH_CONSTANT +
                (seed << 6) + (seed >> 2);
      } else {
        seed ^= std::hash<ValueType>{}(value) + HASH_CONSTANT + (seed << 6) +
                (seed >> 2);
      }
    };

    size_t seed = 0;

    // Apply hash_combine to each field of GAUXCInput
    hash_combine(seed, input.grid_spec);
    hash_combine(seed, input.rad_quad_spec);
    hash_combine(seed, input.prune_spec);
    hash_combine(seed, input.basis_tol);
    hash_combine(seed, input.batch_size);
    hash_combine(seed, input.integrator_kernel);
    hash_combine(seed, input.lwd_kernel);
    hash_combine(seed, input.reduction_kernel);
    hash_combine(seed, input.integrator_ex);
    hash_combine(seed, input.loadbalancer_ex);
    hash_combine(seed, input.weights_ex);

    return seed;
  }
};
}  // namespace std
