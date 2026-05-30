// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <stdexcept>
#include <string>

namespace qdk::chemistry::data {

/**
 * @brief Root of the QDK/Chemistry typed-error hierarchy.
 *
 * All structured errors raised by the symmetry vocabulary, the
 * symmetry-blocked storage primitives, and the single-particle basis
 * containers derive from this type. It derives from @c std::runtime_error so
 * existing @c catch(const std::exception&) sites continue to function.
 */
class QdkError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------
// BasisSet errors
// ---------------------------------------------------------------------------

/**
 * @brief Base class for errors raised by @ref BasisSet.
 */
class BasisSetError : public QdkError {
 public:
  using QdkError::QdkError;
};

/**
 * @brief Raised when supplied AO symmetry extents violate orbit equivalence.
 *
 * For an equivalent (restricted) spin axis, every orbit-equivalent spin label
 * must share the same extent. This error reports the mismatch.
 */
class BasisSetSpinExtentMismatchError : public BasisSetError {
 public:
  using BasisSetError::BasisSetError;
};

// ---------------------------------------------------------------------------
// SingleParticleBasis errors
// ---------------------------------------------------------------------------

/**
 * @brief Base class for errors raised by @ref SingleParticleBasis and its
 * subclasses (@ref Orbitals, @ref ModelOrbitals).
 */
class SingleParticleBasisError : public QdkError {
 public:
  using QdkError::QdkError;
};

/**
 * @brief Raised when a @ref ModelOrbitals method that requires a concrete
 * basis set is invoked.
 */
class ModelOrbitalsNoBasisSetError : public SingleParticleBasisError {
 public:
  using SingleParticleBasisError::SingleParticleBasisError;
};

/**
 * @brief Deprecated alias for @ref ModelOrbitalsNoBasisSetError.
 *
 * @deprecated Use @ref ModelOrbitalsNoBasisSetError instead.
 */
using OrbitalsCannotInBasisError
    [[deprecated("Use ModelOrbitalsNoBasisSetError instead.")]] =
        ModelOrbitalsNoBasisSetError;

/**
 * @brief Raised when a projection onto a model-orbital space fails.
 */
class ModelOrbitalsProjectionError : public SingleParticleBasisError {
 public:
  using SingleParticleBasisError::SingleParticleBasisError;
};

/**
 * @brief Raised when orbital-space partitions are not mutually disjoint.
 */
class OrbitalSpacePartitioningDisjointnessError
    : public SingleParticleBasisError {
 public:
  using SingleParticleBasisError::SingleParticleBasisError;
};

/**
 * @brief Raised when an index in a @ref SymmetryBlockedIndexSet exceeds the
 * declared extent for its label.
 */
class IndexSetOutOfRangeError : public SingleParticleBasisError {
 public:
  using SingleParticleBasisError::SingleParticleBasisError;
};

/**
 * @brief Raised when index lists in a @ref SymmetryBlockedIndexSet are not
 * sorted and unique.
 */
class IndexSetNotSortedUniqueError : public SingleParticleBasisError {
 public:
  using SingleParticleBasisError::SingleParticleBasisError;
};

// ---------------------------------------------------------------------------
// Symmetry vocabulary errors
// ---------------------------------------------------------------------------

/**
 * @brief Base class for errors raised by the single-particle symmetry
 * vocabulary.
 */
class SymmetryError : public QdkError {
 public:
  using QdkError::QdkError;
};

/**
 * @brief Raised when two symmetry vocabularies cannot be combined or compared.
 */
class SymmetryIncompatibleError : public SymmetryError {
 public:
  using SymmetryError::SymmetryError;
};

/**
 * @brief Raised when an operation requires a symmetry configuration that the
 * supplied data does not satisfy (e.g. a non-Sz configuration in an Sz-only
 * code path).
 */
class SymmetryConditionError : public SymmetryError {
 public:
  using SymmetryError::SymmetryError;
};

/**
 * @brief Raised when a projection onto a symmetry sector fails.
 */
class SymmetryProjectionError : public SymmetryError {
 public:
  using SymmetryError::SymmetryError;
};

// ---------------------------------------------------------------------------
// SymmetryBlockedTensor errors
// ---------------------------------------------------------------------------

/**
 * @brief Base class for errors raised by @ref SymmetryBlockedTensor and
 * @ref SymmetryBlockedIndexSet construction/validation.
 */
class SymmetryBlockedTensorError : public QdkError {
 public:
  using QdkError::QdkError;
};

/**
 * @brief Raised when a block's stored shape does not match the declared
 * per-axis extents.
 */
class BlockExtentMismatchError : public SymmetryBlockedTensorError {
 public:
  using SymmetryBlockedTensorError::SymmetryBlockedTensorError;
};

/**
 * @brief Raised when orbit-equivalent blocks are supplied with inconsistent
 * aliasing (e.g. orbit partners that are neither shared nor numerically
 * equal).
 */
class BlockAliasMismatchError : public SymmetryBlockedTensorError {
 public:
  using SymmetryBlockedTensorError::SymmetryBlockedTensorError;
};

/**
 * @brief Raised when a block is keyed by a symmetry label that is not
 * admissible under the tensor's per-axis symmetry vocabularies.
 */
class BlockLabelInvalidError : public SymmetryBlockedTensorError {
 public:
  using SymmetryBlockedTensorError::SymmetryBlockedTensorError;
};

}  // namespace qdk::chemistry::data
