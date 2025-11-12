// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <cstddef>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace qdk::chemistry::scf::util {

/**
 * @brief Generic cache for storing unique pointers to objects
 *
 * Provides hash-based storage and retrieval of objects without unnecessary
 * copies. Objects are stored as unique_ptr and accessed via raw pointers.
 *
 * @tparam T Type of objects to cache
 */
template <typename T>
class Cache {
 public:
  using IndexType = std::size_t;  ///< Type used for hash values
  using ValueType = T;            ///< Type of cached objects

  /**
   * @brief Insert an object in-place using perfect forwarding
   * @tparam Args Constructor argument types
   * @param idx Hash index for the object
   * @param args Constructor arguments
   * @return Pointer to inserted or existing object
   */
  template <typename... Args>
  T* emplace(IndexType idx, Args&&... args) {
    auto it = cache_.find(idx);
    if (it == cache_.end()) {
      auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
      T* raw_ptr = ptr.get();
      cache_[idx] = std::move(ptr);
      return raw_ptr;
    }
    return it->second.get();
  }

  /**
   * @brief Get a pointer to the object by index
   * @param idx Hash index to look up
   * @return Pointer to object or nullptr if not found
   */
  T* get(IndexType idx) noexcept {
    auto it = cache_.find(idx);
    return (it != cache_.end()) ? it->second.get() : nullptr;
  }

  /**
   * @brief Get a const pointer to the object by index
   * @param idx Hash index to look up
   * @return Const pointer to object or nullptr if not found
   */
  const T* get(IndexType idx) const noexcept {
    auto it = cache_.find(idx);
    return (it != cache_.end()) ? it->second.get() : nullptr;
  }

  /**
   * @brief Look up an object by hashing the identifier
   * @tparam Identifier Type that must have a std::hash specialization
   * @param id Identifier to hash and look up
   * @return Pointer to object or nullptr if not found
   */
  template <typename Identifier>
  T* find_by_identifier(const Identifier& id) noexcept {
    IndexType idx = std::hash<Identifier>{}(id);
    return get(idx);
  }

  /**
   * @brief Look up an object by hashing the identifier (const version)
   * @tparam Identifier Type that must have a std::hash specialization
   * @param id Identifier to hash and look up
   * @return Const pointer to object or nullptr if not found
   */
  template <typename Identifier>
  const T* find_by_identifier(const Identifier& id) const noexcept {
    IndexType idx = std::hash<Identifier>{}(id);
    return get(idx);
  }

  /**
   * @brief Look up object by identifier hash using operator[]
   * @tparam Identifier Type that must have a std::hash specialization
   * @param id Identifier to hash and look up
   * @return Pointer to object or nullptr if not found
   */
  template <typename Identifier>
  T* operator[](const Identifier& id) noexcept {
    return find_by_identifier(id);
  }

  /**
   * @brief Look up object by identifier hash using operator[] (const)
   * @tparam Identifier Type that must have a std::hash specialization
   * @param id Identifier to hash and look up
   * @return Const pointer to object or nullptr if not found
   */
  template <typename Identifier>
  const T* operator[](const Identifier& id) const noexcept {
    return find_by_identifier(id);
  }

  /**
   * @brief Remove an object from the cache
   * @param idx Hash index of object to remove
   */
  void erase(IndexType idx) { cache_.erase(idx); }

  /**
   * @brief Clear all objects from the cache
   */
  void clear() { cache_.clear(); }

 private:
  std::unordered_map<IndexType, std::unique_ptr<T>>
      cache_;  ///< Hash map storing cached objects
};

}  // namespace qdk::chemistry::scf::util
