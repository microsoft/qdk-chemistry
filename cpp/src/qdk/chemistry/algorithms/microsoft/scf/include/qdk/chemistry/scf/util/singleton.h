// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <atomic>
#include <memory>
#include <mutex>

namespace qdk::chemistry::scf::util {

/**
 * @brief A singleton utility class that generates and manages a singleton
 * instance of type T
 *
 * This class provides a thread-safe way to create and access a singleton
 * instance of any class. The singleton instance is created lazily on first
 * access.
 *
 * @tparam T The type of the singleton instance to be created
 */
template <typename T>
class Singleton {
 public:
  // Delete copy constructor and assignment operator
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;

  /**
   * @brief Get the singleton instance of type T
   *
   * @tparam Args The argument types for constructing T
   * @param args The arguments for constructing T
   * @return T& Reference to the singleton instance
   */
  template <typename... Args>
  static T& instance(Args&&... args) {
    if (!isInitialized_) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!isInitialized_) {
        instance_ = std::make_unique<T>(std::forward<Args>(args)...);
        isInitialized_ = true;
      }
    }
    return *instance_;
  }

  /**
   * @brief Reset the singleton instance
   *
   * This method destroys the current singleton instance.
   * The next call to instance() will create a new instance.
   */
  static void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    instance_.reset();
    isInitialized_ = false;
  }

 private:
  // Private constructor to prevent direct instantiation
  Singleton() = default;

  // The singleton instance
  static std::unique_ptr<T> instance_;

  // Mutex for thread-safe initialization and reset
  static std::mutex mutex_;

  // Flag to track initialization state
  static std::atomic<bool> isInitialized_;
};

// Define static members
template <typename T>
std::unique_ptr<T> Singleton<T>::instance_;

template <typename T>
std::mutex Singleton<T>::mutex_;

template <typename T>
std::atomic<bool> Singleton<T>::isInitialized_{false};

}  // namespace qdk::chemistry::scf::util
