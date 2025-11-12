// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace qdk::chemistry::scf {
/**
 * @brief Singleton timer class for performance profiling
 *
 * Provides timing utilities for measuring execution time of code sections.
 * Tracks min, max, average, and total time for each named timer.
 */
class Timer {
 public:
  /**
   * @brief Start timing for a named key
   *
   * Records the current time as the start point for the timer identified by
   * key. Can be called multiple times for the same key to accumulate
   * statistics.
   *
   * @param key Unique identifier for this timer
   */
  static void start_timing(const std::string& key) {
    get_instance_()->start_timing_(key);
  }

  /**
   * @brief Stop timing and record duration
   *
   * Calculates elapsed time since start_timing was called for this key.
   * Updates count, total, min, and max statistics.
   *
   * @param key Unique identifier for this timer (must match start_timing call)
   * @return Duration in milliseconds since start_timing was called
   */
  static std::chrono::duration<double, std::milli> stop_timing(
      const std::string& key) {
    return get_instance_()->stop_timing_(key);
  }

  /**
   * @brief Print summary of all timing statistics
   *
   * Outputs a formatted table showing total time, number of calls,
   * average, max, and min times for all recorded timers, sorted by total time.
   */
  static void print_summary() { get_instance_()->print_summary_(); }

  /**
   * @brief Timer record containing timing statistics
   */
  struct record_t {
    std::chrono::time_point<std::chrono::high_resolution_clock>
        start_;  ///< Start time point for current timing
    std::chrono::duration<double, std::milli>
        during_;  ///< Accumulated total duration across all calls
    std::chrono::duration<double, std::milli> min =
        std::chrono::duration<int64_t,
                              std::nano>::max();  ///< Minimum duration observed
    std::chrono::duration<double, std::milli> max =
        std::chrono::duration<int64_t,
                              std::nano>::min();  ///< Maximum duration observed
    size_t count_ = 0;  ///< Number of times this timer has been called

    /**
     * @brief Calculate average duration per call
     *
     * @return Average duration in milliseconds, or 0 if never called
     */
    inline auto avg() const { return count_ ? during_.count() / count_ : 0.0; }
  };

  /**
   * @brief Get timing record for a specific key
   *
   * @param key Timer identifier
   * @return Timer record containing all statistics for this key
   * @throws std::runtime_error if key not found
   */
  static auto get_record(const std::string& key) {
    auto& d_ = get_instance_()->d_;
    auto it = d_.find(key);
    if (it != d_.end()) return it->second;
    throw std::runtime_error("timer key: " + key + " not found");
  }

 private:
  /**
   * @brief Private constructor for singleton pattern
   */
  Timer() = default;

  /**
   * @brief Get the singleton Timer instance
   *
   * @return Shared pointer to the singleton Timer instance
   */
  static std::shared_ptr<Timer> get_instance_() {
    static std::shared_ptr<Timer> d(new Timer);
    return d;
  }

  /**
   * @brief Internal implementation of start_timing
   *
   * @param key Timer identifier
   */
  void start_timing_(const std::string& key) {
    d_[key].start_ = std::chrono::high_resolution_clock::now();
  }

  /**
   * @brief Internal implementation of stop_timing
   *
   * @param key Timer identifier
   * @return Duration since start_timing was called
   */
  std::chrono::duration<double, std::milli> stop_timing_(
      const std::string& key) {
    auto stop = std::chrono::high_resolution_clock::now();
    auto& record = d_[key];
    record.count_++;
    std::chrono::duration<double, std::milli> d = stop - record.start_;
    record.during_ += d;
    record.min = std::min(record.min, d);
    record.max = std::max(record.max, d);
    return d;
  }

  /**
   * @brief Internal implementation of print_summary
   *
   * Formats and prints timing statistics sorted by total duration.
   */
  void print_summary_() {
    std::vector<std::string> keys;
    for (const auto& [k, _] : d_) {
      keys.push_back(k);
    }
    sort(keys.begin(), keys.end(),
         [&](const std::string& x, const std::string& y) {
           return d_[y].during_ < d_[x].during_;
         });

    fmt::print("{:-^100}\n", "Performance Profile");
    fmt::print("{:>15}{:>15}{:>15}{:>15}{:>15}  {:20}\n", "total(ms)", "calls",
               "avg(ms)", "max(ms)", "min(ms)", "name");
    for (const auto& k : keys) {
      fmt::print("{:>15.3f}{:>15}{:>15.3f}{:>15.3f}{:>15.3f}  {:20}\n",
                 d_[k].during_.count(), d_[k].count_,
                 d_[k].during_.count() / d_[k].count_, d_[k].max.count(),
                 d_[k].min.count(), k);
    }
    fmt::print("{:-^100}\n", "");
  }

  std::unordered_map<std::string, record_t>
      d_;  ///< Map of timer keys to their timing records
};

/**
 * @brief RAII-style automatic timer
 *
 * Starts timing on construction and stops on destruction.
 * Useful for timing scope-based code blocks automatically.
 *
 * Example usage:
 * @code
 * {
 *   AutoTimer timer("my_function");
 *   // Code to time
 * } // Timer automatically stops here
 * @endcode
 */
class AutoTimer {
 public:
  /**
   * @brief Construct AutoTimer and start timing
   *
   * @param key Unique identifier for this timer
   */
  explicit AutoTimer(const std::string& key) : key_(key) {
    Timer::start_timing(key_);
  }

  /**
   * @brief Destructor stops timing automatically
   *
   * Called when AutoTimer goes out of scope, recording the elapsed time.
   */
  ~AutoTimer() { Timer::stop_timing(key_); }

 private:
  std::string key_;  ///< Timer identifier key
};
}  // namespace qdk::chemistry::scf
