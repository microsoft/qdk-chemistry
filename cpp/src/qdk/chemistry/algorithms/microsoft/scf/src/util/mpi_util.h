// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <mpi.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace qdk::chemistry::scf {
/**
 * @brief Thread-safe MPI shared counter for work distribution
 *
 * Provides a distributed counter using MPI one-sided communication (RMA).
 * One rank owns the counter, others fetch work indices atomically.
 * A background worker thread prefetches values to hide MPI latency.
 */
class MPISharedCounter {
 public:
  /**
   * @brief Create a shared counter
   * @param rank MPI rank of this process
   * @param init_value Initial counter value (default: 0)
   * @param owner_rank Rank that owns the counter memory (default: 0)
   * @param buffer_size Number of values to prefetch (default: 1)
   */
  explicit MPISharedCounter(int rank, int init_value = 0, int owner_rank = 0,
                            int buffer_size = 1);

  /**
   * @brief Destroy the shared counter
   */
  ~MPISharedCounter();

  MPISharedCounter(const MPISharedCounter&) = delete;
  MPISharedCounter(MPISharedCounter&&) noexcept = delete;
  MPISharedCounter& operator=(const MPISharedCounter&) = delete;
  MPISharedCounter& operator=(MPISharedCounter&&) noexcept = delete;

  /**
   * @brief Get the next value from the counter (thread-safe)
   * @return Next counter value
   */
  int next();

 private:
  int rank_;        ///< MPI rank of this process
  int owner_rank_;  ///< Rank that owns the counter
  MPI_Win win_;     ///< MPI window for RMA operations
  int* counter_;    ///< Pointer to counter memory

  std::mutex mutex_;                   ///< Protects job queue
  std::condition_variable not_empty_;  ///< Signals when jobs are available
  std::condition_variable not_full_;   ///< Signals when buffer has space
  std::atomic_bool stop_;              ///< Signals worker thread to stop
  std::queue<int> jobs_;               ///< Prefetched job indices
  int buffer_size_;                    ///< Maximum number of prefetched values
  std::shared_ptr<std::thread> worker_;  ///< Background worker thread
};
}  // namespace qdk::chemistry::scf
