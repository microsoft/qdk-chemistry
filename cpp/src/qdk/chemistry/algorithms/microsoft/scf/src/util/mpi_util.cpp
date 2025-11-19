// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "util/mpi_util.h"

#include <chrono>
#include <thread>

#include "util/timer.h"

namespace qdk::chemistry::scf {
MPISharedCounter::MPISharedCounter(int rank, int init_value, int owner_rank,
                                   int buffer_size)
    : rank_(rank),
      owner_rank_(owner_rank),
      counter_(nullptr),
      stop_(false),
      buffer_size_(buffer_size) {
  AutoTimer timer("MPISharedCounter::create");
  if (rank_ == owner_rank_) {
    MPI_Alloc_mem(1, MPI_INFO_NULL, &counter_);
    *counter_ = init_value;
  }
  MPI_Win_create(counter_, (rank_ == owner_rank_ ? 1 : 0) * sizeof(int),
                 sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_);

  auto worker_fn = [&]() {
    using namespace std::chrono_literals;

    auto fetch_one = [&]() {
      int val, one = 1;
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner_rank_, 0, win_);
      MPI_Fetch_and_op(&one, &val, MPI_INT, owner_rank_, 0, MPI_SUM, win_);
      MPI_Win_unlock(owner_rank_, win_);
      return val;
    };

    while (!stop_) {
      auto val = fetch_one();
      while (!stop_) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (not_full_.wait_for(lock, 10ms, [this]() {
              return (int)jobs_.size() < buffer_size_;
            })) {
          jobs_.push(val);
          lock.unlock();
          not_empty_.notify_one();
          break;
        }
      }
    }
  };
  worker_ = std::make_shared<std::thread>(worker_fn);
}

MPISharedCounter::~MPISharedCounter() {
  AutoTimer timer("MPISharedCounter::destroy");
  stop_ = true;
  worker_->join();
  MPI_Win_free(&win_);
  if (counter_) {
    MPI_Free_mem(counter_);
  }
}

int MPISharedCounter::next() {
  AutoTimer timer("MPISharedCounter::next");
  std::unique_lock<std::mutex> lock(mutex_);
  not_empty_.wait(lock, [this]() { return jobs_.size() > 0; });
  auto val = jobs_.front();
  jobs_.pop();
  lock.unlock();
  not_full_.notify_one();
  return val;
}
}  // namespace qdk::chemistry::scf
