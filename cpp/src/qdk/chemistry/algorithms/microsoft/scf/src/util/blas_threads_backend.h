// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

/// @file Contract between ScopedBlasThreads and the vendor-specific backend.
///
/// One blas_threads_<vendor>.cpp implements this per vendor, and
/// cmake/qdk-blas-threads.cmake compiles in the one its BLAS vendor label
/// selects, after verifying that it links; otherwise blas_threads_none.cpp.
/// Selecting a file rather than a branch keeps every build free of symbols it
/// does not link.

namespace qdk::chemistry::scf::util::detail {

/// @brief Current BLAS thread count, or 0 if this build has no thread-control
/// API or the backend cannot report one. A 0 means the count cannot be pinned,
/// and blas_backend_set_num_threads is a no-op.
int blas_backend_get_num_threads();

/// @brief Request `n` BLAS threads.
void blas_backend_set_num_threads(int n);

}  // namespace qdk::chemistry::scf::util::detail
