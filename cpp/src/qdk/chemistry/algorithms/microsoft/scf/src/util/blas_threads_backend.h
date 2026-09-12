// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

/// @file Contract between ScopedBlasThreads and the vendor-specific backend.
///
/// One blas_threads_<vendor>.cpp implements this per vendor, and
/// cmake/qdk-blas-threads.cmake compiles in whichever one links against the
/// resolved BLAS, falling back to blas_threads_none.cpp. Selecting a file
/// rather than a branch keeps every build free of symbols it does not link.

namespace qdk::chemistry::scf::util::detail {

/// @brief Name of the bound backend, or nullptr if this build has no BLAS
/// thread-control API, in which case the calls below are no-ops.
const char* blas_backend_name();

/// @brief Current BLAS thread count, or 0 if the backend cannot report one.
int blas_backend_get_num_threads();

/// @brief Request `n` BLAS threads.
void blas_backend_set_num_threads(int n);

}  // namespace qdk::chemistry::scf::util::detail
