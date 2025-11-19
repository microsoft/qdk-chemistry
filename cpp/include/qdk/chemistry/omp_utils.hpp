// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#ifdef _OPENMP
#include <omp.h>
#else
// Fallbacks for non-OpenMP environments
#ifndef QDK_CHEMISTRY_OMP_FALLBACK_DEFINED
#define QDK_CHEMISTRY_OMP_FALLBACK_DEFINED
namespace {
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_max_threads() { return 1; }
inline void omp_set_num_threads(int) { /* no-op */ }
}  // namespace
#endif  // QDK_CHEMISTRY_OMP_FALLBACK_DEFINED
#endif  // _OPENMP
