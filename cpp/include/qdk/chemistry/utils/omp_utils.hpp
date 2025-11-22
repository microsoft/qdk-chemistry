// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/config.hpp>

#ifdef QDK_ENABLE_OPENMP
#include <omp.h>
#else

extern "C" {

/**
 * @brief Fallback replacement for `omp_get_thread_num` for when QDK/Chemistry
 * is built without OpenMP bindings.
 *
 * See https://www.openmp.org/spec-html/5.0/openmpsu113.html for details
 *
 * @returns 0
 */
int omp_get_thread_num();

/**
 * @brief Fallback replacement for `omp_get_num_threads` for when QDK/Chemistry
 * is built without OpenMP bindings.
 *
 * See https://www.openmp.org/spec-html/5.0/openmpsu111.html for details
 *
 * @returns 1
 */
int omp_get_num_threads();

/**
 * @brief Fallback replacement for `omp_get_max_threads` for when QDK/Chemistry
 * is built without OpenMP bindings.
 *
 * See https://www.openmp.org/spec-html/5.0/openmpsu112.html for details
 *
 * @returns 1
 */
int omp_get_max_threads();

/**
 * @brief Fallback replacement for `omp_set_num_threads` for when QDK/Chemistry
 * is built without OpenMP bindings.
 *
 * See https://www.openmp.org/spec-html/5.0/openmpsu110.html for details
 *
 * @param[in] n Number of threads to set for the OpenMP parallel region (unused)
 */
void omp_set_num_threads(int n);
}

#endif  // _OPENMP
