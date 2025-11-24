// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/utils/omp_utils.hpp>

#ifndef QDK_ENABLE_OPENMP

extern "C" {

int omp_get_thread_num() { return 0; }

int omp_get_num_threads() { return 1; }

int omp_get_max_threads() { return 1; }

void omp_set_num_threads(int) { /* no-op */ }
}

#endif
