// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "blas_threads_backend.h"

extern "C" {
void openblas_set_num_threads(int);
int openblas_get_num_threads(void);
}

namespace qdk::chemistry::scf::util::detail {

const char* blas_backend_name() { return "OpenBLAS"; }

int blas_backend_get_num_threads() { return openblas_get_num_threads(); }

void blas_backend_set_num_threads(int n) { openblas_set_num_threads(n); }

}  // namespace qdk::chemistry::scf::util::detail
