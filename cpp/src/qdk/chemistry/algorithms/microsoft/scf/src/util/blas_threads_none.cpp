// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Fallback backend, compiled when no vendor implementation links.

#include "blas_threads_backend.h"

namespace qdk::chemistry::scf::util::detail {

const char* blas_backend_name() { return nullptr; }

int blas_backend_get_num_threads() { return 0; }

void blas_backend_set_num_threads(int) {}

}  // namespace qdk::chemistry::scf::util::detail
