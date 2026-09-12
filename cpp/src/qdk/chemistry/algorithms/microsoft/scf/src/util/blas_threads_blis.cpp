// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "blas_threads_backend.h"

// The header rather than a prototype: dim_t's width is a BLIS build option, so
// declaring it by hand risks an ABI mismatch. A missing header fails to
// compile, which is how the probe declines this backend.
#if __has_include(<blis/blis.h>)
#include <blis/blis.h>
#else
#include <blis.h>
#endif

namespace qdk::chemistry::scf::util::detail {

const char* blas_backend_name() { return "BLIS"; }

int blas_backend_get_num_threads() {
  return static_cast<int>(bli_thread_get_num_threads());
}

void blas_backend_set_num_threads(int n) {
  bli_thread_set_num_threads(static_cast<dim_t>(n));
}

}  // namespace qdk::chemistry::scf::util::detail
