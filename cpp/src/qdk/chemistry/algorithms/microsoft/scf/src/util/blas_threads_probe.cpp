// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Driver for the link probe in cmake/qdk-blas-threads.cmake: linking this
// against a candidate backend is what forces its vendor symbols to resolve.

#include "blas_threads_backend.h"

int main() {
  using namespace qdk::chemistry::scf::util::detail;
  blas_backend_set_num_threads(blas_backend_get_num_threads());
  return blas_backend_name() == nullptr ? 1 : 0;
}
