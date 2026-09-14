// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "blas_threads_backend.h"

extern "C" {
void MKL_Set_Num_Threads(int);
int MKL_Get_Max_Threads(void);
}

namespace qdk::chemistry::scf::util::detail {

const char* blas_backend_name() { return "Intel MKL"; }

// The maximum, not a current count; they differ only under an
// MKL_Set_Num_Threads_Local override, which we never set.
int blas_backend_get_num_threads() { return MKL_Get_Max_Threads(); }

void blas_backend_set_num_threads(int n) { MKL_Set_Num_Threads(n); }

}  // namespace qdk::chemistry::scf::util::detail
