/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <sparsexx/sparsexx_config.hpp>
#if SPARSEXX_ENABLE_MKL

#if __has_include(<mkl.h>)
#include <mkl_dss.h>
#include <mkl_spblas.h>
#else
#error "FATAL: Cannot use sparsexx MKL bindings without MKL!"
#endif /* __has_include(<mkl.h>) */

namespace sparsexx::detail::mkl {
using int_type = MKL_INT;
}
#endif /* SPARSEXX_ENABLE_MKL */
