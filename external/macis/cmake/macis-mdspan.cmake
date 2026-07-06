# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
# Portions Copyright (c) Microsoft Corporation.
#
# See LICENSE.txt for details

include(FetchContent)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG 3fdf85b01e10629ddb18a0a3ffd468d7f9cfa185
)
set( MDSPAN_CXX_STANDARD 20 CACHE STRING "" FORCE)
FetchContent_MakeAvailable( mdspan )
