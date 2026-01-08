# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
# Portions Copyright (c) Microsoft Corporation.
#
# See LICENSE.txt for details

find_package(mdspan CONFIG QUIET)
if( NOT mdspan_FOUND )
  include(FetchContent)

  FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG mdspan-0.6.0
  )
  set( MDSPAN_CXX_STANDARD 20 CACHE STRING "" FORCE)
  FetchContent_MakeAvailable( mdspan )
  set(MACIS_MDSPAN_EXPORT mdspan CACHE STRING "" FORCE)
else()
  # When mdspan is found externally, don't export it
  set(MACIS_MDSPAN_EXPORT "" CACHE STRING "" FORCE)
endif()
