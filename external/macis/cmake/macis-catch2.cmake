# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
# Portions Copyright (c) Microsoft Corporation.
#
# See LICENSE.txt for details

include(FetchContent)

find_package( Catch2 3.0.1 CONFIG QUIET )
if( NOT Catch2_FOUND )
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.3.2
  )

  set(CATCH_BUILD_TESTING OFF CACHE BOOL "Build SelfTest project" FORCE)
  set(CATCH_INSTALL_DOCS OFF CACHE BOOL "Install documentation alongside library" FORCE)
  set(CATCH_INSTALL_HELPERS OFF CACHE BOOL "Install contrib alongside library" FORCE)

  FetchContent_MakeAvailable( catch2 )
endif()
