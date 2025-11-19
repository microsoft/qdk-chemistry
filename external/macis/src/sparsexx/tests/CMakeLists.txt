# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

include( FetchContent )
find_package( Catch2 CONFIG QUIET )
if( NOT Catch2_FOUND )

  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.2
  )

  set(CATCH_BUILD_TESTING OFF CACHE BOOL "Build SelfTest project" FORCE)
  set(CATCH_INSTALL_DOCS OFF CACHE BOOL "Install documentation alongside library"
    FORCE)
  set(CATCH_INSTALL_HELPERS OFF CACHE BOOL "Install contrib alongside library"
    FORCE)

  FetchContent_MakeAvailable( catch2 )

endif()

add_executable( sparsexx_test ut_main.cxx test_types.cxx test_io.cxx test_util.cxx test_graph.cxx test_spmbv.cxx )
target_link_libraries( sparsexx_test PUBLIC sparsexx Catch2::Catch2 )
target_compile_definitions( sparsexx_test PUBLIC
  "SPARSEXX_DATA_DIR=\"${CMAKE_CURRENT_LIST_DIR}\""
)
