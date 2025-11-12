# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

cmake_minimum_required( VERSION 3.11 ) # Require CMake 3.11+

include( CMakePushCheckState )
include( CheckLibraryExists )
include( CheckSymbolExists )
include( FindPackageHandleStandardArgs )
include(CMakeFindDependencyMacro)

include( ${CMAKE_CURRENT_LIST_DIR}/CommonFunctions.cmake )

fill_out_prefix( blacs )

# MPI
if( NOT TARGET MPI::MPI_C )

  get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if( NOT "C" IN_LIST languages )
    enable_language( C )
  endif()

  find_dependency( MPI )

endif()


if( NOT blacs_LIBRARIES )

  message( STATUS "SEARCHING FOR BLACS_LIBRARIES" )
  find_library( BLACS_LIBRARIES
    NAMES blacs
    HINTS ${blacs_PREFIX}
    PATHS ${blacs_LIBRARY_DIR}
    PATH_SUFFICES lib lib64 lib32
  )

else()

  message( STATUS "BLACS LIBRARY WAS SET BY USER: ${blacs_LIBRARIES}" )
  set( BLACS_LIBRARIES ${blacs_LIBRARIES} )

endif()


# Test Linkage
cmake_push_check_state( RESET )

if( BLACS_LIBRARIES )
  set( CMAKE_REQUIRED_LIBRARIES ${BLACS_LIBRARIES} )
endif()
list( APPEND CMAKE_REQUIRED_LIBRARIES MPI::MPI_C )

#set( CMAKE_REQUIRED_QUIET ON )

check_library_exists( "" Cblacs_gridinit   "" BLACS_HAS_GRIDINIT  )
check_library_exists( "" Cigebs2d          "" BLACS_HAS_CIGEBS2D  )
check_library_exists( "" Csys2blacs_handle "" BLACS_HAS_SYS2BLACS )

cmake_pop_check_state()

if( BLACS_HAS_GRIDINIT AND BLACS_HAS_CIGEBS2D AND BLACS_HAS_SYS2BLACS )
  set( BLACS_LINK_OK TRUE )
endif()

mark_as_advanced( BLACS_FOUND BLACS_LIBRARIES BLACS_LINK_OK )

find_package_handle_standard_args( BLACS REQUIRED_VARS BLACS_LINK_OK )

if( BLACS_FOUND AND NOT TARGET BLACS::BLACS )

  message( STATUS "BLACS LIBRARIES: ${BLACS_LIBRARIES};MPI::MPI_C" )

  add_library( BLACS::BLACS INTERFACE IMPORTED )
  set_target_properties( BLACS::BLACS PROPERTIES
    INTERFACE_LINK_LIBRARIES  "${BLACS_LIBRARIES};MPI::MPI_C"
  )

endif()
