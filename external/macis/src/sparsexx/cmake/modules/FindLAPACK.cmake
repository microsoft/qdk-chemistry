# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

cmake_minimum_required( VERSION 3.11 ) # Require CMake 3.11+

include( CMakePushCheckState )
include( CheckLibraryExists )
include( CheckSymbolExists )
include( CMakeFindDependencyMacro )
include( FindPackageHandleStandardArgs )


include( ${CMAKE_CURRENT_LIST_DIR}/CommonFunctions.cmake )

# SANITY CHECK
if( "ilp64" IN_LIST LAPACK_FIND_COMPONENTS AND "lp64" IN_LIST LAPACK_FIND_COMPONENTS )
  message( FATAL_ERROR "LAPACK cannot link to both ILP64 and LP64 iterfaces" )
endif()


# Get list of required / optional components
foreach( _comp ${LAPACK_FIND_COMPONENTS} )
  if( LAPACK_FIND_REQUIRED_${_comp} )
    list( APPEND LAPACK_REQUIRED_COMPONENTS ${_comp} )
  else()
    list( APPEND LAPACK_OPTIONAL_COMPONENTS ${_comp} )
  endif()
endforeach()

fill_out_prefix( lapack )

if( NOT LAPACK_PREFERENCE_LIST )
  set( LAPACK_PREFERENCE_LIST "ReferenceLAPACK" )
endif()

if( NOT lapack_LIBRARIES )

  if( NOT TARGET BLAS::blas )
    find_dependency( BLAS
      COMPONENTS          ${LAPACK_REQUIRED_COMPONENTS}
      OPTIONAL_COMPONENTS ${LAPACK_OPTIONAL_COMPONENTS}
    )
  endif()

  # Check if BLAS LINKS to LAPACK
  cmake_push_check_state( RESET )
  set( CMAKE_REQUIRED_LIBRARIES BLAS::blas )
  set( CMAKE_REQUIRED_QUIET     ON         )

  check_library_exists( "" dsyev_ "" blas_HAS_DSYEV_UNDERSCORE    )
  check_library_exists( "" dsyev  "" blas_HAS_DSYEV_NO_UNDERSCORE )

  cmake_pop_check_state()

  if( blas_HAS_DSYEV_UNDERSCORE OR blas_HAS_DSYEV_NO_UNDERSCORE )
    set( blas_HAS_LAPACK TRUE )
  endif()

  unset( blas_HAS_DSYEV_UNDERSCORE    )
  unset( blas_HAS_DSYEV_NO_UNDERSCORE )

  if( blas_HAS_LAPACK )
    set( lapack_LIBRARIES       BLAS::blas              )
    set( LAPACK_VENDOR          ${BLAS_VENDOR}          )
    if( BLAS_USES_ILP64 )
      set( LAPACK_ilp64_FOUND TRUE )
    else()
      set( LAPACK_lp64_FOUND TRUE )
    endif()
  else()

    if( BLAS_USES_ILP64 )
      message( FATAL_ERROR "ReferenceLAPACK cannot be compiled ILP64" )
    endif()

    list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/linalg-modules )
    foreach( lapack_type ${LAPACK_PREFERENCE_LIST} )

      string( TOLOWER ${lapack_type} lapack_lower_case )
      set( ${lapack_lower_case}_PREFIX         ${lapack_PREFIX}         )
      set( ${lapack_lower_case}_INCLUDE_DIR    ${lapack_INCLUDE_DIR}    )
      set( ${lapack_lower_case}_LIBRARY_DIR    ${lapack_LIBRARY_DIR}    )
      set( ${lapack_lower_case}_PREFERS_STATIC ${lapack_PREFERS_STATIC} )

      find_package( ${lapack_type}
        COMPONENTS          ${LAPACK_REQUIRED_COMPONENTS}
        OPTIONAL_COMPONENTS ${LAPACK_OPTIONAL_COMPONENTS}
      )

      if( ${lapack_type}_FOUND )

        set( LAPACK_VENDOR ${lapack_type} )

        if( ${lapack_type} MATCHES "ReferenceLAPACK" )
          set( lapack_LIBRARIES ReferenceLAPACK::lapack )
        endif()

        # Propagate integers
        if( "ilp64" IN_LIST LAPACK_FIND_COMPONENTS )
          set( LAPACK_ilp64_FOUND ${${lapack_type}_ilp64_FOUND} )
        else()
          set( LAPACK_lp64_FOUND ${${lapack_type}_lp64_FOUND} )
        endif()

        break()

      endif()

    endforeach()

    list(REMOVE_AT CMAKE_MODULE_PATH -1)

    # Append BLAS to LAPACK
    if( lapack_LIBRARIES )
      list( APPEND lapack_LIBRARIES BLAS::blas )
    endif()

  endif()
endif()

if( NOT LAPACK_ilp64_FOUND )
  set( LAPACK_ilp64_FOUND FALSE )
endif()
if( NOT LAPACK_lp64_FOUND )
  set( LAPACK_lp64_FOUND FALSE )
endif()


if( LAPACK_ilp64_FOUND )
  set( LAPACK_USES_ILP64 TRUE )
else()
  set( LAPACK_USES_ILP64 FALSE )
endif()



# Handle implicit LAPACK linkage
if( lapack_LIBRARIES MATCHES "[Ii][Mm][Pp][Ll][Ii][Cc][Ii][Tt]" )
  unset( lapack_LIBRARIES )
endif()

# Check function existance and linkage / name mangling
cmake_push_check_state( RESET )
if( lapack_LIBRARIES )
  set( CMAKE_REQUIRED_LIBRARIES ${lapack_LIBRARIES} )
endif()
set( CMAKE_REQUIRED_QUIET ON )

check_library_exists( "" dsyev       "" LAPACK_NO_UNDERSCORE   )
check_library_exists( "" dsyev_      "" LAPACK_USES_UNDERSCORE )

set( TEST_USES_UNDERSCORE_STR "Performing Test LAPACK_USES_UNDERSCORE" )
set( TEST_NO_UNDERSCORE_STR   "Performing Test LAPACK_NO_UNDERSCORE"   )

message( STATUS  ${TEST_USES_UNDERSCORE_STR} )
if( LAPACK_USES_UNDERSCORE )
  message( STATUS "${TEST_USES_UNDERSCORE_STR} -- found" )
else()
  message( STATUS "${TEST_USES_UNDERSCORE_STR} -- not found" )
endif()

message( STATUS  ${TEST_NO_UNDERSCORE_STR} )
if( LAPACK_NO_UNDERSCORE )
  message( STATUS "${TEST_NO_UNDERSCORE_STR} -- found" )
else()
  message( STATUS "${TEST_NO_UNDERSCORE_STR} -- not found" )
endif()

unset( TEST_USES_UNDERSCORE_STR )
unset( TEST_NO_UNDERSCORE_STR )


cmake_pop_check_state()

if( LAPACK_NO_UNDERSCORE OR LAPACK_USES_UNDERSCORE )
  set( LAPACK_LINK_OK TRUE )
endif()


find_package_handle_standard_args( LAPACK
  REQUIRED_VARS LAPACK_LINK_OK
  HANDLE_COMPONENTS
)

if( LAPACK_FOUND AND NOT TARGET LAPACK::lapack )

  set( LAPACK_LIBRARIES ${lapack_LIBRARIES} )

  add_library( LAPACK::lapack INTERFACE IMPORTED )
  set_target_properties( LAPACK::lapack PROPERTIES
    INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}"
  )

endif()
