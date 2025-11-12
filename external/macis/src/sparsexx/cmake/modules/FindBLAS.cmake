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


include( ${CMAKE_CURRENT_LIST_DIR}/CommonFunctions.cmake )

# SANITY CHECK
if( "ilp64" IN_LIST BLAS_FIND_COMPONENTS AND "lp64" IN_LIST BLAS_FIND_COMPONENTS )
  message( FATAL_ERROR "BLAS cannot link to both ILP64 and LP64 iterfaces" )
endif()


# Get list of required / optional components
foreach( _comp ${BLAS_FIND_COMPONENTS} )
  if( BLAS_FIND_REQUIRED_${_comp} )
    list( APPEND BLAS_REQUIRED_COMPONENTS ${_comp} )
  else()
    list( APPEND BLAS_OPTIONAL_COMPONENTS ${_comp} )
  endif()
endforeach()

fill_out_prefix( blas )

if( NOT BLAS_PREFERENCE_LIST )
  set( BLAS_PREFERENCE_LIST "IntelMKL" "IBMESSL" "BLIS" "OpenBLAS" "ReferenceBLAS" )
endif()

if( NOT blas_LIBRARIES )

  list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/linalg-modules )
  foreach( blas_type ${BLAS_PREFERENCE_LIST} )

    string( TOLOWER ${blas_type} blas_lower_case )
    copy_meta_data( blas ${blas_lower_case} )


    find_package( ${blas_type}
      COMPONENTS          ${BLAS_REQUIRED_COMPONENTS}
      OPTIONAL_COMPONENTS ${BLAS_OPTIONAL_COMPONENTS}
    )

    if( ${blas_type}_FOUND )

      set( BLAS_VENDOR ${blas_type} )

      if    ( ${blas_type} MATCHES "IntelMKL" )
        set( blas_LIBRARIES IntelMKL::mkl )
      elseif( ${blas_type} MATCHES "IBMESSL" )
        set( blas_LIBRARIES IBMESSL::essl )
      elseif( ${blas_type} MATCHES "BLIS" )
        set( blas_LIBRARIES BLIS::blis )
      elseif( ${blas_type} MATCHES "OpenBLAS" )
        set( blas_LIBRARIES OpenBLAS::openblas )
      elseif( ${blas_type} MATCHES "ReferenceBLAS" )
        set( blas_LIBRARIES ReferenceBLAS::blas )
      endif()

      # Propagate integers
      if( "ilp64" IN_LIST BLAS_FIND_COMPONENTS )
        set( BLAS_ilp64_FOUND ${${blas_type}_ilp64_FOUND} )
      else()
        set( BLAS_lp64_FOUND ${${blas_type}_lp64_FOUND} )
      endif()

      # Propagate BLACS / ScaLAPACK
      if( "blacs" IN_LIST BLAS_FIND_COMPONENTS )
        set( BLAS_blacs_FOUND ${${blas_type}_blacs_FOUND} )
      endif()
      if( "scalapack" IN_LIST BLAS_FIND_COMPONENTS )
        set( BLAS_scalapack_FOUND ${${blas_type}_scalapack_FOUND} )
      endif()


      break()

    endif()

  endforeach()

  list(REMOVE_AT CMAKE_MODULE_PATH -1)
endif()

if( NOT BLAS_ilp64_FOUND )
  set( BLAS_ilp64_FOUND FALSE )
endif()
if( NOT BLAS_lp64_FOUND )
  set( BLAS_lp64_FOUND FALSE )
endif()


if( BLAS_ilp64_FOUND )
  set( BLAS_USES_ILP64 TRUE )
else()
  set( BLAS_USES_ILP64 FALSE )
endif()

# Handle implicit BLAS linkage
if( blas_LIBRARIES MATCHES "[Ii][Mm][Pp][Ll][Ii][Cc][Ii][Tt]" )
  unset( blas_LIBRARIES )
endif()

# Check function existance and linkage / name mangling
cmake_push_check_state( RESET )
if( blas_LIBRARIES )
  set( CMAKE_REQUIRED_LIBRARIES ${blas_LIBRARIES} )
endif()
set( CMAKE_REQUIRED_QUIET ON )

check_library_exists( "" dgemm       "" BLAS_NO_UNDERSCORE   )
check_library_exists( "" dgemm_      "" BLAS_USES_UNDERSCORE )

set( TEST_USES_UNDERSCORE_STR "Performing Test BLAS_USES_UNDERSCORE" )
set( TEST_NO_UNDERSCORE_STR   "Performing Test BLAS_NO_UNDERSCORE"   )

message( STATUS  ${TEST_USES_UNDERSCORE_STR} )
if( BLAS_USES_UNDERSCORE )
  message( STATUS "${TEST_USES_UNDERSCORE_STR} -- found" )
else()
  message( STATUS "${TEST_USES_UNDERSCORE_STR} -- not found" )
endif()

message( STATUS  ${TEST_NO_UNDERSCORE_STR} )
if( BLAS_NO_UNDERSCORE )
  message( STATUS "${TEST_NO_UNDERSCORE_STR} -- found" )
else()
  message( STATUS "${TEST_NO_UNDERSCORE_STR} -- not found" )
endif()

unset( TEST_USES_UNDERSCORE_STR )
unset( TEST_NO_UNDERSCORE_STR )


cmake_pop_check_state()

if( BLAS_NO_UNDERSCORE OR BLAS_USES_UNDERSCORE )
  set( BLAS_LINK_OK TRUE )
endif()


find_package_handle_standard_args( BLAS
  REQUIRED_VARS BLAS_LINK_OK
  HANDLE_COMPONENTS
)

if( BLAS_FOUND AND NOT TARGET BLAS::blas )

  set( BLAS_LIBRARIES ${blas_LIBRARIES} )

  add_library( BLAS::blas INTERFACE IMPORTED )
  set_target_properties( BLAS::blas PROPERTIES
    INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}"
  )

endif()
