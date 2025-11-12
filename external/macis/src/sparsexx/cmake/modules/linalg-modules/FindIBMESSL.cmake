# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

if( "ilp64" IN_LIST IBMESSL_FIND_COMPONENTS AND "lp64" IN_LIST IBMESSL_FIND_COMPONENTS )
  message( FATAL_ERROR "IBMESSL cannot link to both ILP64 and LP64 iterfaces" )
endif()

set( ibmessl_LP64_SERIAL_LIBRARY_NAME  "essl"        )
set( ibmessl_LP64_SMP_LIBRARY_NAME     "esslsmp"     )
set( ibmessl_ILP64_SERIAL_LIBRARY_NAME "essl6464"    )
set( ibmessl_ILP64_SMP_LIBRARY_NAME    "esslsmp6464" )


if( NOT ibmessl_PREFERED_THREAD_LEVEL )
  set( ibmessl_PREFERED_THREAD_LEVEL "smp" )
endif()

if( ibmessl_PREFERED_THREAD_LEVEL MATCHES "smp" )
  set( ibmessl_LP64_LIBRARY_NAME  ${ibmessl_LP64_SMP_LIBRARY_NAME}  )
  set( ibmessl_ILP64_LIBRARY_NAME ${ibmessl_ILP64_SMP_LIBRARY_NAME} )
else()
  set( ibmessl_LP64_LIBRARY_NAME  ${ibmessl_LP64_SERIAL_LIBRARY_NAME}  )
  set( ibmessl_ILP64_LIBRARY_NAME ${ibmessl_ILP64_SERIAL_LIBRARY_NAME} )
endif()


find_path( ibmessl_INCLUDE_DIR
  NAMES essl.h
  HINTS ${ibmessl_PREFIX}
  PATHS ${ibmessl_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOC "IBM(R) ESSL header"
)

find_library( ibmessl_LP64_LIBRARY
  NAMES ${ibmessl_LP64_LIBRARY_NAME}
  HINTS ${ibmessl_PREFIX}
  PATHS ${ibmessl_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib lib64 lib32
  DOC "IBM(R) ESSL Library"
)

find_library( ibmessl_ILP64_LIBRARY
  NAMES ${ibmessl_LP64_LIBRARY_NAME}
  HINTS ${ibmessl_PREFIX}
  PATHS ${ibmessl_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib lib64 lib32
  DOC "IBM(R) ESSL Library"
)

if( ibmessl_ILP64_LIBRARY )
  set( IBMESSL_ilp64_FOUND TRUE )
endif()

if( ibmessl_LP64_LIBRARY )
  set( IBMESSL_lp64_FOUND TRUE )
endif()

if( "ilp64" IN_LIST IBMESSL_FIND_COMPONENTS )
  set( ibmessl_LIBRARY ${ibmessl_ILP64_LIBRARY} )
else()
  set( ibmessl_LIBRARY ${ibmessl_LP64_LIBRARY} )
endif()



if( ibmessl_INCLUDE_DIR )
  set( IBMESSL_INCLUDE_DIR ${ibmessl_INCLUDE_DIR} )
endif()

if( ibmessl_LIBRARY )
  set( IBMESSL_LIBRARIES ${ibmessl_LIBRARY} )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( IBMESSL
  REQUIRED_VARS IBMESSL_LIBRARIES IBMESSL_INCLUDE_DIR
#  VERSION_VAR IBMESSL_VERSION_STRING
  HANDLE_COMPONENTS
)

if( IBMESSL_FOUND AND NOT TARGET IBMESSL::essl )

  add_library( IBMESSL::essl INTERFACE IMPORTED )
  set_target_properties( IBMESSL::essl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${IBMESSL_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${IBMESSL_LIBRARIES}"
  )

endif()
