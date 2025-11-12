# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

# SANITY CHECK
if( "ilp64" IN_LIST BLIS_FIND_COMPONENTS AND "lp64" IN_LIST BLIS_FIND_COMPONENTS )
  message( FATAL_ERROR "BLIS cannot link to both ILP64 and LP64 iterfaces" )
endif()

if( blis_PREFERS_STATIC )
  set( blis_LIBRARY_NAME "libblis.a" )
else()
  set( blis_LIBRARY_NAME "blis" )
endif()

find_library( blis_LIBRARY
  NAMES ${blis_LIBRARY_NAME}
  HINTS ${blis_PREFIX}
  PATHS ${blis_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib lib64 lib32
  DOC "BLIS Library"
)

find_path( blis_INCLUDE_DIR
  NAMES blis/blis.h
  HINTS ${blis_PREFIX}
  PATHS ${blis_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOC "BLIS header"
)


if( blis_INCLUDE_DIR )
  set( BLIS_INCLUDE_DIR ${blis_INCLUDE_DIR} )
endif()

if( blis_LIBRARY )
  find_package( Threads QUIET )
  set( BLIS_LIBRARIES ${blis_LIBRARY} Threads::Threads "m")
endif()

# check ILP64
if( EXISTS ${BLIS_INCLUDE_DIR}/blis/blis.h )

  set( idxwidth_pattern
  "^#define[\t ]+BLIS_INT_TYPE_SIZE[\t ]+([0-9\\.]+[0-9\\.]+)$"
  )
  file( STRINGS ${BLIS_INCLUDE_DIR}/blis/blis.h blis_idxwidth
        REGEX ${idxwidth_pattern} )

  string( REGEX REPLACE ${idxwidth_pattern}
          "${BLIS_IDXWIDTH_STRING}\\1"
          BLIS_IDXWIDTH_STRING ${blis_idxwidth} )

  if( ${BLIS_IDXWIDTH_STRING} MATCHES "64" )
    set( BLIS_USES_ILP64 TRUE )
  else()
    set( BLIS_USES_ILP64 FALSE )
  endif()

  unset( idxwidth_pattern      )
  unset( blis_idxwidth        )
  unset( BLIS_IDXWIDTH_STRING )

endif()



# Handle components
if( BLIS_USES_ILP64 )
  set( BLIS_ilp64_FOUND TRUE  )
  set( BLIS_lp64_FOUND  FALSE )
else()
  set( BLIS_ilp64_FOUND FALSE )
  set( BLIS_lp64_FOUND  TRUE  )
endif()





include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( BLIS
  REQUIRED_VARS BLIS_LIBRARIES BLIS_INCLUDE_DIR
#  VERSION_VAR BLIS_VERSION_STRING
  HANDLE_COMPONENTS
)

if( BLIS_FOUND AND NOT TARGET BLIS::blis )

  add_library( BLIS::blis INTERFACE IMPORTED )
  set_target_properties( BLIS::blis PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${BLIS_LIBRARIES}"
  )

endif()
