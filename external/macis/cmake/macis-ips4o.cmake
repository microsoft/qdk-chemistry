# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
# Portions Copyright (c) Microsoft Corporation.
#
# See LICENSE.txt for details

include( FetchContent )
# IPS4O Sort
FetchContent_Declare( ips4o
  GIT_REPOSITORY https://github.com/SaschaWitt/ips4o.git
  GIT_TAG        8d5a97bb4c4048a1eef18b537f191a31462b051a
)
FetchContent_MakeAvailable( ips4o )
add_library( ips4o INTERFACE )
target_include_directories( ips4o INTERFACE ${ips4o_SOURCE_DIR} )
if(NOT WIN32 AND NOT APPLE)
  target_link_libraries( ips4o INTERFACE atomic )
endif()
