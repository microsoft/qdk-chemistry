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
)
FetchContent_MakeAvailable( ips4o )
add_library( ips4o INTERFACE )
target_include_directories( ips4o INTERFACE ${ips4o_SOURCE_DIR} )
target_link_libraries( ips4o INTERFACE atomic )
