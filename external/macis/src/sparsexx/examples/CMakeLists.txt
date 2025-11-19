# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

if(SPARSEXX_ENABLE_MPI)
add_executable( test_dist test_dist.cxx )
target_link_libraries( test_dist PUBLIC sparsexx )
endif()
