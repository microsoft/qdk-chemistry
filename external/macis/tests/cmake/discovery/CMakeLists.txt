# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
# Portions Copyright (c) Microsoft Corporation.
#
# See LICENSE.txt for details

cmake_minimum_required( VERSION 3.18 FATAL_ERROR )
project( macis_cmake_discovery LANGUAGES C CXX )

find_package( macis REQUIRED )
add_executable( macis_link_tester macis_link_tester.cxx )
target_link_libraries( macis_link_tester PUBLIC macis::macis )
