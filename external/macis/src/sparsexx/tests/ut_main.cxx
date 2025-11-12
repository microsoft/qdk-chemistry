/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

int main(int argc, char* argv[]) {
  int result = Catch::Session().run(argc, argv);
  return result;
}
