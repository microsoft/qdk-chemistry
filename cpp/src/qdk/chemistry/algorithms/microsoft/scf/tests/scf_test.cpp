// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "scf_test.h"

TEST_P(SCFTest, CheckEnergyGradients) { Run(); }

// clang-format off
/* ==================== H2O, RHF/RKS, spherical ==================== */
INSTANTIATE_TEST_CASE_P(RKS_INCORE_h2o, SCFTest, ::testing::Values(
  std::make_tuple("h2o_gauxc", "RHF/HF/6-31g*/pure",        "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "RKS/M06-2X/6-31g*/pure",    "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "RKS/wB97x/def2-svp/pure", "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5)
)
);

INSTANTIATE_TEST_CASE_P(RKS_CPU_h2o, SCFTest, ::testing::Values(
  std::make_tuple("h2o_gauxc", "RHF/HF/6-31g*/pure",        "CPU", "CPU", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "RKS/M06-2X/6-31g*/pure",    "CPU", "CPU", "GAUXC", 1e-6, 1e-5)));
  //std::make_tuple("h2o_gauxc", "RKS/wB97x/def2-svp/pure", "CPU", "CPU", "GAUXC", 1e-6, 1e-5)));
INSTANTIATE_TEST_CASE_P(RKS_DFJ_h2o, SCFTest, ::testing::Values(
  std::make_tuple("h2o_gauxc", "RHF-DFJ/def2-svp/def2-universal-jfit/pure",        "INCORE",  "DEFAULT", "GAUXC", 1e-8, 1e-8),
  std::make_tuple("h2o_gauxc", "RKS-DFJ/PBE/def2-svp/def2-universal-jfit/pure",    "INCORE",  "DEFAULT", "GAUXC", 1e-8, 1e-8),
  std::make_tuple("h2o_gauxc", "RKS-DFJ/M06-2X/def2-svp/def2-universal-jfit/pure", "INCORE",  "DEFAULT", "GAUXC", 1e-8, 1e-8)
));


/* ==================== H2O, UHF/UKS, spherical ==================== */
INSTANTIATE_TEST_CASE_P(UKS_INCORE_h2o, SCFTest, ::testing::Values(
  std::make_tuple("h2o_gauxc", "UHF/HF/6-31g*/pure",        "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "UKS/M06-2X/6-31g*/pure",    "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "UKS/wB97x/def2-svp/pure", "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5)));
INSTANTIATE_TEST_CASE_P(UKS_CPU_h2o, SCFTest, ::testing::Values(
  std::make_tuple("h2o_gauxc", "UHF/HF/6-31g*/pure",        "CPU", "CPU", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "UKS/M06-2X/6-31g*/pure",    "CPU", "CPU", "GAUXC", 1e-6, 1e-5)));
  //std::make_tuple("h2o_gauxc", "UKS/wB97x/def2-svp/pure", "CPU", "CPU", "GAUXC", 1e-6, 1e-5)));


/* ==================== H2O, RHF/RKS, cartesian ==================== */
INSTANTIATE_TEST_CASE_P(RKS_INCORE_h2o_cart, SCFTest, ::testing::Values(
  std::make_tuple("h2o_gauxc", "RHF/HF/6-31g*/cart",        "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "RKS/M06-2X/6-31g*/cart",    "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "RKS/wB97x/def2-svp/cart", "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5)));
INSTANTIATE_TEST_CASE_P(RKS_CPU_h2o_cart, SCFTest, ::testing::Values(
  std::make_tuple("h2o_gauxc", "RHF/HF/6-31g*/cart",        "CPU", "CPU", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "RKS/M06-2X/6-31g*/cart",    "CPU", "CPU", "GAUXC", 1e-6, 1e-5)));
  //std::make_tuple("h2o_gauxc", "RKS/wB97x/def2-svp/cart", "CPU", "CPU", "GAUXC", 1e-6, 1e-5)));

/* ==================== H2O, UHF/UKS, cartesian ==================== */
INSTANTIATE_TEST_CASE_P(UKS_INCORE_h2o_cart, SCFTest, ::testing::Values(
  std::make_tuple("h2o_gauxc", "UHF/HF/6-31g*/cart",        "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "UKS/M06-2X/6-31g*/cart",    "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "UKS/wB97x/def2-svp/cart", "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5)));
INSTANTIATE_TEST_CASE_P(UKS_CPU_h2o_cart, SCFTest, ::testing::Values(
  std::make_tuple("h2o_gauxc", "UHF/HF/6-31g*/cart",        "CPU", "CPU", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("h2o_gauxc", "UKS/M06-2X/6-31g*/cart",    "CPU", "CPU", "GAUXC", 1e-6, 1e-5)));
  //std::make_tuple("h2o_gauxc", "UKS/wB97x/def2-svp/cart", "CPU", "CPU", "GAUXC", 1e-6, 1e-5)));

/* ==================== O2 UHF-DFJ, pure =========================*/
INSTANTIATE_TEST_CASE_P(UHF_DFJ_o2, SCFTest, ::testing::Values(
  std::make_tuple("o2", "UHF-DFJ/HF/def2-svp/def2-universal-jfit/pure", "INCORE",  "DEFAULT", "GAUXC", 1e-8, 1e-8)
));
INSTANTIATE_TEST_CASE_P(UKS_DFJ_bf, SCFTest, ::testing::Values(
  std::make_tuple("bf", "UKS-DFJ/PBE/sto-3g/def2-universal-jfit/pure", "INCORE",  "DEFAULT", "GAUXC", 1e-8, 1e-8)
));

#ifdef QDK_CHEMISTRY_RUN_LONG_TESTS
/* ==================== Water10, RHF/RKS, spherical ==================== */
INSTANTIATE_TEST_CASE_P(RKS_INCORE_water10, SCFTest, ::testing::Values(
  std::make_tuple("water10_gauxc", "RHF/HF/6-31g*/pure",        "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("water10_gauxc", "RKS/M06-2X/6-31g*/pure",    "INCORE", "INCORE", "GAUXC", 2e-6, 2e-5)
//  std::make_tuple("water10_gauxc", "RKS/wB97x/def2-svp/pure", "INCORE", "INCORE", "GAUXC", 1e-6, 2e-5) // Too memory intensive for GPU w/ GAUXC requirements
));
INSTANTIATE_TEST_CASE_P(RKS_CPU_water10, SCFTest, ::testing::Values(
  std::make_tuple("water10_gauxc", "RHF/HF/6-31g*/pure",        "CPU", "CPU", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("water10_gauxc", "RKS/M06-2X/6-31g*/pure",    "CPU", "CPU", "GAUXC", 2e-6, 2e-5)));
  //std::make_tuple("water10_gauxc", "RKS/wB97x/def2-svp/pure", "CPU", "CPU", "GAUXC", 1e-6, 2e-5)));

/* ==================== Water10, UHF/UKS, spherical ==================== */
INSTANTIATE_TEST_CASE_P(UKS_INCORE_water10, SCFTest, ::testing::Values(
  std::make_tuple("water10_gauxc", "UHF/HF/6-31g*/pure",        "INCORE", "INCORE", "GAUXC", 1e-6, 2e-4),
  std::make_tuple("water10_gauxc", "UKS/M06-2X/6-31g*/pure",    "INCORE", "INCORE", "GAUXC", 2e-5, 1e-4)
//  std::make_tuple("water10_gauxc", "UKS/wB97x/def2-svp/pure", "INCORE", "INCORE", "GAUXC", 2e-5, 5e-4) // Too memory intensive for GPUs w/ GAUXC
));
INSTANTIATE_TEST_CASE_P(UKS_CPU_water10, SCFTest, ::testing::Values(
  std::make_tuple("water10_gauxc", "UHF/HF/6-31g*/pure",        "CPU", "CPU", "GAUXC", 1e-6, 2e-4),
  std::make_tuple("water10_gauxc", "UKS/M06-2X/6-31g*/pure",    "CPU", "CPU", "GAUXC", 2e-5, 1e-4)));
  //std::make_tuple("water10_gauxc", "UKS/wB97x/def2-svp/pure", "CPU", "CPU", "GAUXC", 2e-5, 5e-4)));

/* ==================== Water10, RHF/RKS, cartesian ==================== */
INSTANTIATE_TEST_CASE_P(RKS_INCORE_water10_cart, SCFTest, ::testing::Values(
  std::make_tuple("water10_gauxc", "RHF/HF/6-31g*/cart",        "INCORE", "INCORE", "GAUXC", 1e-6, 1e-5),
  std::make_tuple("water10_gauxc", "RKS/M06-2X/6-31g*/cart",    "INCORE", "INCORE", "GAUXC", 2e-6, 2e-5)
//  std::make_tuple("water10_gauxc", "RKS/wB97x/def2-svp/cart", "INCORE", "INCORE", "GAUXC", 1e-6, 2e-5) // Too memory intensive got GPU w/ GAUXC requirements
));

/* ==================== Water10, UHF/UKS, cartesian ==================== */
INSTANTIATE_TEST_CASE_P(UKS_INCORE_water10_cart, SCFTest, ::testing::Values(
  std::make_tuple("water10_gauxc", "UHF/HF/6-31g*/cart",        "INCORE", "INCORE", "GAUXC", 1e-6, 5e-4),
  std::make_tuple("water10_gauxc", "UKS/M06-2X/6-31g*/cart",    "INCORE", "INCORE", "GAUXC", 2e-6, 2e-4)
//  std::make_tuple("water10_gauxc", "UKS/wB97x/def2-svp/cart", "INCORE", "INCORE", "GAUXC", 2e-5, 5e-4) // Too memory intensive for GPU w/ GAUXC
));


/* ==================== Water10, RHF, LinearDep =========================== */
INSTANTIATE_TEST_CASE_P(RHF_LINEAR_DEP, SCFTest, ::testing::Values(
  std::make_tuple("water10", "RHF/HF/6-311++g**/pure/lindep", "CPU", "CPU", "GAUXC", 1e-6, 5e-4)));
#endif // QDK_CHEMISTRY_RUN_LONG_TESTS

/* ==================== Read Guess, c2h4 RKS, pure ===================== */
INSTANTIATE_TEST_CASE_P(RKS_HGP_Rd_c2h4, SCFTest, ::testing::Values(
  std::make_tuple("read_guess", "RKS/B3LYP/def2-svp/Rd", "CPU", "CPU", "GAUXC", 1e-6, 1e-5)));
/* ==================== Read Guess, O UKS, pure ======================== */
INSTANTIATE_TEST_CASE_P(UKS_CPU_Rd_o, SCFTest, ::testing::Values(
  std::make_tuple("read_guess", "UKS/B3LYP/def2-svp/Rd", "CPU", "CPU", "GAUXC", 1e-6, 1e-5)));

/* ==================== GAUXC Tests ===================== */
INSTANTIATE_TEST_CASE_P(RKS_CPU_GAUXC_h2o, SCFTest, ::testing::Values(
  std::make_tuple("h2o_gauxc", "RKS/M06-2X/6-31g*/cart", "CPU", "CPU", "GAUXC", 1e-8, 1e-8),
  std::make_tuple("h2o_gauxc", "RKS/M06-2X/6-31g*/pure", "CPU", "CPU", "GAUXC", 1e-8, 1e-8),
  std::make_tuple("h2o_gauxc", "UKS/M06-2X/6-31g*/pure", "CPU", "CPU", "GAUXC", 1e-8, 1e-8),
  std::make_tuple("h2o_gauxc", "RKS/B3LYP/6-31g*/pure",  "CPU", "CPU", "GAUXC", 1e-8, 1e-8),
  std::make_tuple("h2o_gauxc", "UKS/B3LYP/6-31g*/pure",  "CPU", "CPU", "GAUXC", 1e-8, 1e-8),
  std::make_tuple("h2o_gauxc", "RKS/SVWN/6-31g*/pure",   "CPU", "CPU", "GAUXC", 1e-8, 1e-8),
  std::make_tuple("h2o_gauxc", "UKS/SVWN/6-31g*/pure",   "CPU", "CPU", "GAUXC", 1e-8, 1e-8),
  std::make_tuple("h2o_gauxc", "RKS/M06-2X/6-31g*/cart", "CPU", "SNK", "GAUXC", 1e-7, -1e-8),
  std::make_tuple("h2o_gauxc", "RKS/M06-2X/6-31g*/pure", "CPU", "SNK", "GAUXC", 1e-7, -1e-8),
  std::make_tuple("h2o_gauxc", "UKS/M06-2X/6-31g*/pure", "CPU", "SNK", "GAUXC", 1e-7, -1e-8)
  ));

  // Level shifting Tests
INSTANTIATE_TEST_CASE_P(RKS_LEVEL_SHIFTING, SCFTest, ::testing::Values(
  std::make_tuple("benzene", "RKS/PBE/cc-pvdz", "HGP", "HGP", "GAUXC", 1e-8, 1e0)
  ));

// clang-format on
