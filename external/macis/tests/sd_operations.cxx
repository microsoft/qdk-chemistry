/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <iostream>
#include <macis/sd_operations.hpp>

#include "ut_common.hpp"

TEST_CASE("Slater Det Operations") {
  ROOT_ONLY(MPI_COMM_WORLD);

  SECTION("HF Determinant") {
    SECTION("Canonical Ordering") {
      using wfn_type = macis::wfn_t<32>;
      using wfn_traits = macis::wavefunction_traits<wfn_type>;
      auto hf = wfn_traits::canonical_hf_determinant(4, 2);
      REQUIRE(hf == 0x0003000F);
    }
  }

  SECTION("Occupied / Unoccupied Conversion") {
    SECTION("Default") {
      using wfn_type = macis::wfn_t<64>;
      using wfn_traits = macis::wavefunction_traits<wfn_type>;
      wfn_type state = 0x00000000000000AF;
      std::vector<uint32_t> ref_occ = {0, 1, 2, 3, 5, 7};
      std::vector<uint32_t> ref_vir = {4, 6, 8, 9, 10, 11};

      std::vector<uint32_t> occ, vir;
      wfn_traits::state_to_occ_vir(12, state, occ, vir);

      REQUIRE_THAT(occ, Catch::Matchers::Equals(ref_occ));
      REQUIRE_THAT(vir, Catch::Matchers::Equals(ref_vir));
    }

    SECTION("With Orbs") {}
  }

  SECTION("Singles") {
    SECTION("Single Spin") {
      using wfn_type = macis::wfn_t<64>;
      using wfn_traits = macis::wavefunction_traits<wfn_type>;
      std::vector<wfn_type> ref_singles = {
          0b00011110, 0b00101110, 0b01001110, 0b10001110,
          0b00011101, 0b00101101, 0b01001101, 0b10001101,
          0b00011011, 0b00101011, 0b01001011, 0b10001011,
          0b00010111, 0b00100111, 0b01000111, 0b10000111};

      std::vector<wfn_type> singles;
      auto state = wfn_traits::canonical_hf_determinant(4, 0);  // all alpha
      macis::generate_singles(8, state, singles);

      auto cmp = macis::bitset_less_comparator<64>{};
      std::sort(singles.begin(), singles.end(), cmp);
      std::sort(ref_singles.begin(), ref_singles.end(), cmp);

      REQUIRE_THAT(singles, Catch::Matchers::Equals(ref_singles));
    }

    SECTION("Both Spin") {
      using wfn_type = macis::wfn_t<64>;
      using wfn_traits = macis::wavefunction_traits<wfn_type>;
      std::vector<wfn_type> ref_singles = {
          0x0F0000001E, 0x0F0000001D, 0x0F0000001B, 0x0F00000017, 0x0F0000002E,
          0x0F0000002D, 0x0F0000002B, 0x0F00000027, 0x0F0000004E, 0x0F0000004D,
          0x0F0000004B, 0x0F00000047, 0x0F0000008E, 0x0F0000008D, 0x0F0000008B,
          0x0F00000087, 0x1E0000000F, 0x1D0000000F, 0x1B0000000F, 0x170000000F,
          0x2E0000000F, 0x2D0000000F, 0x2B0000000F, 0x270000000F, 0x4E0000000F,
          0x4D0000000F, 0x4B0000000F, 0x470000000F, 0x8E0000000F, 0x8D0000000F,
          0x8B0000000f, 0x870000000f};

      auto state = wfn_traits::canonical_hf_determinant(4, 4);
      std::vector<wfn_type> singles;
      macis::generate_singles_spin(8, state, singles);

      auto cmp = macis::bitset_less_comparator<64>{};
      std::sort(singles.begin(), singles.end(), cmp);
      std::sort(ref_singles.begin(), ref_singles.end(), cmp);

      REQUIRE_THAT(singles, Catch::Matchers::Equals(ref_singles));
    }
  }

  SECTION("Doubles") {
    SECTION("Single Spin") {
      using wfn_type = macis::wfn_t<64>;
      using wfn_traits = macis::wavefunction_traits<wfn_type>;
      std::vector<wfn_type> ref_doubles = {
          0b00111100, 0b01011100, 0b10011100, 0b01101100, 0b10101100,
          0b11001100, 0b00111010, 0b01011010, 0b10011010, 0b01101010,
          0b10101010, 0b11001010, 0b00111001, 0b01011001, 0b10011001,
          0b01101001, 0b10101001, 0b11001001, 0b00110110, 0b01010110,
          0b10010110, 0b01100110, 0b10100110, 0b11000110, 0b00110101,
          0b01010101, 0b10010101, 0b01100101, 0b10100101, 0b11000101,
          0b00110011, 0b01010011, 0b10010011, 0b01100011, 0b10100011,
          0b11000011};

      std::vector<wfn_type> doubles;
      auto state = wfn_traits::canonical_hf_determinant(4, 0);  // all alpha
      macis::generate_doubles(8, state, doubles);

      auto cmp = macis::bitset_less_comparator<64>{};
      std::sort(doubles.begin(), doubles.end(), cmp);
      std::sort(ref_doubles.begin(), ref_doubles.end(), cmp);

      REQUIRE_THAT(doubles, Catch::Matchers::Equals(ref_doubles));
    }

    SECTION("Both Spins") {
      using wfn_type = macis::wfn_t<64>;
      using wfn_traits = macis::wavefunction_traits<wfn_type>;
      std::vector<wfn_type> ref_doubles = {
          0x0F0000003C, 0x0F0000003A, 0x0F00000036, 0x0F0000005C, 0x0F0000005A,
          0x0F00000056, 0x0F0000009C, 0x0F0000009A, 0x0F00000096, 0x0F00000039,
          0x0F00000035, 0x0F00000059, 0x0F00000055, 0x0F00000099, 0x0F00000095,
          0x0F00000033, 0x0F00000053, 0x0F00000093, 0x0F0000006C, 0x0F0000006A,
          0x0F00000066, 0x0F000000AC, 0x0F000000AA, 0x0F000000A6, 0x0F00000069,
          0x0F00000065, 0x0F000000A9, 0x0F000000A5, 0x0F00000063, 0x0F000000A3,
          0x0F000000CC, 0x0F000000CA, 0x0F000000C6, 0x0F000000C9, 0x0F000000C5,
          0x0F000000C3, 0x3C0000000F, 0x3A0000000F, 0x360000000F, 0x5C0000000F,
          0x5A0000000F, 0x560000000F, 0x9C0000000F, 0x9A0000000F, 0x960000000F,
          0x390000000F, 0x350000000F, 0x590000000F, 0x550000000F, 0x990000000F,
          0x950000000F, 0x330000000F, 0x530000000F, 0x930000000F, 0x6C0000000F,
          0x6A0000000F, 0x660000000F, 0xAC0000000F, 0xAA0000000F, 0xA60000000F,
          0x690000000F, 0x650000000F, 0xA90000000F, 0xA50000000F, 0x630000000F,
          0xA30000000F, 0xCC0000000F, 0xCA0000000F, 0xC60000000F, 0xC90000000F,
          0xC50000000F, 0xC30000000F, 0x1E0000001E, 0x1D0000001E, 0x1B0000001E,
          0x170000001E, 0x2E0000001E, 0x2D0000001E, 0x2B0000001E, 0x270000001E,
          0x4E0000001E, 0x4D0000001E, 0x4B0000001E, 0x470000001E, 0x8E0000001E,
          0x8D0000001E, 0x8B0000001E, 0x870000001E, 0x1E0000001D, 0x1D0000001D,
          0x1B0000001D, 0x170000001D, 0x2E0000001D, 0x2D0000001D, 0x2B0000001D,
          0x270000001D, 0x4E0000001D, 0x4D0000001D, 0x4B0000001D, 0x470000001D,
          0x8E0000001D, 0x8D0000001D, 0x8B0000001D, 0x870000001D, 0x1E0000001B,
          0x1D0000001B, 0x1B0000001B, 0x170000001B, 0x2E0000001B, 0x2D0000001B,
          0x2B0000001B, 0x270000001B, 0x4E0000001B, 0x4D0000001B, 0x4B0000001B,
          0x470000001B, 0x8E0000001B, 0x8D0000001B, 0x8B0000001B, 0x870000001B,
          0x1E00000017, 0x1D00000017, 0x1B00000017, 0x1700000017, 0x2E00000017,
          0x2D00000017, 0x2B00000017, 0x2700000017, 0x4E00000017, 0x4D00000017,
          0x4B00000017, 0x4700000017, 0x8E00000017, 0x8D00000017, 0x8B00000017,
          0x8700000017, 0x1E0000002E, 0x1D0000002E, 0x1B0000002E, 0x170000002E,
          0x2E0000002E, 0x2D0000002E, 0x2B0000002E, 0x270000002E, 0x4E0000002E,
          0x4D0000002E, 0x4B0000002E, 0x470000002E, 0x8E0000002E, 0x8D0000002E,
          0x8B0000002E, 0x870000002E, 0x1E0000002D, 0x1D0000002D, 0x1B0000002D,
          0x170000002D, 0x2E0000002D, 0x2D0000002D, 0x2B0000002D, 0x270000002D,
          0x4E0000002D, 0x4D0000002D, 0x4B0000002D, 0x470000002D, 0x8E0000002D,
          0x8D0000002D, 0x8B0000002D, 0x870000002D, 0x1E0000002B, 0x1D0000002B,
          0x1B0000002B, 0x170000002B, 0x2E0000002B, 0x2D0000002B, 0x2B0000002B,
          0x270000002B, 0x4E0000002B, 0x4D0000002B, 0x4B0000002B, 0x470000002B,
          0x8E0000002B, 0x8D0000002B, 0x8B0000002B, 0x870000002B, 0x1E00000027,
          0x1D00000027, 0x1B00000027, 0x1700000027, 0x2E00000027, 0x2D00000027,
          0x2B00000027, 0x2700000027, 0x4E00000027, 0x4D00000027, 0x4B00000027,
          0x4700000027, 0x8E00000027, 0x8D00000027, 0x8B00000027, 0x8700000027,
          0x1E0000004E, 0x1D0000004E, 0x1B0000004E, 0x170000004E, 0x2E0000004E,
          0x2D0000004E, 0x2B0000004E, 0x270000004E, 0x4E0000004E, 0x4D0000004E,
          0x4B0000004E, 0x470000004E, 0x8E0000004E, 0x8D0000004E, 0x8B0000004E,
          0x870000004E, 0x1E0000004D, 0x1D0000004D, 0x1B0000004D, 0x170000004D,
          0x2E0000004D, 0x2D0000004D, 0x2B0000004D, 0x270000004D, 0x4E0000004D,
          0x4D0000004D, 0x4B0000004D, 0x470000004D, 0x8E0000004D, 0x8D0000004D,
          0x8B0000004D, 0x870000004D, 0x1E0000004B, 0x1D0000004B, 0x1B0000004B,
          0x170000004B, 0x2E0000004B, 0x2D0000004B, 0x2B0000004B, 0x270000004B,
          0x4E0000004B, 0x4D0000004B, 0x4B0000004B, 0x470000004B, 0x8E0000004B,
          0x8D0000004B, 0x8B0000004B, 0x870000004B, 0x1E00000047, 0x1D00000047,
          0x1B00000047, 0x1700000047, 0x2E00000047, 0x2D00000047, 0x2B00000047,
          0x2700000047, 0x4E00000047, 0x4D00000047, 0x4B00000047, 0x4700000047,
          0x8E00000047, 0x8D00000047, 0x8B00000047, 0x8700000047, 0x1E0000008E,
          0x1D0000008E, 0x1B0000008E, 0x170000008E, 0x2E0000008E, 0x2D0000008E,
          0x2B0000008E, 0x270000008E, 0x4E0000008E, 0x4D0000008E, 0x4B0000008E,
          0x470000008E, 0x8E0000008E, 0x8D0000008E, 0x8B0000008E, 0x870000008E,
          0x1E0000008D, 0x1D0000008D, 0x1B0000008D, 0x170000008D, 0x2E0000008D,
          0x2D0000008D, 0x2B0000008D, 0x270000008D, 0x4E0000008D, 0x4D0000008D,
          0x4B0000008D, 0x470000008D, 0x8E0000008D, 0x8D0000008D, 0x8B0000008D,
          0x870000008D, 0x1E0000008B, 0x1D0000008B, 0x1B0000008B, 0x170000008B,
          0x2E0000008B, 0x2D0000008B, 0x2B0000008B, 0x270000008B, 0x4E0000008B,
          0x4D0000008B, 0x4B0000008B, 0x470000008B, 0x8E0000008B, 0x8D0000008B,
          0x8B0000008B, 0x870000008B, 0x1E00000087, 0x1D00000087, 0x1B00000087,
          0x1700000087, 0x2E00000087, 0x2D00000087, 0x2B00000087, 0x2700000087,
          0x4E00000087, 0x4D00000087, 0x4B00000087, 0x4700000087, 0x8E00000087,
          0x8D00000087, 0x8B00000087, 0x8700000087};
      auto state = wfn_traits::canonical_hf_determinant(4, 4);
      std::vector<wfn_type> singles, doubles;
      macis::generate_singles_doubles_spin(8, state, singles, doubles);

      auto cmp = macis::bitset_less_comparator<64>{};
      std::sort(doubles.begin(), doubles.end(), cmp);
      std::sort(ref_doubles.begin(), ref_doubles.end(), cmp);

      REQUIRE_THAT(doubles, Catch::Matchers::Equals(ref_doubles));
    }
  }

  SECTION("CIS") {
    using wfn_type = macis::wfn_t<64>;
    using wfn_traits = macis::wavefunction_traits<wfn_type>;
    auto state = wfn_traits::canonical_hf_determinant(4, 4);
    std::vector<wfn_type> singles;
    macis::generate_singles_spin(8, state, singles);
    singles.push_back(state);
    auto cis = macis::generate_cis_hilbert_space(8, state);

    auto cmp = macis::bitset_less_comparator<64>{};
    std::sort(singles.begin(), singles.end(), cmp);
    std::sort(cis.begin(), cis.end(), cmp);
    REQUIRE_THAT(singles, Catch::Matchers::Equals(cis));
  }

  SECTION("CISD") {
    using wfn_type = macis::wfn_t<64>;
    using wfn_traits = macis::wavefunction_traits<wfn_type>;
    auto state = wfn_traits::canonical_hf_determinant(4, 4);
    std::vector<wfn_type> singles, doubles;
    macis::generate_singles_doubles_spin(8, state, singles, doubles);
    singles.insert(singles.end(), doubles.begin(), doubles.end());
    singles.push_back(state);
    auto cisd = macis::generate_cisd_hilbert_space(8, state);

    auto cmp = macis::bitset_less_comparator<64>{};
    std::sort(singles.begin(), singles.end(), cmp);
    std::sort(cisd.begin(), cisd.end(), cmp);
    REQUIRE_THAT(singles, Catch::Matchers::Equals(cisd));
  }

  SECTION("FCI") {
    SECTION("Single Spin") {
      std::vector<macis::wfn_t<64>> ref_combs = {0x3, 0x5, 0x9, 0x6, 0xA, 0xC};
      auto combs = macis::generate_combs<macis::wfn_t<64>>(4, 2);
      REQUIRE_THAT(combs, Catch::Matchers::Equals(ref_combs));
    }

    SECTION("Both Spins") {
      std::vector<macis::wfn_t<64>> ref_dets = {
          0x300000003, 0x500000003, 0x900000003, 0x600000003, 0xA00000003,
          0xC00000003, 0x300000005, 0x500000005, 0x900000005, 0x600000005,
          0xA00000005, 0xC00000005, 0x300000009, 0x500000009, 0x900000009,
          0x600000009, 0xA00000009, 0xC00000009, 0x300000006, 0x500000006,
          0x900000006, 0x600000006, 0xA00000006, 0xC00000006, 0x30000000A,
          0x50000000A, 0x90000000A, 0x60000000A, 0xA0000000A, 0xC0000000A,
          0x30000000C, 0x50000000C, 0x90000000C, 0x60000000C, 0xA0000000C,
          0xC0000000C};

      auto dets = macis::generate_hilbert_space<macis::wfn_t<64>>(4, 2, 2);
      REQUIRE_THAT(dets, Catch::Matchers::Equals(ref_dets));
    }
  }

  SECTION("String Conversions") {
    using wfn_type = macis::wfn_t<64>;
    using wfn_traits = macis::wavefunction_traits<wfn_type>;
    auto state =
        wfn_traits::canonical_hf_determinant(2, 2).flip(3).flip(4 + 32);

    SECTION("To String") {
      auto str = macis::to_canonical_string(state);
      REQUIRE_THAT(str,
                   Catch::Matchers::Equals("220ud000000000000000000000000000"));
    }

    SECTION("From String") {
      SECTION("Full String") {
        std::string str = "220ud000000000000000000000000000";
        auto det = macis::from_canonical_string<wfn_type>(str);
        REQUIRE(det == state);
      }

      SECTION("Short String") {
        std::string str = "220ud";
        auto det = macis::from_canonical_string<wfn_type>(str);
        REQUIRE(det == state);
      }
    }
  }
}
