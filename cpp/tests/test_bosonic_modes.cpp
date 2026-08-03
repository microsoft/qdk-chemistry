// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <qdk/chemistry/data/bosonic_modes.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qdk::chemistry::data;

namespace {

std::filesystem::path temp_path(const std::string& name) {
  return std::filesystem::temp_directory_path() / name;
}

}  // namespace

TEST(BosonicModes, ConstructionExposesPerModeDimension) {
  BosonicModes modes(4, 8);
  EXPECT_EQ(modes.num_modes(), 4u);
  EXPECT_EQ(modes.get_num_molecular_orbitals(), 4u);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(modes.mode_dimension(i), 8u);
    EXPECT_EQ(modes.max_occupation(i), 7u);
  }
  ASSERT_TRUE(modes.uniform_dimension().has_value());
  EXPECT_EQ(*modes.uniform_dimension(), 8u);
  EXPECT_TRUE(modes.has_power_of_two_dimensions());
  EXPECT_EQ(modes.fock_space_dimension(), 8u * 8u * 8u * 8u);
  // The cutoff is attributed per mode: the stored truth is one entry per mode,
  // even though phase-1 public construction only ever makes them equal.
  EXPECT_EQ(modes.mode_dimensions(),
            (std::vector<std::size_t>{8u, 8u, 8u, 8u}));
}

TEST(BosonicModes, IsAnOrbitalsSoTheSectorSeamKeepsWorking) {
  std::shared_ptr<Orbitals> basis = std::make_shared<BosonicModes>(3, 4);
  EXPECT_EQ(basis->num_modes(), 3u);
  const auto* as_bosonic = dynamic_cast<const BosonicModes*>(basis.get());
  ASSERT_NE(as_bosonic, nullptr);
  EXPECT_EQ(as_bosonic->mode_dimension(2), 4u);
  // ModelOrbitals is the intermediate base, so model-orbital behaviour is
  // inherited unchanged.
  EXPECT_NE(dynamic_cast<const ModelOrbitals*>(basis.get()), nullptr);
}

TEST(BosonicModes, RejectsInvalidArguments) {
  EXPECT_THROW(BosonicModes(0, 4), std::invalid_argument);
  EXPECT_THROW(BosonicModes(2, 1), std::invalid_argument);
  EXPECT_THROW(BosonicModes(2, 0), std::invalid_argument);

  BosonicModes modes(2, 4);
  EXPECT_THROW(modes.mode_dimension(2), std::out_of_range);
  EXPECT_THROW(modes.max_occupation(7), std::out_of_range);
}

TEST(BosonicModes, StoresTheCutoffVerbatimAndPadsOnlyOnRequest) {
  // No silent rounding: d = 3 and d = 4 have genuinely different spectra.
  BosonicModes exact(2, 3);
  EXPECT_EQ(exact.mode_dimension(0), 3u);
  EXPECT_FALSE(exact.has_power_of_two_dimensions());

  EXPECT_EQ(BosonicModes::padded_dimension(1), 2u);
  EXPECT_EQ(BosonicModes::padded_dimension(2), 2u);
  EXPECT_EQ(BosonicModes::padded_dimension(3), 4u);
  EXPECT_EQ(BosonicModes::padded_dimension(4), 4u);
  EXPECT_EQ(BosonicModes::padded_dimension(5), 8u);
  EXPECT_EQ(BosonicModes::padded_dimension(9), 16u);

  auto padded = BosonicModes::padded_to_power_of_two(2, 5);
  ASSERT_NE(padded, nullptr);
  EXPECT_EQ(padded->mode_dimension(0), 8u);
  EXPECT_EQ(padded->mode_dimension(1), 8u);
  EXPECT_TRUE(padded->has_power_of_two_dimensions());

  // The instance overload pads an existing basis, leaving the original alone.
  auto padded_instance = exact.with_padded_dimensions();
  ASSERT_NE(padded_instance, nullptr);
  EXPECT_EQ(padded_instance->mode_dimension(0), 4u);
  EXPECT_EQ(padded_instance->mode_dimension(1), 4u);
  EXPECT_EQ(exact.mode_dimension(0), 3u);
}

TEST(BosonicModes, JsonRoundTripThroughOrbitalsDispatch) {
  BosonicModes modes(3, 8);
  const auto json = modes.to_json();
  EXPECT_EQ(json.at("type").get<std::string>(), "BosonicModes");
  // The cutoff is serialized per mode from day one, so a heterogeneous basis
  // round-trips later with no schema change and no version bump.
  EXPECT_EQ(json.at("mode_dimensions").get<std::vector<std::size_t>>(),
            (std::vector<std::size_t>{8u, 8u, 8u}));

  // Direct deserialization.
  auto direct = BosonicModes::from_json(json);
  ASSERT_NE(direct, nullptr);
  EXPECT_EQ(direct->num_modes(), 3u);
  EXPECT_EQ(direct->mode_dimension(1), 8u);

  // Polymorphic dispatch through the Orbitals factory.
  auto loaded = Orbitals::from_json(json);
  ASSERT_NE(loaded, nullptr);
  const auto* as_bosonic = dynamic_cast<const BosonicModes*>(loaded.get());
  ASSERT_NE(as_bosonic, nullptr);
  EXPECT_EQ(as_bosonic->num_modes(), 3u);
  EXPECT_EQ(as_bosonic->mode_dimension(0), 8u);
  EXPECT_EQ(as_bosonic->content_hash(), modes.content_hash());
}

TEST(BosonicModes, HeterogeneousDimensionsRoundTripThroughJson) {
  // Phase 1 offers no public heterogeneous constructor, but the stored
  // representation and the serialized schema are already per-mode, so a
  // payload with differing dimensions must load correctly today.  This is the
  // guarantee that lets per-mode cutoffs land later as a pure addition.
  BosonicModes uniform(3, 4);
  auto json = uniform.to_json();
  json["mode_dimensions"] = std::vector<std::size_t>{2u, 4u, 3u};

  auto loaded = BosonicModes::from_json(json);
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->mode_dimension(0), 2u);
  EXPECT_EQ(loaded->mode_dimension(1), 4u);
  EXPECT_EQ(loaded->mode_dimension(2), 3u);
  EXPECT_EQ(loaded->max_occupation(2), 2u);
  EXPECT_FALSE(loaded->uniform_dimension().has_value());
  EXPECT_FALSE(loaded->has_power_of_two_dimensions());
  EXPECT_EQ(loaded->fock_space_dimension(), 2u * 4u * 3u);
  EXPECT_NE(loaded->content_hash(), uniform.content_hash());

  // Padding an inhomogeneous basis pads every mode independently.
  auto padded = loaded->with_padded_dimensions();
  EXPECT_EQ(padded->mode_dimensions(), (std::vector<std::size_t>{2u, 4u, 4u}));
  EXPECT_TRUE(padded->has_power_of_two_dimensions());
  EXPECT_FALSE(padded->uniform_dimension().has_value());

  // The schema is an array, always. A scalar payload is a hard error rather
  // than a silent broadcast: this class is new, so no scalar format ever
  // shipped and tolerating one would only invite ambiguity.
  auto scalar_json = uniform.to_json();
  scalar_json["mode_dimensions"] = 8u;
  EXPECT_THROW(BosonicModes::from_json(scalar_json), std::runtime_error);

  // An array whose length disagrees with the mode count is rejected too.
  auto short_json = uniform.to_json();
  short_json["mode_dimensions"] = std::vector<std::size_t>{8u, 8u};
  EXPECT_THROW(BosonicModes::from_json(short_json), std::runtime_error);

  // A payload with no cutoff at all is an error, never a silent default.
  auto missing_json = uniform.to_json();
  missing_json.erase("mode_dimensions");
  EXPECT_THROW(BosonicModes::from_json(missing_json), std::runtime_error);
}

TEST(BosonicModes, ModelOrbitalsJsonStillDeserializesAsModelOrbitals) {
  // Backward compatibility: an old file has no "BosonicModes" tag.
  ModelOrbitals model(5);
  auto loaded = Orbitals::from_json(model.to_json());
  ASSERT_NE(loaded, nullptr);
  EXPECT_NE(dynamic_cast<const ModelOrbitals*>(loaded.get()), nullptr);
  EXPECT_EQ(dynamic_cast<const BosonicModes*>(loaded.get()), nullptr);
  EXPECT_EQ(loaded->num_modes(), 5u);
}

TEST(BosonicModes, Hdf5RoundTripThroughOrbitalsDispatch) {
  const auto path = temp_path("qdk_bosonic_modes_roundtrip.h5");
  {
    H5::H5File file(path.string(), H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    BosonicModes modes(2, 16);
    modes.to_hdf5(root);
  }
  {
    H5::H5File file(path.string(), H5F_ACC_RDONLY);
    H5::Group root = file.openGroup("/");
    auto loaded = Orbitals::from_hdf5(root);
    ASSERT_NE(loaded, nullptr);
    const auto* as_bosonic = dynamic_cast<const BosonicModes*>(loaded.get());
    ASSERT_NE(as_bosonic, nullptr);
    EXPECT_EQ(as_bosonic->num_modes(), 2u);
    EXPECT_EQ(as_bosonic->mode_dimension(0), 16u);
    EXPECT_EQ(as_bosonic->max_occupation(1), 15u);
  }
  std::filesystem::remove(path);
}

TEST(BosonicModes, ContentHashSeparatesDifferentCutoffs) {
  BosonicModes a(3, 4);
  BosonicModes b(3, 8);
  BosonicModes c(3, 4);
  EXPECT_EQ(a.content_hash(), c.content_hash());
  EXPECT_NE(a.content_hash(), b.content_hash());
  EXPECT_NE(a.content_hash(), ModelOrbitals(3).content_hash());
}

TEST(BosonicModes, SummaryMentionsTheCutoff) {
  BosonicModes modes(2, 4);
  const auto summary = modes.get_summary();
  EXPECT_NE(summary.find("BosonicModes"), std::string::npos);
  EXPECT_NE(summary.find("Local dimension d: 4"), std::string::npos);
  EXPECT_EQ(modes.get_data_type_name(), "bosonic_modes");
}
