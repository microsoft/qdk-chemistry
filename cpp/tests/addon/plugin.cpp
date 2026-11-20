// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <memory>
#include <qdk/chemistry/algorithms/jk_builder.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace qdk_test_addon {

using namespace qdk::chemistry;

namespace detail {

class TestJKBuilder : public algorithms::JKBuilder {
 public:
  explicit TestJKBuilder(std::string name) : _name(std::move(name)) {}
  std::string name() const override { return _name; }

 protected:
  std::pair<std::shared_ptr<const data::SymmetryBlockedTensor<2>>,
            std::shared_ptr<const data::SymmetryBlockedTensor<2>>>
  _run_impl(
      std::shared_ptr<data::BasisSet>,
      std::shared_ptr<const data::TwoBodyKernel>,
      std::shared_ptr<const data::TwoBodyKernel>,
      std::shared_ptr<const data::SymmetryBlockedTensor<2>>) const override {
    throw std::logic_error("The test add-on J/K builder does not build J/K");
  }

 private:
  std::string _name;
};

}  // namespace detail

// Registers a J/K builder under the name of the built-in one
void register_duplicate() {
  algorithms::JKBuilderFactory::register_instance(
      [] { return std::make_unique<detail::TestJKBuilder>("qdk"); });
}

void throw_non_standard() { throw 42; }

// Registers the "test_addon" J/K builder if the built-in SCF solver is
// registered
void register_plugin() {
  if (!algorithms::ScfSolverFactory::has("qdk")) {
    throw std::runtime_error("The built-in SCF solver is not registered");
  }
  algorithms::JKBuilderFactory::register_instance(
      [] { return std::make_unique<detail::TestJKBuilder>("test_addon"); });
}

}  // namespace qdk_test_addon
