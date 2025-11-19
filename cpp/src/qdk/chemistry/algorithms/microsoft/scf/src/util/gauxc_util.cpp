// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/util/gauxc_util.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace qdk::chemistry::scf::gauxc_util {

/**
 * @brief Converts string to lowercase for case-insensitive comparisons
 *
 * Utility function to convert a string to lowercase using standard library
 * facilities. This enables case-insensitive string matching for functional
 * names and configuration options.
 *
 * @param s Input string to convert
 * @return New string with all characters converted to lowercase
 *
 * @note Uses std::tolower which handles ASCII characters properly
 */
static std::string to_lower(const std::string& s) {
  std::string out = s;
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return out;
}

GauXC::AtomicGridSizeDefault grid_size_from_string(const std::string& str) {
  std::string s = to_lower(str);
  if (s == "fine") return GauXC::AtomicGridSizeDefault::FineGrid;
  if (s == "ultrafine") return GauXC::AtomicGridSizeDefault::UltraFineGrid;
  if (s == "superfine") return GauXC::AtomicGridSizeDefault::SuperFineGrid;
  if (s == "gm3") return GauXC::AtomicGridSizeDefault::GM3;
  if (s == "gm5") return GauXC::AtomicGridSizeDefault::GM5;
  throw std::invalid_argument("Unknown GauXC::AtomicGridSizeDefault: " + str);
}
std::string to_string(GauXC::AtomicGridSizeDefault type) {
  switch (type) {
    case GauXC::AtomicGridSizeDefault::FineGrid:
      return "FINE";
    case GauXC::AtomicGridSizeDefault::UltraFineGrid:
      return "ULTRAFINE";
    case GauXC::AtomicGridSizeDefault::SuperFineGrid:
      return "SUPERFINE";
    case GauXC::AtomicGridSizeDefault::GM3:
      return "GM3";
    case GauXC::AtomicGridSizeDefault::GM5:
      return "GM5";
    default:
      return "UNKNOWN";
  }
}

GauXC::RadialQuad radial_quad_from_string(const std::string& str) {
  std::string s = to_lower(str);
  if (s == "mura-knowles" || s == "muraknowles")
    return GauXC::RadialQuad::MuraKnowles;
  if (s == "treutler-ahlrichs" || s == "treutlerahlrichs")
    return GauXC::RadialQuad::TreutlerAhlrichs;
  if (s == "murray-handy-laming" || s == "murrayhandylaming")
    return GauXC::RadialQuad::MurrayHandyLaming;
  if (s == "becke" || s == "becke1988") return GauXC::RadialQuad::Becke;
  throw std::invalid_argument("Unknown GauXC::RadialQuad: " + str);
}
std::string to_string(GauXC::RadialQuad type) {
  switch (type) {
    case GauXC::RadialQuad::MuraKnowles:
      return "MURAKNOWLES";
    case GauXC::RadialQuad::TreutlerAhlrichs:
      return "TREUTLERAHLRICHS";
    case GauXC::RadialQuad::MurrayHandyLaming:
      return "MURRAYHANDYLAMING";
    case GauXC::RadialQuad::Becke:
      return "BECKE";
    default:
      return "UNKNOWN";
  }
}

GauXC::PruningScheme prune_method_from_string(const std::string& str) {
  std::string s = to_lower(str);
  if (s == "unpruned") return GauXC::PruningScheme::Unpruned;
  if (s == "robust") return GauXC::PruningScheme::Robust;
  if (s == "treutler") return GauXC::PruningScheme::Treutler;
  throw std::invalid_argument("Unknown GauXC::PruningScheme: " + str);
}
std::string to_string(GauXC::PruningScheme type) {
  switch (type) {
    case GauXC::PruningScheme::Unpruned:
      return "UNPRUNED";
    case GauXC::PruningScheme::Robust:
      return "ROBUST";
    case GauXC::PruningScheme::Treutler:
      return "TREUTLER";
    default:
      return "UNKNOWN";
  }
}

GauXC::ExecutionSpace execution_space_from_string(const std::string& str) {
  std::string s = to_lower(str);
  if (s == "host" || s == "cpu") return GauXC::ExecutionSpace::Host;
#if defined(GAUXC_HAS_DEVICE) && defined(QDK_CHEMISTRY_ENABLE_GPU)
  if (s == "device" || s == "gpu") return GauXC::ExecutionSpace::Device;
#endif
  throw std::invalid_argument("Unknown GauXC::ExecutionSpace: " + str);
}
std::string to_string(GauXC::ExecutionSpace type) {
  switch (type) {
    case GauXC::ExecutionSpace::Host:
      return "HOST";
#if defined(GAUXC_HAS_DEVICE) && defined(QDK_CHEMISTRY_ENABLE_GPU)
    case GauXC::ExecutionSpace::Device:
      return "DEVICE";
#endif
    default:
      return "UNKNOWN";
  }
}

}  // namespace qdk::chemistry::scf::gauxc_util
