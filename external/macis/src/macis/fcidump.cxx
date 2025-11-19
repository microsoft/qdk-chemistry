/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <macis/util/fcidump.hpp>
#include <regex>
#include <sstream>
#include <string>

std::vector<std::string> split(const std::string str,
                               const std::string regex_str) {
  std::regex regexz(regex_str);
  std::vector<std::string> list(
      std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
      std::sregex_token_iterator());
  std::vector<std::string> clean_list;
  std::copy_if(list.begin(), list.end(), std::back_inserter(clean_list),
               [](auto& s) { return s.size() > 0; });
  return clean_list;
}

bool is_float(const std::string& str) {
  return std::any_of(str.begin(), str.end(),
                     [](auto c) { return std::isalpha(c) or c == '.'; });
}

auto fcidump_line(const std::vector<std::string>& tokens) {
  auto idx_first = is_float(tokens.back());
  auto int_first = is_float(tokens.front());

  if (idx_first and int_first) throw std::runtime_error("Invalid FCIDUMP Line");

  int32_t p, q, r, s;
  double integral;

  if (idx_first) {
    p = std::stoi(tokens[0]);
    q = std::stoi(tokens[1]);
    r = std::stoi(tokens[2]);
    s = std::stoi(tokens[3]);
    integral = std::stod(tokens[4]);
  } else {
    p = std::stoi(tokens[1]);
    q = std::stoi(tokens[2]);
    r = std::stoi(tokens[3]);
    s = std::stoi(tokens[4]);
    integral = std::stod(tokens[0]);
  }

  if (p < 0 or q < 0 or r < 0 or s < 0)
    throw std::runtime_error("Invalid Orb Idx");

  return std::make_tuple(p, q, r, s, integral);
}

enum FCIDumpFormat { IntegralFirst, IndicesFirst, Invalid };

FCIDumpFormat detect_fcidump_format(const std::string& line) {
  // Skip empty lines and lines with only whitespace
  if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) {
    return FCIDumpFormat::Invalid;
  }

  // Tokenize the line
  auto tokens = split(line, "\\s+");
  if (tokens.size() != 5) {
    return FCIDumpFormat::Invalid;
  }

  // Check if first token is a float (contains '.' or scientific notation)
  bool first_is_float = is_float(tokens[0]);
  // Check if last token is a float
  bool last_is_float = is_float(tokens[4]);

  // If first token is float and last is not, then format is: integral p q r s
  if (first_is_float && !last_is_float) {
    return FCIDumpFormat::IntegralFirst;
  }
  // If last token is float and first is not, then format is: p q r s integral
  else if (!first_is_float && last_is_float) {
    return FCIDumpFormat::IndicesFirst;
  }
  // If both or neither are floats, we need to be more careful
  else {
    // Try to parse as integers - if the first 4 tokens can be parsed as
    // valid orbital indices (positive integers), assume indices first format
    try {
      for (int i = 0; i < 4; i++) {
        int idx = std::stoi(tokens[i]);
        if (idx < 0) {
          // Negative indices suggest this might be integral first format
          return FCIDumpFormat::IntegralFirst;
        }
      }
      // All first 4 tokens are non-negative integers, assume indices first
      return FCIDumpFormat::IndicesFirst;
    } catch (const std::exception&) {
      // If parsing first 4 as integers fails, assume integral first
      return FCIDumpFormat::IntegralFirst;
    }
  }
}

auto fcidump_line_integral_first(const std::string& line) {
  int32_t p, q, r, s;
  double integral;
  int parsed =
      sscanf(line.c_str(), "%lf %d %d %d %d", &integral, &p, &q, &r, &s);
  return std::make_tuple(bool(parsed == 5), p, q, r, s, integral);
}

auto fcidump_line_indices_first(const std::string& line) {
  int32_t p, q, r, s;
  double integral;
  int parsed =
      sscanf(line.c_str(), "%d %d %d %d %lf", &p, &q, &r, &s, &integral);
  return std::make_tuple(bool(parsed == 5), p, q, r, s, integral);
}

auto fcidump_line(FCIDumpFormat format, std::string& line) {
  switch (format) {
    case FCIDumpFormat::IntegralFirst:
      return fcidump_line_integral_first(line);
    case FCIDumpFormat::IndicesFirst:
      return fcidump_line_indices_first(line);
    case FCIDumpFormat::Invalid:
    default:
      return std::make_tuple(false, 0, 0, 0, 0, 0.0);
  }
}

enum LineClassification { Core, OneBody, TwoBody };

LineClassification line_classification(int p, int q, int r, int s) {
  if (!(p or q or r or s))
    return LineClassification::Core;
  else if (p and q and r and s)
    return LineClassification::TwoBody;
  else
    return LineClassification::OneBody;
}

namespace macis {

FCIDumpHeader fcidump_read_header(std::string fname) {
  std::ifstream file(fname);
  std::string line;
  FCIDumpHeader header;

  bool in_header = false;
  std::string header_content;

  while (std::getline(file, line)) {
    // Check for start of header
    if (line.find("&FCI") != std::string::npos) {
      in_header = true;
    }

    if (in_header) {
      header_content += line + " ";

      // Check for end of header
      if (line.find("&END") != std::string::npos) {
        break;
      }
    }
  }

  if (header_content.empty()) {
    throw std::runtime_error("No FCIDUMP header found");
  }

  // Parse NORB
  std::regex norb_regex(R"(NORB\s*=\s*(\d+))");
  std::smatch match;
  if (std::regex_search(header_content, match, norb_regex)) {
    header.norb = std::stoul(match[1].str());
  }

  // Parse NELEC
  std::regex nelec_regex(R"(NELEC\s*=\s*(\d+))");
  if (std::regex_search(header_content, match, nelec_regex)) {
    header.nelec = std::stoul(match[1].str());
  }

  // Parse MS2
  std::regex ms2_regex(R"(MS2\s*=\s*(-?\d+))");
  if (std::regex_search(header_content, match, ms2_regex)) {
    header.ms2 = std::stoi(match[1].str());
  }

  // Parse ISYM
  std::regex isym_regex(R"(ISYM\s*=\s*(\d+))");
  if (std::regex_search(header_content, match, isym_regex)) {
    header.isym = std::stoi(match[1].str());
  }

  // Parse ORBSYM
  std::regex orbsym_regex(R"(ORBSYM\s*=\s*([\d,\s]+))");
  if (std::regex_search(header_content, match, orbsym_regex)) {
    std::string orbsym_str = match[1].str();
    std::regex num_regex(R"(\d+)");
    std::sregex_iterator iter(orbsym_str.begin(), orbsym_str.end(), num_regex);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
      header.orbsym.push_back(std::stoi(iter->str()));
    }
  }

  return header;
}

uint32_t read_fcidump_norb(std::string fname) {
  FCIDumpHeader header = fcidump_read_header(fname);
  if (header.norb == 0) {
    throw std::runtime_error("NORB not found or is zero in FCIDUMP header");
  }
  return header.norb;
}

double read_fcidump_core(std::string fname) {
  // Read entire file into memory
  std::string content;
  {
    std::ifstream file(fname);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + fname);
    }
    content = std::string((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
  }

  // Parse line by line manually
  const char* ptr = content.c_str();
  const char* end = ptr + content.length();
  std::string line;
  bool header_passed = false;
  bool format_detected = false;
  auto format = FCIDumpFormat::Invalid;

  while (ptr < end) {
    const char* line_end = std::find(ptr, end, '\n');
    line.assign(ptr, line_end);
    ptr = line_end + 1;  // Move to the next line

    // Skip header section
    if (!header_passed) {
      if (line.find("&END") != std::string::npos) {
        header_passed = true;
      }
      continue;
    } else if (!format_detected) {
      // Detect format
      format = detect_fcidump_format(line);
      if (format == FCIDumpFormat::Invalid) {
        continue;  // Skip invalid lines
      }

      format_detected = true;
    }

    auto [valid, p, q, r, s, integral] = fcidump_line(format, line);
    if (!valid) continue;  // not a valid FCIDUMP line

    auto lc = line_classification(p, q, r, s);
    if (lc == LineClassification::Core) {
      return integral;
    }
  }
  return 0.0;
}

void read_fcidump_1body(std::string fname, col_major_span<double, 2> T) {
  if (T.extent(0) != T.extent(1)) throw std::runtime_error("T must be square");

  auto norb = read_fcidump_norb(fname);
  if (T.extent(0) != norb)
    throw std::runtime_error("T is of improper dimension");

  // Read entire file into memory
  std::string content;
  {
    std::ifstream file(fname);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + fname);
    }
    content = std::string((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
  }

  // Parse line by line manually
  const char* ptr = content.c_str();
  const char* end = ptr + content.length();
  std::string line;
  bool header_passed = false;
  bool format_detected = false;
  auto format = FCIDumpFormat::Invalid;

  while (ptr < end) {
    const char* line_end = std::find(ptr, end, '\n');
    line.assign(ptr, line_end);
    ptr = line_end + 1;  // Move to the next line

    // Skip header section
    if (!header_passed) {
      if (line.find("&END") != std::string::npos) {
        header_passed = true;
      }
      continue;
    } else if (!format_detected) {
      // Detect format
      format = detect_fcidump_format(line);
      if (format == FCIDumpFormat::Invalid) {
        continue;  // Skip invalid lines
      }

      format_detected = true;
    }

    auto [valid, p, q, r, s, integral] = fcidump_line(format, line);
    if (!valid) continue;  // not a valid FCIDUMP line

    auto lc = line_classification(p, q, r, s);
    if (lc == LineClassification::OneBody) {
      p--;
      q--;
      T(p, q) = integral;
      T(q, p) = integral;
    }
  }
}

void read_fcidump_1body(std::string fname, double* T, size_t LDT) {
  auto norb = read_fcidump_norb(fname);
  col_major_span<double, 2> T_map(T, LDT, norb);
  read_fcidump_1body(fname, KokkosEx::submdspan(T_map, std::pair{0, norb},
                                                Kokkos::full_extent));
}

void read_fcidump_2body(std::string fname, col_major_span<double, 4> V) {
  if (V.extent(0) != V.extent(1)) throw std::runtime_error("V must be square");
  if (V.extent(0) != V.extent(2)) throw std::runtime_error("V must be square");
  if (V.extent(0) != V.extent(3)) throw std::runtime_error("V must be square");

  auto norb = read_fcidump_norb(fname);
  if (V.extent(0) != norb)
    throw std::runtime_error("V is of improper dimension");

  // Read entire file into memory
  std::string content;
  {
    std::ifstream file(fname);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + fname);
    }
    content = std::string((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
  }

  // Parse line by line manually
  const char* ptr = content.c_str();
  const char* end = ptr + content.length();
  std::string line;
  bool header_passed = false;
  bool format_detected = false;
  auto format = FCIDumpFormat::Invalid;

  while (ptr < end) {
    const char* line_end = std::find(ptr, end, '\n');
    line.assign(ptr, line_end);
    ptr = line_end + 1;  // Move to the next line

    // Skip header section
    if (!header_passed) {
      if (line.find("&END") != std::string::npos) {
        header_passed = true;
      }
      continue;
    } else if (!format_detected) {
      // Detect format
      format = detect_fcidump_format(line);
      if (format == FCIDumpFormat::Invalid) {
        continue;  // Skip invalid lines
      }

      format_detected = true;
    }

    auto [valid, p, q, r, s, integral] = fcidump_line(format, line);
    if (!valid) continue;  // not a valid FCIDUMP line

    auto lc = line_classification(p, q, r, s);
    if (lc == LineClassification::TwoBody) {
      p--;
      q--;
      r--;
      s--;
      V(p, q, r, s) = integral;  // (pq|rs)
      V(p, q, s, r) = integral;  // (pq|sr)
      V(q, p, r, s) = integral;  // (qp|rs)
      V(q, p, s, r) = integral;  // (qp|sr)

      V(r, s, p, q) = integral;  // (rs|pq)
      V(s, r, p, q) = integral;  // (sr|pq)
      V(r, s, q, p) = integral;  // (rs|qp)
      V(s, r, q, p) = integral;  // (sr|qp)
    }
  }
}

void read_fcidump_2body(std::string fname, double* V, size_t LDV) {
  auto norb = read_fcidump_norb(fname);
  col_major_span<double, 4> V_map(V, norb, norb, norb, norb);
  read_fcidump_2body(fname, V_map);
}

void write_fcidump(std::string fname, const FCIDumpHeader& header,
                   const double* T, size_t LDT, const double* V, size_t LDV,
                   double E_core, double threshold) {
  auto logger = spdlog::basic_logger_mt("__fcidump", fname);
  logger->set_pattern("%v");
  // constexpr const char* fmt_string = "{:8} {:8} {:8} {:8} {:25.14e}";
  constexpr const char* fmt_string = "{4:25.14e} {0:8} {1:8} {2:8} {3:8}";

  logger->info("&FCI NORB={},NELEC={},MS2={},\n  ISYM={},", header.norb,
               header.nelec, header.ms2, header.isym);
  if (!header.orbsym.empty()) {
    std::ostringstream orbsym_stream;
    orbsym_stream << "  ORBSYM=";
    for (size_t i = 0; i < header.orbsym.size(); ++i) {
      if (i > 0) orbsym_stream << ",";
      orbsym_stream << header.orbsym[i];
    }
    logger->info(orbsym_stream.str());
  }
  logger->info("&END");

  const auto norb = header.norb;
  // Write two body
  for (size_t i = 0; i < norb; ++i)
    for (size_t j = 0; j < norb; ++j)
      for (size_t k = 0; k < norb; ++k)
        for (size_t l = 0; l < norb; ++l) {
          const auto intrgral =
              V[i + j * LDV + k * LDV * LDV + l * LDV * LDV * LDV];
          if (std::abs(intrgral) < threshold) continue;  // Skip small integrals
          logger->info(fmt_string, i + 1, j + 1, k + 1, l + 1,
                       V[i + j * LDV + k * LDV * LDV + l * LDV * LDV * LDV]);
        }

  // Write one body
  for (size_t i = 0; i < norb; ++i)
    for (size_t j = 0; j < norb; ++j) {
      if (std::abs(T[i + j * LDT]) < threshold)
        continue;  // Skip small integrals
      logger->info(fmt_string, i + 1, j + 1, 0, 0, T[i + j * LDT]);
    }

  // Write core
  logger->info(fmt_string, 0, 0, 0, 0, E_core);
  logger->flush();
  spdlog::drop("__fcidump");
}

void read_rdms_binary(std::string fname, size_t norb, double* ORDM, size_t LDD1,
                      double* TRDM, size_t LDD2) {
  // Zero out rdms in case of sparse data
  for (size_t i = 0; i < norb; ++i)
    for (size_t j = 0; j < norb; ++j) {
      ORDM[i + j * LDD1] = 0;
    }

  for (size_t i = 0; i < norb; ++i)
    for (size_t j = 0; j < norb; ++j)
      for (size_t k = 0; k < norb; ++k)
        for (size_t l = 0; l < norb; ++l) {
          TRDM[i + j * LDD2 + k * LDD2 * LDD2 + l * LDD2 * LDD2 * LDD2] = 0;
        }

  std::ifstream in_file(fname, std::ios::binary);
  if (!in_file) throw std::runtime_error(fname + " not available");

  int _norb_read;
  in_file.read((char*)&_norb_read, sizeof(int));
  if (_norb_read != (int)norb)
    throw std::runtime_error("NORB in RDM file doesn't match " +
                             std::to_string(norb) + " " +
                             std::to_string(_norb_read));

  std::vector<double> raw(norb * norb * norb * norb);

  // Read 1RDM
  in_file.read((char*)raw.data(), norb * norb * sizeof(double));
  for (size_t i = 0; i < norb; ++i)
    for (size_t j = 0; j < norb; ++j) {
      ORDM[i + j * LDD1] = raw[i + j * norb];
    }

  // Read 2RDM
  in_file.read((char*)raw.data(), norb * norb * norb * norb * sizeof(double));
  for (size_t i = 0; i < norb; ++i)
    for (size_t j = 0; j < norb; ++j)
      for (size_t k = 0; k < norb; ++k)
        for (size_t l = 0; l < norb; ++l) {
          TRDM[i + j * LDD2 + k * LDD2 * LDD2 + l * LDD2 * LDD2 * LDD2] =
              raw[i + j * norb + k * norb * norb + l * norb * norb * norb];
        }
}

void write_rdms_binary(std::string fname, size_t norb, const double* ORDM,
                       size_t LDD1, const double* TRDM, size_t LDD2) {
  std::ofstream out_file(fname, std::ios::binary);
  if (!out_file) throw std::runtime_error(fname + " not available");

  int _norb_write = norb;
  out_file.write((char*)&_norb_write, sizeof(int));

  std::vector<double> raw(norb * norb * norb * norb);

  // Pack and Write 1RDM
  for (size_t i = 0; i < norb; ++i)
    for (size_t j = 0; j < norb; ++j) {
      raw[i + j * norb] = ORDM[i + j * LDD1];
    }
  out_file.write((char*)raw.data(), norb * norb * sizeof(double));

  // Pack and Write 2RDM
  for (size_t i = 0; i < norb; ++i)
    for (size_t j = 0; j < norb; ++j)
      for (size_t k = 0; k < norb; ++k)
        for (size_t l = 0; l < norb; ++l) {
          raw[i + j * norb + k * norb * norb + l * norb * norb * norb] =
              TRDM[i + j * LDD2 + k * LDD2 * LDD2 + l * LDD2 * LDD2 * LDD2];
        }
  out_file.write((char*)raw.data(), norb * norb * norb * norb * sizeof(double));
}

void read_fcidump_all(std::string fname, double* T, size_t LDT, double* V,
                      size_t LDV, double& E_core) {
  auto norb = read_fcidump_norb(fname);
  if (norb == 0) {
    throw std::runtime_error("NORB not found or is zero in FCIDUMP header");
  }

  // Initialize arrays to zero
  if (norb == LDT) {
    std::memset(T, 0, norb * norb * sizeof(double));
  } else {
    for (size_t i = 0; i < norb; ++i)
      for (size_t j = 0; j < norb; ++j) {
        T[i + j * LDT] = 0.0;
      }
  }

  if (norb == LDV) {
    std::memset(V, 0, norb * norb * norb * norb * sizeof(double));
  } else {
    // Initialize V to zero
    const size_t LDV2 = LDV * LDV;
    const size_t LDV3 = LDV2 * LDV;
    for (size_t i = 0; i < norb; ++i)
      for (size_t j = 0; j < norb; ++j)
        for (size_t k = 0; k < norb; ++k)
          for (size_t l = 0; l < norb; ++l) {
            V[i + j * LDV + k * LDV2 + l * LDV3] = 0.0;
          }
  }

  // Default core energy
  E_core = 0.0;

  // Read all integrals in a single file pass
  // std::ifstream file(fname);
  std::string content;
  {
    std::ifstream file(fname);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + fname);
    }
    content = std::string((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
  }

  std::string line;
  bool header_passed = false;
  bool format_detected = false;
  auto format = FCIDumpFormat::Invalid;

  // Parse line by line manually
  const char* ptr = content.c_str();
  const char* end = ptr + content.length();

  const size_t LDV2 = LDV * LDV;
  const size_t LDV3 = LDV2 * LDV;
  while (ptr < end) {
    const char* line_end = std::find(ptr, end, '\n');
    line.assign(ptr, line_end);
    ptr = line_end + 1;  // Move to the next line

    // Skip header section
    if (!header_passed) {
      if (line.find("&END") != std::string::npos) {
        header_passed = true;
      }
      continue;
    } else if (!format_detected) {
      // Detect format
      format = detect_fcidump_format(line);
      if (format == FCIDumpFormat::Invalid) {
        continue;  // Skip invalid lines
      }
      format_detected = true;
    }

    // Parse the line
    auto [valid, p, q, r, s, integral] = fcidump_line(format, line);
    if (!valid) {
      continue;
    }  // not a valid FCIDUMP line

    auto lc = line_classification(p, q, r, s);

    if (lc == LineClassification::Core) {
      // Core energy
      E_core = integral;
    } else if (lc == LineClassification::OneBody) {
      // One-body term
      p--;
      q--;
      T[p + q * LDT] = integral;
      T[q + p * LDT] = integral;
    } else if (lc == LineClassification::TwoBody) {
      // Two-body term
      p--;
      q--;
      r--;
      s--;

      V[p + q * LDV + r * LDV2 + s * LDV3] = integral;  // (pq|rs)
      V[p + q * LDV + s * LDV2 + r * LDV3] = integral;  // (pq|sr)
      V[q + p * LDV + r * LDV2 + s * LDV3] = integral;  // (qp|rs)
      V[q + p * LDV + s * LDV2 + r * LDV3] = integral;  // (qp|sr)

      V[r + s * LDV + p * LDV2 + q * LDV3] = integral;  // (rs|pq)
      V[s + r * LDV + p * LDV2 + q * LDV3] = integral;  // (sr|pq)
      V[r + s * LDV + q * LDV2 + p * LDV3] = integral;  // (rs|qp)
      V[s + r * LDV + q * LDV2 + p * LDV3] = integral;  // (sr|qp)
    }
  }
}

}  // namespace macis
