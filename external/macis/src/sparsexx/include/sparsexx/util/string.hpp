/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <locale>
#include <string>
#include <vector>

namespace sparsexx {

/**
 * @brief Trims whitespace characters from the left side of a string
 *
 * Removes all leading whitespace characters (spaces, tabs, newlines, etc.)
 * from the beginning of the string. The string is modified in-place.
 *
 * @param s The string to trim (modified in-place)
 * @return Reference to the modified string
 */
static inline std::string& ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [](int c) { return !std::isspace(c); }));
  return s;
}
/**
 * @brief Trims whitespace characters from the right side of a string
 *
 * Removes all trailing whitespace characters (spaces, tabs, newlines, etc.)
 * from the end of the string. The string is modified in-place.
 *
 * @param s The string to trim (modified in-place)
 * @return Reference to the modified string
 */
static inline std::string& rtrim(std::string& s) {
  s.erase(
      std::find_if(s.rbegin(), s.rend(), [](int c) { return !std::isspace(c); })
          .base(),
      s.end());
  return s;
}

/**
 * @brief Trims whitespace characters from both sides of a string
 *
 * Removes all leading and trailing whitespace characters from the string.
 * This is equivalent to calling ltrim followed by rtrim. The string is
 * modified in-place.
 *
 * @param s The string to trim (modified in-place)
 * @return Reference to the modified string
 */
static inline std::string& trim(std::string& s) { return ltrim(rtrim(s)); }

/**
 * @brief Splits a string into tokens based on delimiter characters
 *
 * Tokenizes the input string by splitting it at any of the specified
 * delimiter characters. Leading and trailing delimiters are ignored,
 * and each resulting token is trimmed of whitespace.
 *
 * @param str The input string to tokenize
 * @param delimiters String containing delimiter characters (default: " ")
 * @return Vector of trimmed token strings
 *
 * @note Empty tokens (consecutive delimiters) are not included in the result
 * @note Each token is automatically trimmed of whitespace after splitting
 */
static inline std::vector<std::string> tokenize(
    const std::string& str, const std::string& delimiters = " ") {
  std::vector<std::string> tokens;
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }

  for (auto& t : tokens) trim(t);
  return tokens;

}  // tokenize

}  // namespace sparsexx
