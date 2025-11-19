/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

/// Copyright (c) 2021
/// David Williams-Young
/// Lawrence Berkeley National Laboratory
#pragma once

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/// INI File Handler
class INIFile {
  std::shared_ptr<std::ifstream> inFile_ = nullptr;  ///< INI file

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      dict_;
  ///< INI data fields partitioned by section headings

  /// Parses the input file
  void parse();

  /// Splits query string on "."
  static std::pair<std::string, std::string> splitQuery(const std::string &);

  /**
   *  std::ofstream constructor.
   *
   *  Sets and parses input file from std::ofstream object
   *  \param [in] inFile  File object to parse
   */
  INIFile(std::shared_ptr<std::ifstream> inFile) : inFile_(inFile) { parse(); }

 public:
  // Disable default, copy and move constructors and assignment operators
  INIFile() = delete;
  INIFile(const INIFile &) = delete;
  INIFile(INIFile &&) = delete;
  INIFile &operator=(const INIFile &) = delete;
  INIFile &operator=(INIFile &&) = delete;

  /**
   *  Filename constructor.
   *
   *  Sets and parses  input file given a file name
   *  \param [in] inFileName  Name of  input file
   */
  INIFile(std::string inFileName)
      : INIFile(std::make_shared<std::ifstream>(inFileName)) {}

  /**
   *  \brief Template function which returns the value of a data field
   *  from the input file in a specified datatype given a formatted
   *  query string.
   *
   *  i.e.
   *
   *  INI entry:
   *    [SCF]
   *    DENTOL = 1E-6
   *
   *  Query
   *    double tol = input.getData<double>("SCF.DENTOL");
   *
   *  This example returns the value of the string data field "SCF.DENTOL"
   *  as a double precision number. Various specializations of this function
   *  exist for various datatypes
   *
   *  \param [in] s Formatted query string to be parsed
   *  \return       Value of query data field as specified datatype
   */
  template <typename T>
  T getData(std::string s);

  /**
   *  Checks whether or not the parsed  input file contains
   *  a query section.
   *
   *  \paral  [in] str Query string of a section heading
   *  \return      True if input file contains that heading
   */
  inline bool containsSection(std::string str) const {
    return dict_.find(str) != dict_.end();
  }

  /**
   *  Checks whether or not the parsed  input file contains
   *  a query data field.
   *
   *  \paral  [in] str Query string of a data field (includes section heading)
   *  \return      True if input file contains that data field
   */
  inline bool containsData(std::string str) const {
    auto pr = splitQuery(str);
    if (not containsSection(pr.first)) return false;
    return dict_.at(pr.first).find(pr.second) != dict_.at(pr.first).end();
  }

  inline std::vector<std::string> getDataInSection(std::string section) {
    std::vector<std::string> datasets;

    if (containsSection(section)) {
      for (auto &data : dict_[section]) datasets.emplace_back(data.first);
    }

    return datasets;
  }

};  // INIFile class
