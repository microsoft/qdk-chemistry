// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/config.h>

#include <string>

namespace qdk::chemistry::scf {
/// An enum to classify the available backends for electron-repulsion (ERI)
/// manipulation
enum class ERIMethod {
#ifdef QDK_CHEMISTRY_ENABLE_RYS
  Rys = 1,  ///< GPU accelerated Rys quadrature scheme
#endif
#ifdef QDK_CHEMISTRY_ENABLE_HGP
  HGP = 2,  ///< GPU accelerated Head-Gordon-Pople scheme
#endif
  Incore = 3,  ///< Incore storage of ERI tensor
#ifdef QDK_CHEMISTRY_ENABLE_LIBINTX
  LibintX = 4,  ///< LibintX for direct Coulomb matrix build (DFJ-only)
#endif
  SnK = 5,           ///< Seminumerical Exchange
  Libint2Direct = 6  ///< Direct scheme using Libint2 for integral evaluation
};

/// Convert an ERIMethod to a `std::string`
inline std::string to_string(ERIMethod m) {
  switch (m) {
#ifdef QDK_CHEMISTRY_ENABLE_RYS
    case ERIMethod::Rys:
      return "RYS";
#endif
#ifdef QDK_CHEMISTRY_ENABLE_HGP
    case ERIMethod::HGP:
      return "HGP";
#endif
    case ERIMethod::Incore:
      return "INCORE";
#ifdef QDK_CHEMISTRY_ENABLE_LIBINTX
    case ERIMethod::LibintX:
      return "LIBINTX";
#endif
    case ERIMethod::SnK:
      return "SnK";
    case ERIMethod::Libint2Direct:
      return "Libint2_Direct";
    default:
      return "<Unknown>";
  }
}

/// An enum to classify the available backends for exchange-correlation (EXC)
/// integral evaluation
enum class EXCMethod {
  GauXC  ///< GauXC Backend
};

/// Convert an EXCMEthod to a `std::string`
inline std::string to_string(EXCMethod m) {
  switch (m) {
    case EXCMethod::GauXC:
      return "GauXC";
    default:
      return "<Unknown>";
  }
}

/// An enum to classify the available iterative linear solvers
enum class IterativeLinearSolver {
  GMRES  ///< Generalized Minimum RESidual method
};

/// An enum to classify the available density initialization methods
enum class DensityInitializationMethod {
  SOAD,          ///< Superposition of atomic densities
  Core,          ///< Core Hamiltonian guess (P = 0)
  UserProvided,  ///< User-provided density matrix (incore)
  File,          ///< User-provided density matrix (file)
  Atom,          ///< Atom-by-atom guess
};
}  // namespace qdk::chemistry::scf
