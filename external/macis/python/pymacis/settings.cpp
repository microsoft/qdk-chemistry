// Copyright (c) Microsoft Corporation.

#include "settings.hpp"

void extract_mcscf_settings(const py::dict &settings,
                            macis::MCSCFSettings &mcscf_settings) {
#define OPT_KEYWORD(STR, RES, DTYPE)                     \
  if (ci_settings.contains(STR)) {                       \
    mcscf_settings.RES = ci_settings[STR].cast<DTYPE>(); \
  }

  if (settings.contains("ci")) {
    py::dict ci_settings = settings["ci"].cast<py::dict>();

    OPT_KEYWORD("res_tol", ci_res_tol, double);
    OPT_KEYWORD("max_subspace", ci_max_subspace, size_t);
    OPT_KEYWORD("matel_tol", ci_matel_tol, double);
  }
#undef OPT_KEYWORD
}

void extract_asci_settings(const py::dict &settings,
                           macis::ASCISettings &asci_settings) {
#define OPT_KEYWORD(STR, RES, DTYPE)                  \
  if (asci_dict.contains(STR)) {                      \
    asci_settings.RES = asci_dict[STR].cast<DTYPE>(); \
  }

  if (settings.contains("asci")) {
    py::dict asci_dict = settings["asci"].cast<py::dict>();

    OPT_KEYWORD("ntdets_max", ntdets_max, size_t);
    OPT_KEYWORD("ntdets_min", ntdets_min, size_t);
    OPT_KEYWORD("ncdets_max", ncdets_max, size_t);
    OPT_KEYWORD("ham_el_tol", h_el_tol, double);
    OPT_KEYWORD("rv_prune_tol", rv_prune_tol, double);
    OPT_KEYWORD("pair_max_lim", pair_size_max, size_t);
    OPT_KEYWORD("grow_factor", grow_factor, int);
    OPT_KEYWORD("max_refine_iter", max_refine_iter, size_t);
    OPT_KEYWORD("refine_etol", refine_energy_tol, double);
    OPT_KEYWORD("grow_with_rot", grow_with_rot, bool);
    OPT_KEYWORD("rot_size_start", rot_size_start, size_t);
    OPT_KEYWORD("constraint_lvl", constraint_level, int);
  }
#undef OPT_KEYWORD
}
