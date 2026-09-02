// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Cross-checks the SW-PT2 kernel against an independent symbolic derivation of
// the projected commutator 1/2 [S, H_OD].
//
// The sibling `test_swpt2_kernel.cpp` grounds its expected values in first
// principles and in a determinant-space oracle. That oracle builds a dense
// 2^n_so matrix, so it stops around ten spin-orbitals, and it shares this
// kernel's conventions -- it re-derives the same algebra by a different route.
// The table below is independent in *derivation*: it was produced offline by a
// computer-algebra expansion of the same commutator, with no knowledge of this
// implementation, and it costs nothing to evaluate at sizes far past the
// oracle's reach.
//
// The table is pre-generated and checked in verbatim, in the emitting tool's
// own output format, so that regenerating it is a copy-paste with no
// translation step. It is not derived at build time and this test adds no
// build dependency. The derivation procedure and the tool used are recorded in
// the team's internal knowledge base; if the kernel's conventions change, these
// tests fail loudly and the table has to be regenerated there.
//
// Columns, one term per line:
//   rA rB S-block V-block ordering wick_sign [deltas] residual_rank
//   [residual creators] [residual annihilators] flags
// Blocks read `creations|annihilations` over i (inactive), u (active) and
// a (virtual). Slots name an operand leg, e.g. `S.c0@i` is operand S's first
// creation, drawn from the inactive space. A delta joins the two legs that
// contract. Only terms whose surviving legs are all active are listed, since
// only those survive the projection onto the active space.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"

namespace {

namespace sw = qdk::chemistry::algorithms::microsoft::swpt2;

/// Guards against silently comparing against a truncated or stale table.
constexpr int kProjectedTermCount = 664;

constexpr const char* kReferenceTerms = R"TERMS(
1 1 i|u u|i SV -1 [S.c0@i~V.a0@i] 1 [V.c0@u] [S.a0@u] PX
1 1 i|u u|i SV 1 [S.c0@i~V.a0@i,S.a0@u~V.c0@u] 0 [] [] PX
1 1 i|a a|i SV 1 [S.c0@i~V.a0@i,S.a0@a~V.c0@a] 0 [] [] PX
1 1 u|i i|u VS -1 [S.a0@i~V.c0@i] 1 [S.c0@u] [V.a0@u] PX
1 1 u|i i|u VS 1 [S.a0@i~V.c0@i,S.c0@u~V.a0@u] 0 [] [] PX
1 1 u|a a|u SV 1 [S.a0@a~V.c0@a] 1 [S.c0@u] [V.a0@u] PX
1 1 a|i i|a VS 1 [S.a0@i~V.c0@i,S.c0@a~V.a0@a] 0 [] [] PX
1 1 a|u u|a VS 1 [S.c0@a~V.a0@a] 1 [V.c0@u] [S.a0@u] PX
1 2 i|u iu|ii SV -1 [S.c0@i~V.a1@i,S.a0@u~V.c1@u,V.a0@i~V.c0@i] 0 [] [] PX
1 2 i|u iu|ii SV -1 [S.c0@i~V.a0@i,V.a1@i~V.c0@i] 1 [V.c1@u] [S.a0@u] PX
1 2 i|u iu|ii SV 1 [S.c0@i~V.a0@i,S.a0@u~V.c1@u,V.a1@i~V.c0@i] 0 [] [] PX
1 2 i|u iu|ii SV 1 [S.c0@i~V.a1@i,V.a0@i~V.c0@i] 1 [V.c1@u] [S.a0@u] PX
1 2 i|u ui|ii SV 1 [S.c0@i~V.a1@i,S.a0@u~V.c0@u,V.a0@i~V.c1@i] 0 [] [] PX
1 2 i|u ui|ii SV 1 [S.c0@i~V.a0@i,V.a1@i~V.c1@i] 1 [V.c0@u] [S.a0@u] PX
1 2 i|u ui|ii SV -1 [S.c0@i~V.a0@i,S.a0@u~V.c0@u,V.a1@i~V.c1@i] 0 [] [] PX
1 2 i|u ui|ii SV -1 [S.c0@i~V.a1@i,V.a0@i~V.c1@i] 1 [V.c0@u] [S.a0@u] PX
1 2 i|u uu|iu SV 1 [S.c0@i~V.a0@i] 2 [V.c0@u,V.c1@u] [S.a0@u,V.a1@u] PX
1 2 i|u uu|iu SV 1 [S.c0@i~V.a0@i,S.a0@u~V.c1@u] 1 [V.c0@u] [V.a1@u] PX
1 2 i|u uu|iu SV -1 [S.c0@i~V.a0@i,S.a0@u~V.c0@u] 1 [V.c1@u] [V.a1@u] PX
1 2 i|u uu|ui SV -1 [S.c0@i~V.a1@i] 2 [V.c0@u,V.c1@u] [S.a0@u,V.a0@u] PX
1 2 i|u uu|ui SV -1 [S.c0@i~V.a1@i,S.a0@u~V.c1@u] 1 [V.c0@u] [V.a0@u] PX
1 2 i|u uu|ui SV 1 [S.c0@i~V.a1@i,S.a0@u~V.c0@u] 1 [V.c1@u] [V.a0@u] PX
1 2 i|a ia|ii SV 1 [S.c0@i~V.a0@i,S.a0@a~V.c1@a,V.a1@i~V.c0@i] 0 [] [] PX
1 2 i|a ia|ii SV -1 [S.c0@i~V.a1@i,S.a0@a~V.c1@a,V.a0@i~V.c0@i] 0 [] [] PX
1 2 i|a ua|iu SV 1 [S.c0@i~V.a0@i,S.a0@a~V.c1@a] 1 [V.c0@u] [V.a1@u] PX
1 2 i|a ua|ui SV -1 [S.c0@i~V.a1@i,S.a0@a~V.c1@a] 1 [V.c0@u] [V.a0@u] PX
1 2 i|a ai|ii SV -1 [S.c0@i~V.a0@i,S.a0@a~V.c0@a,V.a1@i~V.c1@i] 0 [] [] PX
1 2 i|a ai|ii SV 1 [S.c0@i~V.a1@i,S.a0@a~V.c0@a,V.a0@i~V.c1@i] 0 [] [] PX
1 2 i|a au|iu SV -1 [S.c0@i~V.a0@i,S.a0@a~V.c0@a] 1 [V.c1@u] [V.a1@u] PX
1 2 i|a au|ui SV 1 [S.c0@i~V.a1@i,S.a0@a~V.c0@a] 1 [V.c1@u] [V.a0@u] PX
1 2 u|i ii|iu VS 1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i] 1 [S.c0@u] [V.a1@u] PX
1 2 u|i ii|iu VS -1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.c0@u~V.a1@u] 0 [] [] PX
1 2 u|i ii|iu VS -1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i] 1 [S.c0@u] [V.a1@u] PX
1 2 u|i ii|iu VS 1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.c0@u~V.a1@u] 0 [] [] PX
1 2 u|i ii|ui VS -1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i] 1 [S.c0@u] [V.a0@u] PX
1 2 u|i ii|ui VS 1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@u~V.a0@u] 0 [] [] PX
1 2 u|i ii|ui VS 1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i] 1 [S.c0@u] [V.a0@u] PX
1 2 u|i ii|ui VS -1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.c0@u~V.a0@u] 0 [] [] PX
1 2 u|i iu|uu VS -1 [S.a0@i~V.c0@i] 2 [V.c1@u,S.c0@u] [V.a0@u,V.a1@u] PX
1 2 u|i iu|uu VS -1 [S.a0@i~V.c0@i,S.c0@u~V.a0@u] 1 [V.c1@u] [V.a1@u] PX
1 2 u|i iu|uu VS 1 [S.a0@i~V.c0@i,S.c0@u~V.a1@u] 1 [V.c1@u] [V.a0@u] PX
1 2 u|i ui|uu VS 1 [S.a0@i~V.c1@i] 2 [V.c0@u,S.c0@u] [V.a0@u,V.a1@u] PX
1 2 u|i ui|uu VS 1 [S.a0@i~V.c1@i,S.c0@u~V.a0@u] 1 [V.c0@u] [V.a1@u] PX
1 2 u|i ui|uu VS -1 [S.a0@i~V.c1@i,S.c0@u~V.a1@u] 1 [V.c0@u] [V.a0@u] PX
1 2 u|a ia|iu SV -1 [S.a0@a~V.c1@a,V.a0@i~V.c0@i] 1 [S.c0@u] [V.a1@u] PX
1 2 u|a ia|ui SV 1 [S.a0@a~V.c1@a,V.a1@i~V.c0@i] 1 [S.c0@u] [V.a0@u] PX
1 2 u|a ua|uu SV 1 [S.a0@a~V.c1@a] 2 [S.c0@u,V.c0@u] [V.a0@u,V.a1@u] PX
1 2 u|a ai|iu SV 1 [S.a0@a~V.c0@a,V.a0@i~V.c1@i] 1 [S.c0@u] [V.a1@u] PX
1 2 u|a ai|ui SV -1 [S.a0@a~V.c0@a,V.a1@i~V.c1@i] 1 [S.c0@u] [V.a0@u] PX
1 2 u|a au|uu SV -1 [S.a0@a~V.c0@a] 2 [S.c0@u,V.c1@u] [V.a0@u,V.a1@u] PX
1 2 a|i ii|ia VS 1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.c0@a~V.a1@a] 0 [] [] PX
1 2 a|i ii|ia VS -1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.c0@a~V.a1@a] 0 [] [] PX
1 2 a|i ii|ai VS -1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.c0@a~V.a0@a] 0 [] [] PX
1 2 a|i ii|ai VS 1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@a~V.a0@a] 0 [] [] PX
1 2 a|i iu|ua VS 1 [S.a0@i~V.c0@i,S.c0@a~V.a1@a] 1 [V.c1@u] [V.a0@u] PX
1 2 a|i iu|au VS -1 [S.a0@i~V.c0@i,S.c0@a~V.a0@a] 1 [V.c1@u] [V.a1@u] PX
1 2 a|i ui|ua VS -1 [S.a0@i~V.c1@i,S.c0@a~V.a1@a] 1 [V.c0@u] [V.a0@u] PX
1 2 a|i ui|au VS 1 [S.a0@i~V.c1@i,S.c0@a~V.a0@a] 1 [V.c0@u] [V.a1@u] PX
1 2 a|u iu|ia VS -1 [V.a0@i~V.c0@i,S.c0@a~V.a1@a] 1 [V.c1@u] [S.a0@u] PX
1 2 a|u iu|ai VS 1 [V.a1@i~V.c0@i,S.c0@a~V.a0@a] 1 [V.c1@u] [S.a0@u] PX
1 2 a|u ui|ia VS 1 [V.a0@i~V.c1@i,S.c0@a~V.a1@a] 1 [V.c0@u] [S.a0@u] PX
1 2 a|u ui|ai VS -1 [V.a1@i~V.c1@i,S.c0@a~V.a0@a] 1 [V.c0@u] [S.a0@u] PX
1 2 a|u uu|ua VS -1 [S.c0@a~V.a1@a] 2 [V.c0@u,V.c1@u] [V.a0@u,S.a0@u] PX
1 2 a|u uu|au VS 1 [S.c0@a~V.a0@a] 2 [V.c0@u,V.c1@u] [V.a1@u,S.a0@u] PX
2 1 ii|iu u|i SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i] 1 [V.c0@u] [S.a1@u] PX
2 1 ii|iu u|i SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,S.a1@u~V.c0@u] 0 [] [] PX
2 1 ii|iu u|i SV -1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i] 1 [V.c0@u] [S.a1@u] PX
2 1 ii|iu u|i SV 1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,S.a1@u~V.c0@u] 0 [] [] PX
2 1 ii|ia a|i SV 1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,S.a1@a~V.c0@a] 0 [] [] PX
2 1 ii|ia a|i SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,S.a1@a~V.c0@a] 0 [] [] PX
2 1 ii|ui u|i SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i] 1 [V.c0@u] [S.a0@u] PX
2 1 ii|ui u|i SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,S.a0@u~V.c0@u] 0 [] [] PX
2 1 ii|ui u|i SV 1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i] 1 [V.c0@u] [S.a0@u] PX
2 1 ii|ui u|i SV -1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,S.a0@u~V.c0@u] 0 [] [] PX
2 1 ii|ai a|i SV -1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,S.a0@a~V.c0@a] 0 [] [] PX
2 1 ii|ai a|i SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,S.a0@a~V.c0@a] 0 [] [] PX
2 1 iu|ii i|u VS -1 [S.a1@i~V.c0@i,S.c1@u~V.a0@u,S.a0@i~S.c0@i] 0 [] [] PX
2 1 iu|ii i|u VS -1 [S.a0@i~V.c0@i,S.a1@i~S.c0@i] 1 [S.c1@u] [V.a0@u] PX
2 1 iu|ii i|u VS 1 [S.a0@i~V.c0@i,S.c1@u~V.a0@u,S.a1@i~S.c0@i] 0 [] [] PX
2 1 iu|ii i|u VS 1 [S.a1@i~V.c0@i,S.a0@i~S.c0@i] 1 [S.c1@u] [V.a0@u] PX
2 1 iu|ia a|u SV -1 [S.a0@i~S.c0@i,S.a1@a~V.c0@a] 1 [S.c1@u] [V.a0@u] PX
2 1 iu|uu u|i SV -1 [S.c0@i~V.a0@i] 2 [S.c1@u,V.c0@u] [S.a0@u,S.a1@u] PX
2 1 iu|uu u|i SV -1 [S.c0@i~V.a0@i,S.a0@u~V.c0@u] 1 [S.c1@u] [S.a1@u] PX
2 1 iu|uu u|i SV 1 [S.c0@i~V.a0@i,S.a1@u~V.c0@u] 1 [S.c1@u] [S.a0@u] PX
2 1 iu|ua a|i SV 1 [S.c0@i~V.a0@i,S.a1@a~V.c0@a] 1 [S.c1@u] [S.a0@u] PX
2 1 iu|ai a|u SV 1 [S.a1@i~S.c0@i,S.a0@a~V.c0@a] 1 [S.c1@u] [V.a0@u] PX
2 1 iu|au a|i SV -1 [S.c0@i~V.a0@i,S.a0@a~V.c0@a] 1 [S.c1@u] [S.a1@u] PX
2 1 ia|ii i|a VS 1 [S.a0@i~V.c0@i,S.c1@a~V.a0@a,S.a1@i~S.c0@i] 0 [] [] PX
2 1 ia|ii i|a VS -1 [S.a1@i~V.c0@i,S.c1@a~V.a0@a,S.a0@i~S.c0@i] 0 [] [] PX
2 1 ia|iu u|a VS -1 [S.c1@a~V.a0@a,S.a0@i~S.c0@i] 1 [V.c0@u] [S.a1@u] PX
2 1 ia|ui u|a VS 1 [S.c1@a~V.a0@a,S.a1@i~S.c0@i] 1 [V.c0@u] [S.a0@u] PX
2 1 ui|ii i|u VS 1 [S.a1@i~V.c0@i,S.c0@u~V.a0@u,S.a0@i~S.c1@i] 0 [] [] PX
2 1 ui|ii i|u VS 1 [S.a0@i~V.c0@i,S.a1@i~S.c1@i] 1 [S.c0@u] [V.a0@u] PX
2 1 ui|ii i|u VS -1 [S.a0@i~V.c0@i,S.c0@u~V.a0@u,S.a1@i~S.c1@i] 0 [] [] PX
2 1 ui|ii i|u VS -1 [S.a1@i~V.c0@i,S.a0@i~S.c1@i] 1 [S.c0@u] [V.a0@u] PX
2 1 ui|ia a|u SV 1 [S.a0@i~S.c1@i,S.a1@a~V.c0@a] 1 [S.c0@u] [V.a0@u] PX
2 1 ui|uu u|i SV 1 [S.c1@i~V.a0@i] 2 [S.c0@u,V.c0@u] [S.a0@u,S.a1@u] PX
2 1 ui|uu u|i SV 1 [S.c1@i~V.a0@i,S.a0@u~V.c0@u] 1 [S.c0@u] [S.a1@u] PX
2 1 ui|uu u|i SV -1 [S.c1@i~V.a0@i,S.a1@u~V.c0@u] 1 [S.c0@u] [S.a0@u] PX
2 1 ui|ua a|i SV -1 [S.c1@i~V.a0@i,S.a1@a~V.c0@a] 1 [S.c0@u] [S.a0@u] PX
2 1 ui|ai a|u SV -1 [S.a1@i~S.c1@i,S.a0@a~V.c0@a] 1 [S.c0@u] [V.a0@u] PX
2 1 ui|au a|i SV 1 [S.c1@i~V.a0@i,S.a0@a~V.c0@a] 1 [S.c0@u] [S.a1@u] PX
2 1 uu|iu i|u VS 1 [S.a0@i~V.c0@i] 2 [S.c0@u,S.c1@u] [V.a0@u,S.a1@u] PX
2 1 uu|iu i|u VS 1 [S.a0@i~V.c0@i,S.c1@u~V.a0@u] 1 [S.c0@u] [S.a1@u] PX
2 1 uu|iu i|u VS -1 [S.a0@i~V.c0@i,S.c0@u~V.a0@u] 1 [S.c1@u] [S.a1@u] PX
2 1 uu|ui i|u VS -1 [S.a1@i~V.c0@i] 2 [S.c0@u,S.c1@u] [V.a0@u,S.a0@u] PX
2 1 uu|ui i|u VS -1 [S.a1@i~V.c0@i,S.c1@u~V.a0@u] 1 [S.c0@u] [S.a0@u] PX
2 1 uu|ui i|u VS 1 [S.a1@i~V.c0@i,S.c0@u~V.a0@u] 1 [S.c1@u] [S.a0@u] PX
2 1 uu|ua a|u SV -1 [S.a1@a~V.c0@a] 2 [S.c0@u,S.c1@u] [S.a0@u,V.a0@u] PX
2 1 uu|au a|u SV 1 [S.a0@a~V.c0@a] 2 [S.c0@u,S.c1@u] [S.a1@u,V.a0@u] PX
2 1 ua|iu i|a VS 1 [S.a0@i~V.c0@i,S.c1@a~V.a0@a] 1 [S.c0@u] [S.a1@u] PX
2 1 ua|ui i|a VS -1 [S.a1@i~V.c0@i,S.c1@a~V.a0@a] 1 [S.c0@u] [S.a0@u] PX
2 1 ua|uu u|a VS 1 [S.c1@a~V.a0@a] 2 [V.c0@u,S.c0@u] [S.a0@u,S.a1@u] PX
2 1 ai|ii i|a VS -1 [S.a0@i~V.c0@i,S.c0@a~V.a0@a,S.a1@i~S.c1@i] 0 [] [] PX
2 1 ai|ii i|a VS 1 [S.a1@i~V.c0@i,S.c0@a~V.a0@a,S.a0@i~S.c1@i] 0 [] [] PX
2 1 ai|iu u|a VS 1 [S.c0@a~V.a0@a,S.a0@i~S.c1@i] 1 [V.c0@u] [S.a1@u] PX
2 1 ai|ui u|a VS -1 [S.c0@a~V.a0@a,S.a1@i~S.c1@i] 1 [V.c0@u] [S.a0@u] PX
2 1 au|iu i|a VS -1 [S.a0@i~V.c0@i,S.c0@a~V.a0@a] 1 [S.c1@u] [S.a1@u] PX
2 1 au|ui i|a VS 1 [S.a1@i~V.c0@i,S.c0@a~V.a0@a] 1 [S.c1@u] [S.a0@u] PX
2 1 au|uu u|a VS -1 [S.c0@a~V.a0@a] 2 [V.c0@u,S.c1@u] [S.a0@u,S.a1@u] PX
2 2 ii|iu iu|ii SV 1 [S.c0@i~V.a1@i,S.a0@i~S.c1@i,V.a0@i~V.c0@i] 1 [V.c1@u] [S.a1@u] PX
2 2 ii|iu iu|ii SV -1 [S.c0@i~V.a1@i,S.a0@i~S.c1@i,S.a1@u~V.c1@u,V.a0@i~V.c0@i] 0 [] [] PX
2 2 ii|iu iu|ii SV 1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,S.a1@u~V.c1@u,V.a1@i~V.c0@i] 0 [] [] PX
2 2 ii|iu iu|ii SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a1@i,S.a1@u~V.c1@u,V.a0@i~V.c0@i] 0 [] [] PX
2 2 ii|iu iu|ii SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a1@i,V.a0@i~V.c0@i] 1 [V.c1@u] [S.a1@u] PX
2 2 ii|iu iu|ii SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,S.a1@u~V.c1@u,V.a1@i~V.c0@i] 0 [] [] PX
2 2 ii|iu iu|ii SV -1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,V.a1@i~V.c0@i] 1 [V.c1@u] [S.a1@u] PX
2 2 ii|iu iu|ii SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,V.a1@i~V.c0@i] 1 [V.c1@u] [S.a1@u] PX
2 2 ii|iu ui|ii SV -1 [S.c0@i~V.a1@i,S.a0@i~S.c1@i,V.a0@i~V.c1@i] 1 [V.c0@u] [S.a1@u] PX
2 2 ii|iu ui|ii SV 1 [S.c0@i~V.a1@i,S.a0@i~S.c1@i,S.a1@u~V.c0@u,V.a0@i~V.c1@i] 0 [] [] PX
2 2 ii|iu ui|ii SV -1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,S.a1@u~V.c0@u,V.a1@i~V.c1@i] 0 [] [] PX
2 2 ii|iu ui|ii SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a1@i,S.a1@u~V.c0@u,V.a0@i~V.c1@i] 0 [] [] PX
2 2 ii|iu ui|ii SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a1@i,V.a0@i~V.c1@i] 1 [V.c0@u] [S.a1@u] PX
2 2 ii|iu ui|ii SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,S.a1@u~V.c0@u,V.a1@i~V.c1@i] 0 [] [] PX
2 2 ii|iu ui|ii SV 1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,V.a1@i~V.c1@i] 1 [V.c0@u] [S.a1@u] PX
2 2 ii|iu ui|ii SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,V.a1@i~V.c1@i] 1 [V.c0@u] [S.a1@u] PX
2 2 ii|iu uu|iu SV 1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,S.a1@u~V.c1@u] 1 [V.c0@u] [V.a1@u] PX
2 2 ii|iu uu|iu SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i] 2 [V.c0@u,V.c1@u] [S.a1@u,V.a1@u] PX
2 2 ii|iu uu|iu SV 1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i] 2 [V.c0@u,V.c1@u] [S.a1@u,V.a1@u] PX
2 2 ii|iu uu|iu SV -1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,S.a1@u~V.c0@u] 1 [V.c1@u] [V.a1@u] PX
2 2 ii|iu uu|iu SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,S.a1@u~V.c1@u] 1 [V.c0@u] [V.a1@u] PX
2 2 ii|iu uu|iu SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,S.a1@u~V.c0@u] 1 [V.c1@u] [V.a1@u] PX
2 2 ii|iu uu|ui SV -1 [S.c0@i~V.a1@i,S.a0@i~S.c1@i,S.a1@u~V.c1@u] 1 [V.c0@u] [V.a0@u] PX
2 2 ii|iu uu|ui SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a1@i] 2 [V.c0@u,V.c1@u] [S.a1@u,V.a0@u] PX
2 2 ii|iu uu|ui SV -1 [S.c0@i~V.a1@i,S.a0@i~S.c1@i] 2 [V.c0@u,V.c1@u] [S.a1@u,V.a0@u] PX
2 2 ii|iu uu|ui SV 1 [S.c0@i~V.a1@i,S.a0@i~S.c1@i,S.a1@u~V.c0@u] 1 [V.c1@u] [V.a0@u] PX
2 2 ii|iu uu|ui SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a1@i,S.a1@u~V.c1@u] 1 [V.c0@u] [V.a0@u] PX
2 2 ii|iu uu|ui SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a1@i,S.a1@u~V.c0@u] 1 [V.c1@u] [V.a0@u] PX
2 2 ii|ia ia|ii SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,S.a1@a~V.c1@a,V.a1@i~V.c0@i] 0 [] [] PX
2 2 ii|ia ia|ii SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a1@i,S.a1@a~V.c1@a,V.a0@i~V.c0@i] 0 [] [] PX
2 2 ii|ia ia|ii SV -1 [S.c0@i~V.a1@i,S.a0@i~S.c1@i,S.a1@a~V.c1@a,V.a0@i~V.c0@i] 0 [] [] PX
2 2 ii|ia ia|ii SV 1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,S.a1@a~V.c1@a,V.a1@i~V.c0@i] 0 [] [] PX
2 2 ii|ia ua|iu SV 1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,S.a1@a~V.c1@a] 1 [V.c0@u] [V.a1@u] PX
2 2 ii|ia ua|iu SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,S.a1@a~V.c1@a] 1 [V.c0@u] [V.a1@u] PX
2 2 ii|ia ua|ui SV -1 [S.c0@i~V.a1@i,S.a0@i~S.c1@i,S.a1@a~V.c1@a] 1 [V.c0@u] [V.a0@u] PX
2 2 ii|ia ua|ui SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a1@i,S.a1@a~V.c1@a] 1 [V.c0@u] [V.a0@u] PX
2 2 ii|ia ai|ii SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,S.a1@a~V.c0@a,V.a1@i~V.c1@i] 0 [] [] PX
2 2 ii|ia ai|ii SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a1@i,S.a1@a~V.c0@a,V.a0@i~V.c1@i] 0 [] [] PX
2 2 ii|ia ai|ii SV 1 [S.c0@i~V.a1@i,S.a0@i~S.c1@i,S.a1@a~V.c0@a,V.a0@i~V.c1@i] 0 [] [] PX
2 2 ii|ia ai|ii SV -1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,S.a1@a~V.c0@a,V.a1@i~V.c1@i] 0 [] [] PX
2 2 ii|ia au|iu SV -1 [S.c0@i~V.a0@i,S.a0@i~S.c1@i,S.a1@a~V.c0@a] 1 [V.c1@u] [V.a1@u] PX
2 2 ii|ia au|iu SV 1 [S.a0@i~S.c0@i,S.c1@i~V.a0@i,S.a1@a~V.c0@a] 1 [V.c1@u] [V.a1@u] PX
2 2 ii|ia au|ui SV 1 [S.c0@i~V.a1@i,S.a0@i~S.c1@i,S.a1@a~V.c0@a] 1 [V.c1@u] [V.a0@u] PX
2 2 ii|ia au|ui SV -1 [S.a0@i~S.c0@i,S.c1@i~V.a1@i,S.a1@a~V.c0@a] 1 [V.c1@u] [V.a0@u] PX
2 2 ii|ui iu|ii SV -1 [S.c0@i~V.a1@i,S.a1@i~S.c1@i,V.a0@i~V.c0@i] 1 [V.c1@u] [S.a0@u] PX
2 2 ii|ui iu|ii SV 1 [S.c0@i~V.a1@i,S.a1@i~S.c1@i,S.a0@u~V.c1@u,V.a0@i~V.c0@i] 0 [] [] PX
2 2 ii|ui iu|ii SV -1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,S.a0@u~V.c1@u,V.a1@i~V.c0@i] 0 [] [] PX
2 2 ii|ui iu|ii SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a1@i,S.a0@u~V.c1@u,V.a0@i~V.c0@i] 0 [] [] PX
2 2 ii|ui iu|ii SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a1@i,V.a0@i~V.c0@i] 1 [V.c1@u] [S.a0@u] PX
2 2 ii|ui iu|ii SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,S.a0@u~V.c1@u,V.a1@i~V.c0@i] 0 [] [] PX
2 2 ii|ui iu|ii SV 1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,V.a1@i~V.c0@i] 1 [V.c1@u] [S.a0@u] PX
2 2 ii|ui iu|ii SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,V.a1@i~V.c0@i] 1 [V.c1@u] [S.a0@u] PX
2 2 ii|ui ui|ii SV 1 [S.c0@i~V.a1@i,S.a1@i~S.c1@i,V.a0@i~V.c1@i] 1 [V.c0@u] [S.a0@u] PX
2 2 ii|ui ui|ii SV -1 [S.c0@i~V.a1@i,S.a1@i~S.c1@i,S.a0@u~V.c0@u,V.a0@i~V.c1@i] 0 [] [] PX
2 2 ii|ui ui|ii SV 1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,S.a0@u~V.c0@u,V.a1@i~V.c1@i] 0 [] [] PX
2 2 ii|ui ui|ii SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a1@i,S.a0@u~V.c0@u,V.a0@i~V.c1@i] 0 [] [] PX
2 2 ii|ui ui|ii SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a1@i,V.a0@i~V.c1@i] 1 [V.c0@u] [S.a0@u] PX
2 2 ii|ui ui|ii SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,S.a0@u~V.c0@u,V.a1@i~V.c1@i] 0 [] [] PX
2 2 ii|ui ui|ii SV -1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,V.a1@i~V.c1@i] 1 [V.c0@u] [S.a0@u] PX
2 2 ii|ui ui|ii SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,V.a1@i~V.c1@i] 1 [V.c0@u] [S.a0@u] PX
2 2 ii|ui uu|iu SV -1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,S.a0@u~V.c1@u] 1 [V.c0@u] [V.a1@u] PX
2 2 ii|ui uu|iu SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i] 2 [V.c0@u,V.c1@u] [S.a0@u,V.a1@u] PX
2 2 ii|ui uu|iu SV -1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i] 2 [V.c0@u,V.c1@u] [S.a0@u,V.a1@u] PX
2 2 ii|ui uu|iu SV 1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,S.a0@u~V.c0@u] 1 [V.c1@u] [V.a1@u] PX
2 2 ii|ui uu|iu SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,S.a0@u~V.c1@u] 1 [V.c0@u] [V.a1@u] PX
2 2 ii|ui uu|iu SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,S.a0@u~V.c0@u] 1 [V.c1@u] [V.a1@u] PX
2 2 ii|ui uu|ui SV 1 [S.c0@i~V.a1@i,S.a1@i~S.c1@i,S.a0@u~V.c1@u] 1 [V.c0@u] [V.a0@u] PX
2 2 ii|ui uu|ui SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a1@i] 2 [V.c0@u,V.c1@u] [S.a0@u,V.a0@u] PX
2 2 ii|ui uu|ui SV 1 [S.c0@i~V.a1@i,S.a1@i~S.c1@i] 2 [V.c0@u,V.c1@u] [S.a0@u,V.a0@u] PX
2 2 ii|ui uu|ui SV -1 [S.c0@i~V.a1@i,S.a1@i~S.c1@i,S.a0@u~V.c0@u] 1 [V.c1@u] [V.a0@u] PX
2 2 ii|ui uu|ui SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a1@i,S.a0@u~V.c1@u] 1 [V.c0@u] [V.a0@u] PX
2 2 ii|ui uu|ui SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a1@i,S.a0@u~V.c0@u] 1 [V.c1@u] [V.a0@u] PX
2 2 ii|uu uu|ii SV -1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@u~V.c1@u,S.a1@u~V.c0@u] 0 [] [] PX
2 2 ii|uu uu|ii SV 1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@u~V.c1@u,S.a1@u~V.c0@u] 0 [] [] PX
2 2 ii|uu uu|ii SV -1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i] 2 [V.c0@u,V.c1@u] [S.a0@u,S.a1@u] PX
2 2 ii|uu uu|ii SV 1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i] 2 [V.c0@u,V.c1@u] [S.a0@u,S.a1@u] PX
2 2 ii|uu uu|ii SV 1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a1@u~V.c1@u] 1 [V.c0@u] [S.a0@u] PX
2 2 ii|uu uu|ii SV -1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a1@u~V.c1@u] 1 [V.c0@u] [S.a0@u] PX
2 2 ii|uu uu|ii SV -1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@u~V.c0@u] 1 [V.c1@u] [S.a1@u] PX
2 2 ii|uu uu|ii SV 1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a1@u~V.c0@u] 1 [V.c1@u] [S.a0@u] PX
2 2 ii|uu uu|ii SV 1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@u~V.c0@u,S.a1@u~V.c1@u] 0 [] [] PX
2 2 ii|uu uu|ii SV 1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@u~V.c1@u] 1 [V.c0@u] [S.a1@u] PX
2 2 ii|uu uu|ii SV 1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@u~V.c0@u] 1 [V.c1@u] [S.a1@u] PX
2 2 ii|uu uu|ii SV -1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@u~V.c1@u] 1 [V.c0@u] [S.a1@u] PX
2 2 ii|uu uu|ii SV -1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a1@u~V.c0@u] 1 [V.c1@u] [S.a0@u] PX
2 2 ii|uu uu|ii SV -1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@u~V.c0@u,S.a1@u~V.c1@u] 0 [] [] PX
2 2 ii|ua ua|ii SV 1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@u~V.c0@u,S.a1@a~V.c1@a] 0 [] [] PX
2 2 ii|ua ua|ii SV -1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a1@a~V.c1@a] 1 [V.c0@u] [S.a0@u] PX
2 2 ii|ua ua|ii SV -1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@u~V.c0@u,S.a1@a~V.c1@a] 0 [] [] PX
2 2 ii|ua ua|ii SV 1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a1@a~V.c1@a] 1 [V.c0@u] [S.a0@u] PX
2 2 ii|ua au|ii SV -1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@u~V.c1@u,S.a1@a~V.c0@a] 0 [] [] PX
2 2 ii|ua au|ii SV 1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a1@a~V.c0@a] 1 [V.c1@u] [S.a0@u] PX
2 2 ii|ua au|ii SV 1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@u~V.c1@u,S.a1@a~V.c0@a] 0 [] [] PX
2 2 ii|ua au|ii SV -1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a1@a~V.c0@a] 1 [V.c1@u] [S.a0@u] PX
2 2 ii|ai ia|ii SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,S.a0@a~V.c1@a,V.a1@i~V.c0@i] 0 [] [] PX
2 2 ii|ai ia|ii SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a1@i,S.a0@a~V.c1@a,V.a0@i~V.c0@i] 0 [] [] PX
2 2 ii|ai ia|ii SV 1 [S.c0@i~V.a1@i,S.a1@i~S.c1@i,S.a0@a~V.c1@a,V.a0@i~V.c0@i] 0 [] [] PX
2 2 ii|ai ia|ii SV -1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,S.a0@a~V.c1@a,V.a1@i~V.c0@i] 0 [] [] PX
2 2 ii|ai ua|iu SV -1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,S.a0@a~V.c1@a] 1 [V.c0@u] [V.a1@u] PX
2 2 ii|ai ua|iu SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,S.a0@a~V.c1@a] 1 [V.c0@u] [V.a1@u] PX
2 2 ii|ai ua|ui SV 1 [S.c0@i~V.a1@i,S.a1@i~S.c1@i,S.a0@a~V.c1@a] 1 [V.c0@u] [V.a0@u] PX
2 2 ii|ai ua|ui SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a1@i,S.a0@a~V.c1@a] 1 [V.c0@u] [V.a0@u] PX
2 2 ii|ai ai|ii SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,S.a0@a~V.c0@a,V.a1@i~V.c1@i] 0 [] [] PX
2 2 ii|ai ai|ii SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a1@i,S.a0@a~V.c0@a,V.a0@i~V.c1@i] 0 [] [] PX
2 2 ii|ai ai|ii SV -1 [S.c0@i~V.a1@i,S.a1@i~S.c1@i,S.a0@a~V.c0@a,V.a0@i~V.c1@i] 0 [] [] PX
2 2 ii|ai ai|ii SV 1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,S.a0@a~V.c0@a,V.a1@i~V.c1@i] 0 [] [] PX
2 2 ii|ai au|iu SV 1 [S.c0@i~V.a0@i,S.a1@i~S.c1@i,S.a0@a~V.c0@a] 1 [V.c1@u] [V.a1@u] PX
2 2 ii|ai au|iu SV -1 [S.a1@i~S.c0@i,S.c1@i~V.a0@i,S.a0@a~V.c0@a] 1 [V.c1@u] [V.a1@u] PX
2 2 ii|ai au|ui SV -1 [S.c0@i~V.a1@i,S.a1@i~S.c1@i,S.a0@a~V.c0@a] 1 [V.c1@u] [V.a0@u] PX
2 2 ii|ai au|ui SV 1 [S.a1@i~S.c0@i,S.c1@i~V.a1@i,S.a0@a~V.c0@a] 1 [V.c1@u] [V.a0@u] PX
2 2 ii|au ua|ii SV 1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@a~V.c1@a] 1 [V.c0@u] [S.a1@u] PX
2 2 ii|au ua|ii SV -1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@a~V.c1@a,S.a1@u~V.c0@u] 0 [] [] PX
2 2 ii|au ua|ii SV -1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@a~V.c1@a] 1 [V.c0@u] [S.a1@u] PX
2 2 ii|au ua|ii SV 1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@a~V.c1@a,S.a1@u~V.c0@u] 0 [] [] PX
2 2 ii|au au|ii SV -1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@a~V.c0@a] 1 [V.c1@u] [S.a1@u] PX
2 2 ii|au au|ii SV 1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@a~V.c0@a,S.a1@u~V.c1@u] 0 [] [] PX
2 2 ii|au au|ii SV 1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@a~V.c0@a] 1 [V.c1@u] [S.a1@u] PX
2 2 ii|au au|ii SV -1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@a~V.c0@a,S.a1@u~V.c1@u] 0 [] [] PX
2 2 ii|aa aa|ii SV 1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@a~V.c1@a,S.a1@a~V.c0@a] 0 [] [] PX
2 2 ii|aa aa|ii SV 1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@a~V.c0@a,S.a1@a~V.c1@a] 0 [] [] PX
2 2 ii|aa aa|ii SV -1 [S.c0@i~V.a0@i,S.c1@i~V.a1@i,S.a0@a~V.c1@a,S.a1@a~V.c0@a] 0 [] [] PX
2 2 ii|aa aa|ii SV -1 [S.c0@i~V.a1@i,S.c1@i~V.a0@i,S.a0@a~V.c0@a,S.a1@a~V.c1@a] 0 [] [] PX
2 2 iu|ii ii|iu VS 1 [S.a1@i~V.c0@i,V.a0@i~V.c1@i,S.a0@i~S.c0@i] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|ii ii|iu VS -1 [S.a1@i~V.c0@i,V.a0@i~V.c1@i,S.c1@u~V.a1@u,S.a0@i~S.c0@i] 0 [] [] PX
2 2 iu|ii ii|iu VS 1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.c1@u~V.a1@u,S.a1@i~S.c0@i] 0 [] [] PX
2 2 iu|ii ii|iu VS 1 [V.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@u~V.a1@u,S.a0@i~S.c0@i] 0 [] [] PX
2 2 iu|ii ii|iu VS -1 [V.a0@i~V.c0@i,S.a1@i~V.c1@i,S.a0@i~S.c0@i] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|ii ii|iu VS -1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.c1@u~V.a1@u,S.a1@i~S.c0@i] 0 [] [] PX
2 2 iu|ii ii|iu VS -1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.a1@i~S.c0@i] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|ii ii|iu VS 1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.a1@i~S.c0@i] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|ii ii|ui VS -1 [S.a1@i~V.c0@i,V.a1@i~V.c1@i,S.a0@i~S.c0@i] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|ii ii|ui VS 1 [S.a1@i~V.c0@i,V.a1@i~V.c1@i,S.c1@u~V.a0@u,S.a0@i~S.c0@i] 0 [] [] PX
2 2 iu|ii ii|ui VS -1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.c1@u~V.a0@u,S.a1@i~S.c0@i] 0 [] [] PX
2 2 iu|ii ii|ui VS -1 [V.a1@i~V.c0@i,S.a1@i~V.c1@i,S.c1@u~V.a0@u,S.a0@i~S.c0@i] 0 [] [] PX
2 2 iu|ii ii|ui VS 1 [V.a1@i~V.c0@i,S.a1@i~V.c1@i,S.a0@i~S.c0@i] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|ii ii|ui VS 1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@u~V.a0@u,S.a1@i~S.c0@i] 0 [] [] PX
2 2 iu|ii ii|ui VS 1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.a1@i~S.c0@i] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|ii ii|ui VS -1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.a1@i~S.c0@i] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|ii iu|uu VS -1 [S.a0@i~V.c0@i,S.c1@u~V.a0@u,S.a1@i~S.c0@i] 1 [V.c1@u] [V.a1@u] PX
2 2 iu|ii iu|uu VS -1 [S.a0@i~V.c0@i,S.a1@i~S.c0@i] 2 [V.c1@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 iu|ii iu|uu VS 1 [S.a0@i~V.c0@i,S.c1@u~V.a1@u,S.a1@i~S.c0@i] 1 [V.c1@u] [V.a0@u] PX
2 2 iu|ii iu|uu VS 1 [S.a1@i~V.c0@i,S.a0@i~S.c0@i] 2 [V.c1@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 iu|ii iu|uu VS -1 [S.a1@i~V.c0@i,S.c1@u~V.a1@u,S.a0@i~S.c0@i] 1 [V.c1@u] [V.a0@u] PX
2 2 iu|ii iu|uu VS 1 [S.a1@i~V.c0@i,S.c1@u~V.a0@u,S.a0@i~S.c0@i] 1 [V.c1@u] [V.a1@u] PX
2 2 iu|ii ui|uu VS 1 [S.a0@i~V.c1@i,S.c1@u~V.a0@u,S.a1@i~S.c0@i] 1 [V.c0@u] [V.a1@u] PX
2 2 iu|ii ui|uu VS 1 [S.a0@i~V.c1@i,S.a1@i~S.c0@i] 2 [V.c0@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 iu|ii ui|uu VS -1 [S.a0@i~V.c1@i,S.c1@u~V.a1@u,S.a1@i~S.c0@i] 1 [V.c0@u] [V.a0@u] PX
2 2 iu|ii ui|uu VS -1 [S.a1@i~V.c1@i,S.a0@i~S.c0@i] 2 [V.c0@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 iu|ii ui|uu VS 1 [S.a1@i~V.c1@i,S.c1@u~V.a1@u,S.a0@i~S.c0@i] 1 [V.c0@u] [V.a0@u] PX
2 2 iu|ii ui|uu VS -1 [S.a1@i~V.c1@i,S.c1@u~V.a0@u,S.a0@i~S.c0@i] 1 [V.c0@u] [V.a1@u] PX
2 2 iu|ia ia|iu SV 1 [S.a0@i~S.c0@i,S.a1@a~V.c1@a,V.a0@i~V.c0@i] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|ia ia|ui SV -1 [S.a0@i~S.c0@i,S.a1@a~V.c1@a,V.a1@i~V.c0@i] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|ia ua|uu SV -1 [S.a0@i~S.c0@i,S.a1@a~V.c1@a] 2 [S.c1@u,V.c0@u] [V.a0@u,V.a1@u] PX
2 2 iu|ia ai|iu SV -1 [S.a0@i~S.c0@i,S.a1@a~V.c0@a,V.a0@i~V.c1@i] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|ia ai|ui SV 1 [S.a0@i~S.c0@i,S.a1@a~V.c0@a,V.a1@i~V.c1@i] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|ia au|uu SV 1 [S.a0@i~S.c0@i,S.a1@a~V.c0@a] 2 [S.c1@u,V.c1@u] [V.a0@u,V.a1@u] PX
2 2 iu|uu iu|ii SV -1 [S.c0@i~V.a0@i,S.a0@u~V.c1@u,V.a1@i~V.c0@i] 1 [S.c1@u] [S.a1@u] PX
2 2 iu|uu iu|ii SV -1 [S.c0@i~V.a0@i,V.a1@i~V.c0@i] 2 [S.c1@u,V.c1@u] [S.a0@u,S.a1@u] PX
2 2 iu|uu iu|ii SV 1 [S.c0@i~V.a0@i,S.a1@u~V.c1@u,V.a1@i~V.c0@i] 1 [S.c1@u] [S.a0@u] PX
2 2 iu|uu iu|ii SV 1 [S.c0@i~V.a1@i,V.a0@i~V.c0@i] 2 [S.c1@u,V.c1@u] [S.a0@u,S.a1@u] PX
2 2 iu|uu iu|ii SV -1 [S.c0@i~V.a1@i,S.a1@u~V.c1@u,V.a0@i~V.c0@i] 1 [S.c1@u] [S.a0@u] PX
2 2 iu|uu iu|ii SV 1 [S.c0@i~V.a1@i,S.a0@u~V.c1@u,V.a0@i~V.c0@i] 1 [S.c1@u] [S.a1@u] PX
2 2 iu|uu ui|ii SV 1 [S.c0@i~V.a0@i,S.a0@u~V.c0@u,V.a1@i~V.c1@i] 1 [S.c1@u] [S.a1@u] PX
2 2 iu|uu ui|ii SV 1 [S.c0@i~V.a0@i,V.a1@i~V.c1@i] 2 [S.c1@u,V.c0@u] [S.a0@u,S.a1@u] PX
2 2 iu|uu ui|ii SV -1 [S.c0@i~V.a0@i,S.a1@u~V.c0@u,V.a1@i~V.c1@i] 1 [S.c1@u] [S.a0@u] PX
2 2 iu|uu ui|ii SV -1 [S.c0@i~V.a1@i,V.a0@i~V.c1@i] 2 [S.c1@u,V.c0@u] [S.a0@u,S.a1@u] PX
2 2 iu|uu ui|ii SV 1 [S.c0@i~V.a1@i,S.a1@u~V.c0@u,V.a0@i~V.c1@i] 1 [S.c1@u] [S.a0@u] PX
2 2 iu|uu ui|ii SV -1 [S.c0@i~V.a1@i,S.a0@u~V.c0@u,V.a0@i~V.c1@i] 1 [S.c1@u] [S.a1@u] PX
2 2 iu|uu uu|iu SV 1 [S.c0@i~V.a0@i,S.a0@u~V.c0@u] 2 [S.c1@u,V.c1@u] [S.a1@u,V.a1@u] PX
2 2 iu|uu uu|iu SV 1 [S.c0@i~V.a0@i] 3 [S.c1@u,V.c0@u,V.c1@u] [S.a0@u,S.a1@u,V.a1@u] PX
2 2 iu|uu uu|iu SV -1 [S.c0@i~V.a0@i,S.a1@u~V.c0@u] 2 [S.c1@u,V.c1@u] [S.a0@u,V.a1@u] PX
2 2 iu|uu uu|iu SV 1 [S.c0@i~V.a0@i,S.a0@u~V.c0@u,S.a1@u~V.c1@u] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|uu uu|iu SV 1 [S.c0@i~V.a0@i,S.a1@u~V.c1@u] 2 [S.c1@u,V.c0@u] [S.a0@u,V.a1@u] PX
2 2 iu|uu uu|iu SV -1 [S.c0@i~V.a0@i,S.a0@u~V.c1@u,S.a1@u~V.c0@u] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|uu uu|iu SV -1 [S.c0@i~V.a0@i,S.a0@u~V.c1@u] 2 [S.c1@u,V.c0@u] [S.a1@u,V.a1@u] PX
2 2 iu|uu uu|ui SV -1 [S.c0@i~V.a1@i,S.a0@u~V.c0@u] 2 [S.c1@u,V.c1@u] [S.a1@u,V.a0@u] PX
2 2 iu|uu uu|ui SV -1 [S.c0@i~V.a1@i] 3 [S.c1@u,V.c0@u,V.c1@u] [S.a0@u,S.a1@u,V.a0@u] PX
2 2 iu|uu uu|ui SV 1 [S.c0@i~V.a1@i,S.a1@u~V.c0@u] 2 [S.c1@u,V.c1@u] [S.a0@u,V.a0@u] PX
2 2 iu|uu uu|ui SV -1 [S.c0@i~V.a1@i,S.a0@u~V.c0@u,S.a1@u~V.c1@u] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|uu uu|ui SV -1 [S.c0@i~V.a1@i,S.a1@u~V.c1@u] 2 [S.c1@u,V.c0@u] [S.a0@u,V.a0@u] PX
2 2 iu|uu uu|ui SV 1 [S.c0@i~V.a1@i,S.a0@u~V.c1@u,S.a1@u~V.c0@u] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|uu uu|ui SV 1 [S.c0@i~V.a1@i,S.a0@u~V.c1@u] 2 [S.c1@u,V.c0@u] [S.a1@u,V.a0@u] PX
2 2 iu|ua ia|ii SV 1 [S.c0@i~V.a0@i,S.a1@a~V.c1@a,V.a1@i~V.c0@i] 1 [S.c1@u] [S.a0@u] PX
2 2 iu|ua ia|ii SV -1 [S.c0@i~V.a1@i,S.a1@a~V.c1@a,V.a0@i~V.c0@i] 1 [S.c1@u] [S.a0@u] PX
2 2 iu|ua ua|iu SV 1 [S.c0@i~V.a0@i,S.a1@a~V.c1@a] 2 [S.c1@u,V.c0@u] [S.a0@u,V.a1@u] PX
2 2 iu|ua ua|iu SV 1 [S.c0@i~V.a0@i,S.a0@u~V.c0@u,S.a1@a~V.c1@a] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|ua ua|ui SV -1 [S.c0@i~V.a1@i,S.a1@a~V.c1@a] 2 [S.c1@u,V.c0@u] [S.a0@u,V.a0@u] PX
2 2 iu|ua ua|ui SV -1 [S.c0@i~V.a1@i,S.a0@u~V.c0@u,S.a1@a~V.c1@a] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|ua ai|ii SV -1 [S.c0@i~V.a0@i,S.a1@a~V.c0@a,V.a1@i~V.c1@i] 1 [S.c1@u] [S.a0@u] PX
2 2 iu|ua ai|ii SV 1 [S.c0@i~V.a1@i,S.a1@a~V.c0@a,V.a0@i~V.c1@i] 1 [S.c1@u] [S.a0@u] PX
2 2 iu|ua au|iu SV -1 [S.c0@i~V.a0@i,S.a1@a~V.c0@a] 2 [S.c1@u,V.c1@u] [S.a0@u,V.a1@u] PX
2 2 iu|ua au|iu SV -1 [S.c0@i~V.a0@i,S.a0@u~V.c1@u,S.a1@a~V.c0@a] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|ua au|ui SV 1 [S.c0@i~V.a1@i,S.a1@a~V.c0@a] 2 [S.c1@u,V.c1@u] [S.a0@u,V.a0@u] PX
2 2 iu|ua au|ui SV 1 [S.c0@i~V.a1@i,S.a0@u~V.c1@u,S.a1@a~V.c0@a] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|ai ia|iu SV -1 [S.a1@i~S.c0@i,S.a0@a~V.c1@a,V.a0@i~V.c0@i] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|ai ia|ui SV 1 [S.a1@i~S.c0@i,S.a0@a~V.c1@a,V.a1@i~V.c0@i] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|ai ua|uu SV 1 [S.a1@i~S.c0@i,S.a0@a~V.c1@a] 2 [S.c1@u,V.c0@u] [V.a0@u,V.a1@u] PX
2 2 iu|ai ai|iu SV 1 [S.a1@i~S.c0@i,S.a0@a~V.c0@a,V.a0@i~V.c1@i] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|ai ai|ui SV -1 [S.a1@i~S.c0@i,S.a0@a~V.c0@a,V.a1@i~V.c1@i] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|ai au|uu SV -1 [S.a1@i~S.c0@i,S.a0@a~V.c0@a] 2 [S.c1@u,V.c1@u] [V.a0@u,V.a1@u] PX
2 2 iu|au ia|ii SV -1 [S.c0@i~V.a0@i,S.a0@a~V.c1@a,V.a1@i~V.c0@i] 1 [S.c1@u] [S.a1@u] PX
2 2 iu|au ia|ii SV 1 [S.c0@i~V.a1@i,S.a0@a~V.c1@a,V.a0@i~V.c0@i] 1 [S.c1@u] [S.a1@u] PX
2 2 iu|au ua|iu SV -1 [S.c0@i~V.a0@i,S.a0@a~V.c1@a,S.a1@u~V.c0@u] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|au ua|iu SV -1 [S.c0@i~V.a0@i,S.a0@a~V.c1@a] 2 [S.c1@u,V.c0@u] [S.a1@u,V.a1@u] PX
2 2 iu|au ua|ui SV 1 [S.c0@i~V.a1@i,S.a0@a~V.c1@a,S.a1@u~V.c0@u] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|au ua|ui SV 1 [S.c0@i~V.a1@i,S.a0@a~V.c1@a] 2 [S.c1@u,V.c0@u] [S.a1@u,V.a0@u] PX
2 2 iu|au ai|ii SV 1 [S.c0@i~V.a0@i,S.a0@a~V.c0@a,V.a1@i~V.c1@i] 1 [S.c1@u] [S.a1@u] PX
2 2 iu|au ai|ii SV -1 [S.c0@i~V.a1@i,S.a0@a~V.c0@a,V.a0@i~V.c1@i] 1 [S.c1@u] [S.a1@u] PX
2 2 iu|au au|iu SV 1 [S.c0@i~V.a0@i,S.a0@a~V.c0@a,S.a1@u~V.c1@u] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|au au|iu SV 1 [S.c0@i~V.a0@i,S.a0@a~V.c0@a] 2 [S.c1@u,V.c1@u] [S.a1@u,V.a1@u] PX
2 2 iu|au au|ui SV -1 [S.c0@i~V.a1@i,S.a0@a~V.c0@a,S.a1@u~V.c1@u] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|au au|ui SV -1 [S.c0@i~V.a1@i,S.a0@a~V.c0@a] 2 [S.c1@u,V.c1@u] [S.a1@u,V.a0@u] PX
2 2 iu|aa aa|iu SV 1 [S.c0@i~V.a0@i,S.a0@a~V.c0@a,S.a1@a~V.c1@a] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|aa aa|iu SV -1 [S.c0@i~V.a0@i,S.a0@a~V.c1@a,S.a1@a~V.c0@a] 1 [S.c1@u] [V.a1@u] PX
2 2 iu|aa aa|ui SV -1 [S.c0@i~V.a1@i,S.a0@a~V.c0@a,S.a1@a~V.c1@a] 1 [S.c1@u] [V.a0@u] PX
2 2 iu|aa aa|ui SV 1 [S.c0@i~V.a1@i,S.a0@a~V.c1@a,S.a1@a~V.c0@a] 1 [S.c1@u] [V.a0@u] PX
2 2 ia|ii ii|ia VS -1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.c1@a~V.a1@a,S.a1@i~S.c0@i] 0 [] [] PX
2 2 ia|ii ii|ia VS 1 [V.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@a~V.a1@a,S.a0@i~S.c0@i] 0 [] [] PX
2 2 ia|ii ii|ia VS -1 [S.a1@i~V.c0@i,V.a0@i~V.c1@i,S.c1@a~V.a1@a,S.a0@i~S.c0@i] 0 [] [] PX
2 2 ia|ii ii|ia VS 1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.c1@a~V.a1@a,S.a1@i~S.c0@i] 0 [] [] PX
2 2 ia|ii ii|ai VS 1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@a~V.a0@a,S.a1@i~S.c0@i] 0 [] [] PX
2 2 ia|ii ii|ai VS -1 [V.a1@i~V.c0@i,S.a1@i~V.c1@i,S.c1@a~V.a0@a,S.a0@i~S.c0@i] 0 [] [] PX
2 2 ia|ii ii|ai VS 1 [S.a1@i~V.c0@i,V.a1@i~V.c1@i,S.c1@a~V.a0@a,S.a0@i~S.c0@i] 0 [] [] PX
2 2 ia|ii ii|ai VS -1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.c1@a~V.a0@a,S.a1@i~S.c0@i] 0 [] [] PX
2 2 ia|ii iu|ua VS 1 [S.a0@i~V.c0@i,S.c1@a~V.a1@a,S.a1@i~S.c0@i] 1 [V.c1@u] [V.a0@u] PX
2 2 ia|ii iu|ua VS -1 [S.a1@i~V.c0@i,S.c1@a~V.a1@a,S.a0@i~S.c0@i] 1 [V.c1@u] [V.a0@u] PX
2 2 ia|ii iu|au VS -1 [S.a0@i~V.c0@i,S.c1@a~V.a0@a,S.a1@i~S.c0@i] 1 [V.c1@u] [V.a1@u] PX
2 2 ia|ii iu|au VS 1 [S.a1@i~V.c0@i,S.c1@a~V.a0@a,S.a0@i~S.c0@i] 1 [V.c1@u] [V.a1@u] PX
2 2 ia|ii ui|ua VS -1 [S.a0@i~V.c1@i,S.c1@a~V.a1@a,S.a1@i~S.c0@i] 1 [V.c0@u] [V.a0@u] PX
2 2 ia|ii ui|ua VS 1 [S.a1@i~V.c1@i,S.c1@a~V.a1@a,S.a0@i~S.c0@i] 1 [V.c0@u] [V.a0@u] PX
2 2 ia|ii ui|au VS 1 [S.a0@i~V.c1@i,S.c1@a~V.a0@a,S.a1@i~S.c0@i] 1 [V.c0@u] [V.a1@u] PX
2 2 ia|ii ui|au VS -1 [S.a1@i~V.c1@i,S.c1@a~V.a0@a,S.a0@i~S.c0@i] 1 [V.c0@u] [V.a1@u] PX
2 2 ia|iu iu|ia VS 1 [V.a0@i~V.c0@i,S.c1@a~V.a1@a,S.a0@i~S.c0@i] 1 [V.c1@u] [S.a1@u] PX
2 2 ia|iu iu|ai VS -1 [V.a1@i~V.c0@i,S.c1@a~V.a0@a,S.a0@i~S.c0@i] 1 [V.c1@u] [S.a1@u] PX
2 2 ia|iu ui|ia VS -1 [V.a0@i~V.c1@i,S.c1@a~V.a1@a,S.a0@i~S.c0@i] 1 [V.c0@u] [S.a1@u] PX
2 2 ia|iu ui|ai VS 1 [V.a1@i~V.c1@i,S.c1@a~V.a0@a,S.a0@i~S.c0@i] 1 [V.c0@u] [S.a1@u] PX
2 2 ia|iu uu|ua VS 1 [S.c1@a~V.a1@a,S.a0@i~S.c0@i] 2 [V.c0@u,V.c1@u] [V.a0@u,S.a1@u] PX
2 2 ia|iu uu|au VS -1 [S.c1@a~V.a0@a,S.a0@i~S.c0@i] 2 [V.c0@u,V.c1@u] [V.a1@u,S.a1@u] PX
2 2 ia|ui iu|ia VS -1 [V.a0@i~V.c0@i,S.c1@a~V.a1@a,S.a1@i~S.c0@i] 1 [V.c1@u] [S.a0@u] PX
2 2 ia|ui iu|ai VS 1 [V.a1@i~V.c0@i,S.c1@a~V.a0@a,S.a1@i~S.c0@i] 1 [V.c1@u] [S.a0@u] PX
2 2 ia|ui ui|ia VS 1 [V.a0@i~V.c1@i,S.c1@a~V.a1@a,S.a1@i~S.c0@i] 1 [V.c0@u] [S.a0@u] PX
2 2 ia|ui ui|ai VS -1 [V.a1@i~V.c1@i,S.c1@a~V.a0@a,S.a1@i~S.c0@i] 1 [V.c0@u] [S.a0@u] PX
2 2 ia|ui uu|ua VS -1 [S.c1@a~V.a1@a,S.a1@i~S.c0@i] 2 [V.c0@u,V.c1@u] [V.a0@u,S.a0@u] PX
2 2 ia|ui uu|au VS 1 [S.c1@a~V.a0@a,S.a1@i~S.c0@i] 2 [V.c0@u,V.c1@u] [V.a1@u,S.a0@u] PX
2 2 ui|ii ii|iu VS -1 [S.a1@i~V.c0@i,V.a0@i~V.c1@i,S.a0@i~S.c1@i] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|ii ii|iu VS 1 [S.a1@i~V.c0@i,V.a0@i~V.c1@i,S.c0@u~V.a1@u,S.a0@i~S.c1@i] 0 [] [] PX
2 2 ui|ii ii|iu VS -1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.c0@u~V.a1@u,S.a1@i~S.c1@i] 0 [] [] PX
2 2 ui|ii ii|iu VS -1 [V.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@u~V.a1@u,S.a0@i~S.c1@i] 0 [] [] PX
2 2 ui|ii ii|iu VS 1 [V.a0@i~V.c0@i,S.a1@i~V.c1@i,S.a0@i~S.c1@i] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|ii ii|iu VS 1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.c0@u~V.a1@u,S.a1@i~S.c1@i] 0 [] [] PX
2 2 ui|ii ii|iu VS 1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.a1@i~S.c1@i] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|ii ii|iu VS -1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.a1@i~S.c1@i] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|ii ii|ui VS 1 [S.a1@i~V.c0@i,V.a1@i~V.c1@i,S.a0@i~S.c1@i] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|ii ii|ui VS -1 [S.a1@i~V.c0@i,V.a1@i~V.c1@i,S.c0@u~V.a0@u,S.a0@i~S.c1@i] 0 [] [] PX
2 2 ui|ii ii|ui VS 1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.c0@u~V.a0@u,S.a1@i~S.c1@i] 0 [] [] PX
2 2 ui|ii ii|ui VS 1 [V.a1@i~V.c0@i,S.a1@i~V.c1@i,S.c0@u~V.a0@u,S.a0@i~S.c1@i] 0 [] [] PX
2 2 ui|ii ii|ui VS -1 [V.a1@i~V.c0@i,S.a1@i~V.c1@i,S.a0@i~S.c1@i] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|ii ii|ui VS -1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@u~V.a0@u,S.a1@i~S.c1@i] 0 [] [] PX
2 2 ui|ii ii|ui VS -1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.a1@i~S.c1@i] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|ii ii|ui VS 1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.a1@i~S.c1@i] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|ii iu|uu VS 1 [S.a0@i~V.c0@i,S.c0@u~V.a0@u,S.a1@i~S.c1@i] 1 [V.c1@u] [V.a1@u] PX
2 2 ui|ii iu|uu VS 1 [S.a0@i~V.c0@i,S.a1@i~S.c1@i] 2 [V.c1@u,S.c0@u] [V.a0@u,V.a1@u] PX
2 2 ui|ii iu|uu VS -1 [S.a0@i~V.c0@i,S.c0@u~V.a1@u,S.a1@i~S.c1@i] 1 [V.c1@u] [V.a0@u] PX
2 2 ui|ii iu|uu VS -1 [S.a1@i~V.c0@i,S.a0@i~S.c1@i] 2 [V.c1@u,S.c0@u] [V.a0@u,V.a1@u] PX
2 2 ui|ii iu|uu VS 1 [S.a1@i~V.c0@i,S.c0@u~V.a1@u,S.a0@i~S.c1@i] 1 [V.c1@u] [V.a0@u] PX
2 2 ui|ii iu|uu VS -1 [S.a1@i~V.c0@i,S.c0@u~V.a0@u,S.a0@i~S.c1@i] 1 [V.c1@u] [V.a1@u] PX
2 2 ui|ii ui|uu VS -1 [S.a0@i~V.c1@i,S.c0@u~V.a0@u,S.a1@i~S.c1@i] 1 [V.c0@u] [V.a1@u] PX
2 2 ui|ii ui|uu VS -1 [S.a0@i~V.c1@i,S.a1@i~S.c1@i] 2 [V.c0@u,S.c0@u] [V.a0@u,V.a1@u] PX
2 2 ui|ii ui|uu VS 1 [S.a0@i~V.c1@i,S.c0@u~V.a1@u,S.a1@i~S.c1@i] 1 [V.c0@u] [V.a0@u] PX
2 2 ui|ii ui|uu VS 1 [S.a1@i~V.c1@i,S.a0@i~S.c1@i] 2 [V.c0@u,S.c0@u] [V.a0@u,V.a1@u] PX
2 2 ui|ii ui|uu VS -1 [S.a1@i~V.c1@i,S.c0@u~V.a1@u,S.a0@i~S.c1@i] 1 [V.c0@u] [V.a0@u] PX
2 2 ui|ii ui|uu VS 1 [S.a1@i~V.c1@i,S.c0@u~V.a0@u,S.a0@i~S.c1@i] 1 [V.c0@u] [V.a1@u] PX
2 2 ui|ia ia|iu SV -1 [S.a0@i~S.c1@i,S.a1@a~V.c1@a,V.a0@i~V.c0@i] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|ia ia|ui SV 1 [S.a0@i~S.c1@i,S.a1@a~V.c1@a,V.a1@i~V.c0@i] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|ia ua|uu SV 1 [S.a0@i~S.c1@i,S.a1@a~V.c1@a] 2 [S.c0@u,V.c0@u] [V.a0@u,V.a1@u] PX
2 2 ui|ia ai|iu SV 1 [S.a0@i~S.c1@i,S.a1@a~V.c0@a,V.a0@i~V.c1@i] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|ia ai|ui SV -1 [S.a0@i~S.c1@i,S.a1@a~V.c0@a,V.a1@i~V.c1@i] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|ia au|uu SV -1 [S.a0@i~S.c1@i,S.a1@a~V.c0@a] 2 [S.c0@u,V.c1@u] [V.a0@u,V.a1@u] PX
2 2 ui|uu iu|ii SV 1 [S.c1@i~V.a0@i,S.a0@u~V.c1@u,V.a1@i~V.c0@i] 1 [S.c0@u] [S.a1@u] PX
2 2 ui|uu iu|ii SV 1 [S.c1@i~V.a0@i,V.a1@i~V.c0@i] 2 [S.c0@u,V.c1@u] [S.a0@u,S.a1@u] PX
2 2 ui|uu iu|ii SV -1 [S.c1@i~V.a0@i,S.a1@u~V.c1@u,V.a1@i~V.c0@i] 1 [S.c0@u] [S.a0@u] PX
2 2 ui|uu iu|ii SV -1 [S.c1@i~V.a1@i,V.a0@i~V.c0@i] 2 [S.c0@u,V.c1@u] [S.a0@u,S.a1@u] PX
2 2 ui|uu iu|ii SV 1 [S.c1@i~V.a1@i,S.a1@u~V.c1@u,V.a0@i~V.c0@i] 1 [S.c0@u] [S.a0@u] PX
2 2 ui|uu iu|ii SV -1 [S.c1@i~V.a1@i,S.a0@u~V.c1@u,V.a0@i~V.c0@i] 1 [S.c0@u] [S.a1@u] PX
2 2 ui|uu ui|ii SV -1 [S.c1@i~V.a0@i,S.a0@u~V.c0@u,V.a1@i~V.c1@i] 1 [S.c0@u] [S.a1@u] PX
2 2 ui|uu ui|ii SV -1 [S.c1@i~V.a0@i,V.a1@i~V.c1@i] 2 [S.c0@u,V.c0@u] [S.a0@u,S.a1@u] PX
2 2 ui|uu ui|ii SV 1 [S.c1@i~V.a0@i,S.a1@u~V.c0@u,V.a1@i~V.c1@i] 1 [S.c0@u] [S.a0@u] PX
2 2 ui|uu ui|ii SV 1 [S.c1@i~V.a1@i,V.a0@i~V.c1@i] 2 [S.c0@u,V.c0@u] [S.a0@u,S.a1@u] PX
2 2 ui|uu ui|ii SV -1 [S.c1@i~V.a1@i,S.a1@u~V.c0@u,V.a0@i~V.c1@i] 1 [S.c0@u] [S.a0@u] PX
2 2 ui|uu ui|ii SV 1 [S.c1@i~V.a1@i,S.a0@u~V.c0@u,V.a0@i~V.c1@i] 1 [S.c0@u] [S.a1@u] PX
2 2 ui|uu uu|iu SV -1 [S.c1@i~V.a0@i,S.a0@u~V.c0@u] 2 [S.c0@u,V.c1@u] [S.a1@u,V.a1@u] PX
2 2 ui|uu uu|iu SV -1 [S.c1@i~V.a0@i] 3 [S.c0@u,V.c0@u,V.c1@u] [S.a0@u,S.a1@u,V.a1@u] PX
2 2 ui|uu uu|iu SV 1 [S.c1@i~V.a0@i,S.a1@u~V.c0@u] 2 [S.c0@u,V.c1@u] [S.a0@u,V.a1@u] PX
2 2 ui|uu uu|iu SV -1 [S.c1@i~V.a0@i,S.a0@u~V.c0@u,S.a1@u~V.c1@u] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|uu uu|iu SV -1 [S.c1@i~V.a0@i,S.a1@u~V.c1@u] 2 [S.c0@u,V.c0@u] [S.a0@u,V.a1@u] PX
2 2 ui|uu uu|iu SV 1 [S.c1@i~V.a0@i,S.a0@u~V.c1@u,S.a1@u~V.c0@u] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|uu uu|iu SV 1 [S.c1@i~V.a0@i,S.a0@u~V.c1@u] 2 [S.c0@u,V.c0@u] [S.a1@u,V.a1@u] PX
2 2 ui|uu uu|ui SV 1 [S.c1@i~V.a1@i,S.a0@u~V.c0@u] 2 [S.c0@u,V.c1@u] [S.a1@u,V.a0@u] PX
2 2 ui|uu uu|ui SV 1 [S.c1@i~V.a1@i] 3 [S.c0@u,V.c0@u,V.c1@u] [S.a0@u,S.a1@u,V.a0@u] PX
2 2 ui|uu uu|ui SV -1 [S.c1@i~V.a1@i,S.a1@u~V.c0@u] 2 [S.c0@u,V.c1@u] [S.a0@u,V.a0@u] PX
2 2 ui|uu uu|ui SV 1 [S.c1@i~V.a1@i,S.a0@u~V.c0@u,S.a1@u~V.c1@u] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|uu uu|ui SV 1 [S.c1@i~V.a1@i,S.a1@u~V.c1@u] 2 [S.c0@u,V.c0@u] [S.a0@u,V.a0@u] PX
2 2 ui|uu uu|ui SV -1 [S.c1@i~V.a1@i,S.a0@u~V.c1@u,S.a1@u~V.c0@u] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|uu uu|ui SV -1 [S.c1@i~V.a1@i,S.a0@u~V.c1@u] 2 [S.c0@u,V.c0@u] [S.a1@u,V.a0@u] PX
2 2 ui|ua ia|ii SV -1 [S.c1@i~V.a0@i,S.a1@a~V.c1@a,V.a1@i~V.c0@i] 1 [S.c0@u] [S.a0@u] PX
2 2 ui|ua ia|ii SV 1 [S.c1@i~V.a1@i,S.a1@a~V.c1@a,V.a0@i~V.c0@i] 1 [S.c0@u] [S.a0@u] PX
2 2 ui|ua ua|iu SV -1 [S.c1@i~V.a0@i,S.a1@a~V.c1@a] 2 [S.c0@u,V.c0@u] [S.a0@u,V.a1@u] PX
2 2 ui|ua ua|iu SV -1 [S.c1@i~V.a0@i,S.a0@u~V.c0@u,S.a1@a~V.c1@a] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|ua ua|ui SV 1 [S.c1@i~V.a1@i,S.a1@a~V.c1@a] 2 [S.c0@u,V.c0@u] [S.a0@u,V.a0@u] PX
2 2 ui|ua ua|ui SV 1 [S.c1@i~V.a1@i,S.a0@u~V.c0@u,S.a1@a~V.c1@a] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|ua ai|ii SV 1 [S.c1@i~V.a0@i,S.a1@a~V.c0@a,V.a1@i~V.c1@i] 1 [S.c0@u] [S.a0@u] PX
2 2 ui|ua ai|ii SV -1 [S.c1@i~V.a1@i,S.a1@a~V.c0@a,V.a0@i~V.c1@i] 1 [S.c0@u] [S.a0@u] PX
2 2 ui|ua au|iu SV 1 [S.c1@i~V.a0@i,S.a1@a~V.c0@a] 2 [S.c0@u,V.c1@u] [S.a0@u,V.a1@u] PX
2 2 ui|ua au|iu SV 1 [S.c1@i~V.a0@i,S.a0@u~V.c1@u,S.a1@a~V.c0@a] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|ua au|ui SV -1 [S.c1@i~V.a1@i,S.a1@a~V.c0@a] 2 [S.c0@u,V.c1@u] [S.a0@u,V.a0@u] PX
2 2 ui|ua au|ui SV -1 [S.c1@i~V.a1@i,S.a0@u~V.c1@u,S.a1@a~V.c0@a] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|ai ia|iu SV 1 [S.a1@i~S.c1@i,S.a0@a~V.c1@a,V.a0@i~V.c0@i] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|ai ia|ui SV -1 [S.a1@i~S.c1@i,S.a0@a~V.c1@a,V.a1@i~V.c0@i] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|ai ua|uu SV -1 [S.a1@i~S.c1@i,S.a0@a~V.c1@a] 2 [S.c0@u,V.c0@u] [V.a0@u,V.a1@u] PX
2 2 ui|ai ai|iu SV -1 [S.a1@i~S.c1@i,S.a0@a~V.c0@a,V.a0@i~V.c1@i] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|ai ai|ui SV 1 [S.a1@i~S.c1@i,S.a0@a~V.c0@a,V.a1@i~V.c1@i] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|ai au|uu SV 1 [S.a1@i~S.c1@i,S.a0@a~V.c0@a] 2 [S.c0@u,V.c1@u] [V.a0@u,V.a1@u] PX
2 2 ui|au ia|ii SV 1 [S.c1@i~V.a0@i,S.a0@a~V.c1@a,V.a1@i~V.c0@i] 1 [S.c0@u] [S.a1@u] PX
2 2 ui|au ia|ii SV -1 [S.c1@i~V.a1@i,S.a0@a~V.c1@a,V.a0@i~V.c0@i] 1 [S.c0@u] [S.a1@u] PX
2 2 ui|au ua|iu SV 1 [S.c1@i~V.a0@i,S.a0@a~V.c1@a,S.a1@u~V.c0@u] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|au ua|iu SV 1 [S.c1@i~V.a0@i,S.a0@a~V.c1@a] 2 [S.c0@u,V.c0@u] [S.a1@u,V.a1@u] PX
2 2 ui|au ua|ui SV -1 [S.c1@i~V.a1@i,S.a0@a~V.c1@a,S.a1@u~V.c0@u] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|au ua|ui SV -1 [S.c1@i~V.a1@i,S.a0@a~V.c1@a] 2 [S.c0@u,V.c0@u] [S.a1@u,V.a0@u] PX
2 2 ui|au ai|ii SV -1 [S.c1@i~V.a0@i,S.a0@a~V.c0@a,V.a1@i~V.c1@i] 1 [S.c0@u] [S.a1@u] PX
2 2 ui|au ai|ii SV 1 [S.c1@i~V.a1@i,S.a0@a~V.c0@a,V.a0@i~V.c1@i] 1 [S.c0@u] [S.a1@u] PX
2 2 ui|au au|iu SV -1 [S.c1@i~V.a0@i,S.a0@a~V.c0@a,S.a1@u~V.c1@u] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|au au|iu SV -1 [S.c1@i~V.a0@i,S.a0@a~V.c0@a] 2 [S.c0@u,V.c1@u] [S.a1@u,V.a1@u] PX
2 2 ui|au au|ui SV 1 [S.c1@i~V.a1@i,S.a0@a~V.c0@a,S.a1@u~V.c1@u] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|au au|ui SV 1 [S.c1@i~V.a1@i,S.a0@a~V.c0@a] 2 [S.c0@u,V.c1@u] [S.a1@u,V.a0@u] PX
2 2 ui|aa aa|iu SV -1 [S.c1@i~V.a0@i,S.a0@a~V.c0@a,S.a1@a~V.c1@a] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|aa aa|iu SV 1 [S.c1@i~V.a0@i,S.a0@a~V.c1@a,S.a1@a~V.c0@a] 1 [S.c0@u] [V.a1@u] PX
2 2 ui|aa aa|ui SV 1 [S.c1@i~V.a1@i,S.a0@a~V.c0@a,S.a1@a~V.c1@a] 1 [S.c0@u] [V.a0@u] PX
2 2 ui|aa aa|ui SV -1 [S.c1@i~V.a1@i,S.a0@a~V.c1@a,S.a1@a~V.c0@a] 1 [S.c0@u] [V.a0@u] PX
2 2 uu|ii ii|uu VS -1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@u~V.a0@u,S.c0@u~V.a1@u] 0 [] [] PX
2 2 uu|ii ii|uu VS 1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@u~V.a0@u,S.c0@u~V.a1@u] 0 [] [] PX
2 2 uu|ii ii|uu VS -1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i] 2 [S.c0@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 uu|ii ii|uu VS 1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i] 2 [S.c0@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 uu|ii ii|uu VS 1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@u~V.a1@u] 1 [S.c0@u] [V.a0@u] PX
2 2 uu|ii ii|uu VS -1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@u~V.a1@u] 1 [S.c0@u] [V.a0@u] PX
2 2 uu|ii ii|uu VS -1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@u~V.a0@u] 1 [S.c1@u] [V.a1@u] PX
2 2 uu|ii ii|uu VS 1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@u~V.a1@u] 1 [S.c1@u] [V.a0@u] PX
2 2 uu|ii ii|uu VS 1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@u~V.a0@u,S.c1@u~V.a1@u] 0 [] [] PX
2 2 uu|ii ii|uu VS 1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@u~V.a0@u] 1 [S.c0@u] [V.a1@u] PX
2 2 uu|ii ii|uu VS 1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@u~V.a0@u] 1 [S.c1@u] [V.a1@u] PX
2 2 uu|ii ii|uu VS -1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@u~V.a0@u] 1 [S.c0@u] [V.a1@u] PX
2 2 uu|ii ii|uu VS -1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@u~V.a1@u] 1 [S.c1@u] [V.a0@u] PX
2 2 uu|ii ii|uu VS -1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@u~V.a0@u,S.c1@u~V.a1@u] 0 [] [] PX
2 2 uu|iu ii|iu VS 1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.c1@u~V.a1@u] 1 [S.c0@u] [S.a1@u] PX
2 2 uu|iu ii|iu VS -1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i] 2 [S.c0@u,S.c1@u] [V.a1@u,S.a1@u] PX
2 2 uu|iu ii|iu VS 1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i] 2 [S.c0@u,S.c1@u] [V.a1@u,S.a1@u] PX
2 2 uu|iu ii|iu VS -1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.c0@u~V.a1@u] 1 [S.c1@u] [S.a1@u] PX
2 2 uu|iu ii|iu VS -1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.c1@u~V.a1@u] 1 [S.c0@u] [S.a1@u] PX
2 2 uu|iu ii|iu VS 1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.c0@u~V.a1@u] 1 [S.c1@u] [S.a1@u] PX
2 2 uu|iu ii|ui VS -1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.c1@u~V.a0@u] 1 [S.c0@u] [S.a1@u] PX
2 2 uu|iu ii|ui VS 1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i] 2 [S.c0@u,S.c1@u] [V.a0@u,S.a1@u] PX
2 2 uu|iu ii|ui VS -1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i] 2 [S.c0@u,S.c1@u] [V.a0@u,S.a1@u] PX
2 2 uu|iu ii|ui VS 1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.c0@u~V.a0@u] 1 [S.c1@u] [S.a1@u] PX
2 2 uu|iu ii|ui VS 1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@u~V.a0@u] 1 [S.c0@u] [S.a1@u] PX
2 2 uu|iu ii|ui VS -1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@u~V.a0@u] 1 [S.c1@u] [S.a1@u] PX
2 2 uu|iu iu|uu VS 1 [S.a0@i~V.c0@i,S.c0@u~V.a0@u] 2 [V.c1@u,S.c1@u] [V.a1@u,S.a1@u] PX
2 2 uu|iu iu|uu VS 1 [S.a0@i~V.c0@i] 3 [V.c1@u,S.c0@u,S.c1@u] [V.a0@u,V.a1@u,S.a1@u] PX
2 2 uu|iu iu|uu VS -1 [S.a0@i~V.c0@i,S.c0@u~V.a1@u] 2 [V.c1@u,S.c1@u] [V.a0@u,S.a1@u] PX
2 2 uu|iu iu|uu VS 1 [S.a0@i~V.c0@i,S.c0@u~V.a0@u,S.c1@u~V.a1@u] 1 [V.c1@u] [S.a1@u] PX
2 2 uu|iu iu|uu VS 1 [S.a0@i~V.c0@i,S.c1@u~V.a1@u] 2 [V.c1@u,S.c0@u] [V.a0@u,S.a1@u] PX
2 2 uu|iu iu|uu VS -1 [S.a0@i~V.c0@i,S.c1@u~V.a0@u,S.c0@u~V.a1@u] 1 [V.c1@u] [S.a1@u] PX
2 2 uu|iu iu|uu VS -1 [S.a0@i~V.c0@i,S.c1@u~V.a0@u] 2 [V.c1@u,S.c0@u] [V.a1@u,S.a1@u] PX
2 2 uu|iu ui|uu VS -1 [S.a0@i~V.c1@i,S.c0@u~V.a0@u] 2 [V.c0@u,S.c1@u] [V.a1@u,S.a1@u] PX
2 2 uu|iu ui|uu VS -1 [S.a0@i~V.c1@i] 3 [V.c0@u,S.c0@u,S.c1@u] [V.a0@u,V.a1@u,S.a1@u] PX
2 2 uu|iu ui|uu VS 1 [S.a0@i~V.c1@i,S.c0@u~V.a1@u] 2 [V.c0@u,S.c1@u] [V.a0@u,S.a1@u] PX
2 2 uu|iu ui|uu VS -1 [S.a0@i~V.c1@i,S.c0@u~V.a0@u,S.c1@u~V.a1@u] 1 [V.c0@u] [S.a1@u] PX
2 2 uu|iu ui|uu VS -1 [S.a0@i~V.c1@i,S.c1@u~V.a1@u] 2 [V.c0@u,S.c0@u] [V.a0@u,S.a1@u] PX
2 2 uu|iu ui|uu VS 1 [S.a0@i~V.c1@i,S.c1@u~V.a0@u,S.c0@u~V.a1@u] 1 [V.c0@u] [S.a1@u] PX
2 2 uu|iu ui|uu VS 1 [S.a0@i~V.c1@i,S.c1@u~V.a0@u] 2 [V.c0@u,S.c0@u] [V.a1@u,S.a1@u] PX
2 2 uu|ui ii|iu VS -1 [S.a1@i~V.c0@i,V.a0@i~V.c1@i,S.c1@u~V.a1@u] 1 [S.c0@u] [S.a0@u] PX
2 2 uu|ui ii|iu VS 1 [V.a0@i~V.c0@i,S.a1@i~V.c1@i] 2 [S.c0@u,S.c1@u] [V.a1@u,S.a0@u] PX
2 2 uu|ui ii|iu VS -1 [S.a1@i~V.c0@i,V.a0@i~V.c1@i] 2 [S.c0@u,S.c1@u] [V.a1@u,S.a0@u] PX
2 2 uu|ui ii|iu VS 1 [S.a1@i~V.c0@i,V.a0@i~V.c1@i,S.c0@u~V.a1@u] 1 [S.c1@u] [S.a0@u] PX
2 2 uu|ui ii|iu VS 1 [V.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@u~V.a1@u] 1 [S.c0@u] [S.a0@u] PX
2 2 uu|ui ii|iu VS -1 [V.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@u~V.a1@u] 1 [S.c1@u] [S.a0@u] PX
2 2 uu|ui ii|ui VS 1 [S.a1@i~V.c0@i,V.a1@i~V.c1@i,S.c1@u~V.a0@u] 1 [S.c0@u] [S.a0@u] PX
2 2 uu|ui ii|ui VS -1 [V.a1@i~V.c0@i,S.a1@i~V.c1@i] 2 [S.c0@u,S.c1@u] [V.a0@u,S.a0@u] PX
2 2 uu|ui ii|ui VS 1 [S.a1@i~V.c0@i,V.a1@i~V.c1@i] 2 [S.c0@u,S.c1@u] [V.a0@u,S.a0@u] PX
2 2 uu|ui ii|ui VS -1 [S.a1@i~V.c0@i,V.a1@i~V.c1@i,S.c0@u~V.a0@u] 1 [S.c1@u] [S.a0@u] PX
2 2 uu|ui ii|ui VS -1 [V.a1@i~V.c0@i,S.a1@i~V.c1@i,S.c1@u~V.a0@u] 1 [S.c0@u] [S.a0@u] PX
2 2 uu|ui ii|ui VS 1 [V.a1@i~V.c0@i,S.a1@i~V.c1@i,S.c0@u~V.a0@u] 1 [S.c1@u] [S.a0@u] PX
2 2 uu|ui iu|uu VS -1 [S.a1@i~V.c0@i,S.c0@u~V.a0@u] 2 [V.c1@u,S.c1@u] [V.a1@u,S.a0@u] PX
2 2 uu|ui iu|uu VS -1 [S.a1@i~V.c0@i] 3 [V.c1@u,S.c0@u,S.c1@u] [V.a0@u,V.a1@u,S.a0@u] PX
2 2 uu|ui iu|uu VS 1 [S.a1@i~V.c0@i,S.c0@u~V.a1@u] 2 [V.c1@u,S.c1@u] [V.a0@u,S.a0@u] PX
2 2 uu|ui iu|uu VS -1 [S.a1@i~V.c0@i,S.c0@u~V.a0@u,S.c1@u~V.a1@u] 1 [V.c1@u] [S.a0@u] PX
2 2 uu|ui iu|uu VS -1 [S.a1@i~V.c0@i,S.c1@u~V.a1@u] 2 [V.c1@u,S.c0@u] [V.a0@u,S.a0@u] PX
2 2 uu|ui iu|uu VS 1 [S.a1@i~V.c0@i,S.c1@u~V.a0@u,S.c0@u~V.a1@u] 1 [V.c1@u] [S.a0@u] PX
2 2 uu|ui iu|uu VS 1 [S.a1@i~V.c0@i,S.c1@u~V.a0@u] 2 [V.c1@u,S.c0@u] [V.a1@u,S.a0@u] PX
2 2 uu|ui ui|uu VS 1 [S.a1@i~V.c1@i,S.c0@u~V.a0@u] 2 [V.c0@u,S.c1@u] [V.a1@u,S.a0@u] PX
2 2 uu|ui ui|uu VS 1 [S.a1@i~V.c1@i] 3 [V.c0@u,S.c0@u,S.c1@u] [V.a0@u,V.a1@u,S.a0@u] PX
2 2 uu|ui ui|uu VS -1 [S.a1@i~V.c1@i,S.c0@u~V.a1@u] 2 [V.c0@u,S.c1@u] [V.a0@u,S.a0@u] PX
2 2 uu|ui ui|uu VS 1 [S.a1@i~V.c1@i,S.c0@u~V.a0@u,S.c1@u~V.a1@u] 1 [V.c0@u] [S.a0@u] PX
2 2 uu|ui ui|uu VS 1 [S.a1@i~V.c1@i,S.c1@u~V.a1@u] 2 [V.c0@u,S.c0@u] [V.a0@u,S.a0@u] PX
2 2 uu|ui ui|uu VS -1 [S.a1@i~V.c1@i,S.c1@u~V.a0@u,S.c0@u~V.a1@u] 1 [V.c0@u] [S.a0@u] PX
2 2 uu|ui ui|uu VS -1 [S.a1@i~V.c1@i,S.c1@u~V.a0@u] 2 [V.c0@u,S.c0@u] [V.a1@u,S.a0@u] PX
2 2 uu|ua ia|iu SV 1 [S.a1@a~V.c1@a,V.a0@i~V.c0@i] 2 [S.c0@u,S.c1@u] [S.a0@u,V.a1@u] PX
2 2 uu|ua ia|ui SV -1 [S.a1@a~V.c1@a,V.a1@i~V.c0@i] 2 [S.c0@u,S.c1@u] [S.a0@u,V.a0@u] PX
2 2 uu|ua ua|uu SV 1 [S.a0@u~V.c0@u,S.a1@a~V.c1@a] 2 [S.c0@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 uu|ua ua|uu SV -1 [S.a1@a~V.c1@a] 3 [S.c0@u,S.c1@u,V.c0@u] [S.a0@u,V.a0@u,V.a1@u] PX
2 2 uu|ua ai|iu SV -1 [S.a1@a~V.c0@a,V.a0@i~V.c1@i] 2 [S.c0@u,S.c1@u] [S.a0@u,V.a1@u] PX
2 2 uu|ua ai|ui SV 1 [S.a1@a~V.c0@a,V.a1@i~V.c1@i] 2 [S.c0@u,S.c1@u] [S.a0@u,V.a0@u] PX
2 2 uu|ua au|uu SV -1 [S.a0@u~V.c1@u,S.a1@a~V.c0@a] 2 [S.c0@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 uu|ua au|uu SV 1 [S.a1@a~V.c0@a] 3 [S.c0@u,S.c1@u,V.c1@u] [S.a0@u,V.a0@u,V.a1@u] PX
2 2 uu|au ia|iu SV -1 [S.a0@a~V.c1@a,V.a0@i~V.c0@i] 2 [S.c0@u,S.c1@u] [S.a1@u,V.a1@u] PX
2 2 uu|au ia|ui SV 1 [S.a0@a~V.c1@a,V.a1@i~V.c0@i] 2 [S.c0@u,S.c1@u] [S.a1@u,V.a0@u] PX
2 2 uu|au ua|uu SV -1 [S.a0@a~V.c1@a,S.a1@u~V.c0@u] 2 [S.c0@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 uu|au ua|uu SV 1 [S.a0@a~V.c1@a] 3 [S.c0@u,S.c1@u,V.c0@u] [S.a1@u,V.a0@u,V.a1@u] PX
2 2 uu|au ai|iu SV 1 [S.a0@a~V.c0@a,V.a0@i~V.c1@i] 2 [S.c0@u,S.c1@u] [S.a1@u,V.a1@u] PX
2 2 uu|au ai|ui SV -1 [S.a0@a~V.c0@a,V.a1@i~V.c1@i] 2 [S.c0@u,S.c1@u] [S.a1@u,V.a0@u] PX
2 2 uu|au au|uu SV 1 [S.a0@a~V.c0@a,S.a1@u~V.c1@u] 2 [S.c0@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 uu|au au|uu SV -1 [S.a0@a~V.c0@a] 3 [S.c0@u,S.c1@u,V.c1@u] [S.a1@u,V.a0@u,V.a1@u] PX
2 2 uu|aa aa|uu SV 1 [S.a0@a~V.c0@a,S.a1@a~V.c1@a] 2 [S.c0@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 uu|aa aa|uu SV -1 [S.a0@a~V.c1@a,S.a1@a~V.c0@a] 2 [S.c0@u,S.c1@u] [V.a0@u,V.a1@u] PX
2 2 ua|ii ii|ua VS 1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@u~V.a0@u,S.c1@a~V.a1@a] 0 [] [] PX
2 2 ua|ii ii|ua VS -1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@a~V.a1@a] 1 [S.c0@u] [V.a0@u] PX
2 2 ua|ii ii|ua VS -1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@u~V.a0@u,S.c1@a~V.a1@a] 0 [] [] PX
2 2 ua|ii ii|ua VS 1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@a~V.a1@a] 1 [S.c0@u] [V.a0@u] PX
2 2 ua|ii ii|au VS 1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@a~V.a0@a] 1 [S.c0@u] [V.a1@u] PX
2 2 ua|ii ii|au VS -1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@a~V.a0@a,S.c0@u~V.a1@u] 0 [] [] PX
2 2 ua|ii ii|au VS -1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@a~V.a0@a] 1 [S.c0@u] [V.a1@u] PX
2 2 ua|ii ii|au VS 1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@a~V.a0@a,S.c0@u~V.a1@u] 0 [] [] PX
2 2 ua|iu ii|ia VS 1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.c1@a~V.a1@a] 1 [S.c0@u] [S.a1@u] PX
2 2 ua|iu ii|ia VS -1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.c1@a~V.a1@a] 1 [S.c0@u] [S.a1@u] PX
2 2 ua|iu ii|ai VS -1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.c1@a~V.a0@a] 1 [S.c0@u] [S.a1@u] PX
2 2 ua|iu ii|ai VS 1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@a~V.a0@a] 1 [S.c0@u] [S.a1@u] PX
2 2 ua|iu iu|ua VS 1 [S.a0@i~V.c0@i,S.c1@a~V.a1@a] 2 [V.c1@u,S.c0@u] [V.a0@u,S.a1@u] PX
2 2 ua|iu iu|ua VS 1 [S.a0@i~V.c0@i,S.c0@u~V.a0@u,S.c1@a~V.a1@a] 1 [V.c1@u] [S.a1@u] PX
2 2 ua|iu iu|au VS -1 [S.a0@i~V.c0@i,S.c1@a~V.a0@a,S.c0@u~V.a1@u] 1 [V.c1@u] [S.a1@u] PX
2 2 ua|iu iu|au VS -1 [S.a0@i~V.c0@i,S.c1@a~V.a0@a] 2 [V.c1@u,S.c0@u] [V.a1@u,S.a1@u] PX
2 2 ua|iu ui|ua VS -1 [S.a0@i~V.c1@i,S.c1@a~V.a1@a] 2 [V.c0@u,S.c0@u] [V.a0@u,S.a1@u] PX
2 2 ua|iu ui|ua VS -1 [S.a0@i~V.c1@i,S.c0@u~V.a0@u,S.c1@a~V.a1@a] 1 [V.c0@u] [S.a1@u] PX
2 2 ua|iu ui|au VS 1 [S.a0@i~V.c1@i,S.c1@a~V.a0@a,S.c0@u~V.a1@u] 1 [V.c0@u] [S.a1@u] PX
2 2 ua|iu ui|au VS 1 [S.a0@i~V.c1@i,S.c1@a~V.a0@a] 2 [V.c0@u,S.c0@u] [V.a1@u,S.a1@u] PX
2 2 ua|ui ii|ia VS -1 [S.a1@i~V.c0@i,V.a0@i~V.c1@i,S.c1@a~V.a1@a] 1 [S.c0@u] [S.a0@u] PX
2 2 ua|ui ii|ia VS 1 [V.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@a~V.a1@a] 1 [S.c0@u] [S.a0@u] PX
2 2 ua|ui ii|ai VS 1 [S.a1@i~V.c0@i,V.a1@i~V.c1@i,S.c1@a~V.a0@a] 1 [S.c0@u] [S.a0@u] PX
2 2 ua|ui ii|ai VS -1 [V.a1@i~V.c0@i,S.a1@i~V.c1@i,S.c1@a~V.a0@a] 1 [S.c0@u] [S.a0@u] PX
2 2 ua|ui iu|ua VS -1 [S.a1@i~V.c0@i,S.c1@a~V.a1@a] 2 [V.c1@u,S.c0@u] [V.a0@u,S.a0@u] PX
2 2 ua|ui iu|ua VS -1 [S.a1@i~V.c0@i,S.c0@u~V.a0@u,S.c1@a~V.a1@a] 1 [V.c1@u] [S.a0@u] PX
2 2 ua|ui iu|au VS 1 [S.a1@i~V.c0@i,S.c1@a~V.a0@a,S.c0@u~V.a1@u] 1 [V.c1@u] [S.a0@u] PX
2 2 ua|ui iu|au VS 1 [S.a1@i~V.c0@i,S.c1@a~V.a0@a] 2 [V.c1@u,S.c0@u] [V.a1@u,S.a0@u] PX
2 2 ua|ui ui|ua VS 1 [S.a1@i~V.c1@i,S.c1@a~V.a1@a] 2 [V.c0@u,S.c0@u] [V.a0@u,S.a0@u] PX
2 2 ua|ui ui|ua VS 1 [S.a1@i~V.c1@i,S.c0@u~V.a0@u,S.c1@a~V.a1@a] 1 [V.c0@u] [S.a0@u] PX
2 2 ua|ui ui|au VS -1 [S.a1@i~V.c1@i,S.c1@a~V.a0@a,S.c0@u~V.a1@u] 1 [V.c0@u] [S.a0@u] PX
2 2 ua|ui ui|au VS -1 [S.a1@i~V.c1@i,S.c1@a~V.a0@a] 2 [V.c0@u,S.c0@u] [V.a1@u,S.a0@u] PX
2 2 ua|uu iu|ia VS -1 [V.a0@i~V.c0@i,S.c1@a~V.a1@a] 2 [V.c1@u,S.c0@u] [S.a0@u,S.a1@u] PX
2 2 ua|uu iu|ai VS 1 [V.a1@i~V.c0@i,S.c1@a~V.a0@a] 2 [V.c1@u,S.c0@u] [S.a0@u,S.a1@u] PX
2 2 ua|uu ui|ia VS 1 [V.a0@i~V.c1@i,S.c1@a~V.a1@a] 2 [V.c0@u,S.c0@u] [S.a0@u,S.a1@u] PX
2 2 ua|uu ui|ai VS -1 [V.a1@i~V.c1@i,S.c1@a~V.a0@a] 2 [V.c0@u,S.c0@u] [S.a0@u,S.a1@u] PX
2 2 ua|uu uu|ua VS 1 [S.c0@u~V.a0@u,S.c1@a~V.a1@a] 2 [V.c0@u,V.c1@u] [S.a0@u,S.a1@u] PX
2 2 ua|uu uu|ua VS -1 [S.c1@a~V.a1@a] 3 [V.c0@u,V.c1@u,S.c0@u] [V.a0@u,S.a0@u,S.a1@u] PX
2 2 ua|uu uu|au VS -1 [S.c1@a~V.a0@a,S.c0@u~V.a1@u] 2 [V.c0@u,V.c1@u] [S.a0@u,S.a1@u] PX
2 2 ua|uu uu|au VS 1 [S.c1@a~V.a0@a] 3 [V.c0@u,V.c1@u,S.c0@u] [V.a1@u,S.a0@u,S.a1@u] PX
2 2 ai|ii ii|ia VS 1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.c0@a~V.a1@a,S.a1@i~S.c1@i] 0 [] [] PX
2 2 ai|ii ii|ia VS -1 [V.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@a~V.a1@a,S.a0@i~S.c1@i] 0 [] [] PX
2 2 ai|ii ii|ia VS 1 [S.a1@i~V.c0@i,V.a0@i~V.c1@i,S.c0@a~V.a1@a,S.a0@i~S.c1@i] 0 [] [] PX
2 2 ai|ii ii|ia VS -1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.c0@a~V.a1@a,S.a1@i~S.c1@i] 0 [] [] PX
2 2 ai|ii ii|ai VS -1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@a~V.a0@a,S.a1@i~S.c1@i] 0 [] [] PX
2 2 ai|ii ii|ai VS 1 [V.a1@i~V.c0@i,S.a1@i~V.c1@i,S.c0@a~V.a0@a,S.a0@i~S.c1@i] 0 [] [] PX
2 2 ai|ii ii|ai VS -1 [S.a1@i~V.c0@i,V.a1@i~V.c1@i,S.c0@a~V.a0@a,S.a0@i~S.c1@i] 0 [] [] PX
2 2 ai|ii ii|ai VS 1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.c0@a~V.a0@a,S.a1@i~S.c1@i] 0 [] [] PX
2 2 ai|ii iu|ua VS -1 [S.a0@i~V.c0@i,S.c0@a~V.a1@a,S.a1@i~S.c1@i] 1 [V.c1@u] [V.a0@u] PX
2 2 ai|ii iu|ua VS 1 [S.a1@i~V.c0@i,S.c0@a~V.a1@a,S.a0@i~S.c1@i] 1 [V.c1@u] [V.a0@u] PX
2 2 ai|ii iu|au VS 1 [S.a0@i~V.c0@i,S.c0@a~V.a0@a,S.a1@i~S.c1@i] 1 [V.c1@u] [V.a1@u] PX
2 2 ai|ii iu|au VS -1 [S.a1@i~V.c0@i,S.c0@a~V.a0@a,S.a0@i~S.c1@i] 1 [V.c1@u] [V.a1@u] PX
2 2 ai|ii ui|ua VS 1 [S.a0@i~V.c1@i,S.c0@a~V.a1@a,S.a1@i~S.c1@i] 1 [V.c0@u] [V.a0@u] PX
2 2 ai|ii ui|ua VS -1 [S.a1@i~V.c1@i,S.c0@a~V.a1@a,S.a0@i~S.c1@i] 1 [V.c0@u] [V.a0@u] PX
2 2 ai|ii ui|au VS -1 [S.a0@i~V.c1@i,S.c0@a~V.a0@a,S.a1@i~S.c1@i] 1 [V.c0@u] [V.a1@u] PX
2 2 ai|ii ui|au VS 1 [S.a1@i~V.c1@i,S.c0@a~V.a0@a,S.a0@i~S.c1@i] 1 [V.c0@u] [V.a1@u] PX
2 2 ai|iu iu|ia VS -1 [V.a0@i~V.c0@i,S.c0@a~V.a1@a,S.a0@i~S.c1@i] 1 [V.c1@u] [S.a1@u] PX
2 2 ai|iu iu|ai VS 1 [V.a1@i~V.c0@i,S.c0@a~V.a0@a,S.a0@i~S.c1@i] 1 [V.c1@u] [S.a1@u] PX
2 2 ai|iu ui|ia VS 1 [V.a0@i~V.c1@i,S.c0@a~V.a1@a,S.a0@i~S.c1@i] 1 [V.c0@u] [S.a1@u] PX
2 2 ai|iu ui|ai VS -1 [V.a1@i~V.c1@i,S.c0@a~V.a0@a,S.a0@i~S.c1@i] 1 [V.c0@u] [S.a1@u] PX
2 2 ai|iu uu|ua VS -1 [S.c0@a~V.a1@a,S.a0@i~S.c1@i] 2 [V.c0@u,V.c1@u] [V.a0@u,S.a1@u] PX
2 2 ai|iu uu|au VS 1 [S.c0@a~V.a0@a,S.a0@i~S.c1@i] 2 [V.c0@u,V.c1@u] [V.a1@u,S.a1@u] PX
2 2 ai|ui iu|ia VS 1 [V.a0@i~V.c0@i,S.c0@a~V.a1@a,S.a1@i~S.c1@i] 1 [V.c1@u] [S.a0@u] PX
2 2 ai|ui iu|ai VS -1 [V.a1@i~V.c0@i,S.c0@a~V.a0@a,S.a1@i~S.c1@i] 1 [V.c1@u] [S.a0@u] PX
2 2 ai|ui ui|ia VS -1 [V.a0@i~V.c1@i,S.c0@a~V.a1@a,S.a1@i~S.c1@i] 1 [V.c0@u] [S.a0@u] PX
2 2 ai|ui ui|ai VS 1 [V.a1@i~V.c1@i,S.c0@a~V.a0@a,S.a1@i~S.c1@i] 1 [V.c0@u] [S.a0@u] PX
2 2 ai|ui uu|ua VS 1 [S.c0@a~V.a1@a,S.a1@i~S.c1@i] 2 [V.c0@u,V.c1@u] [V.a0@u,S.a0@u] PX
2 2 ai|ui uu|au VS -1 [S.c0@a~V.a0@a,S.a1@i~S.c1@i] 2 [V.c0@u,V.c1@u] [V.a1@u,S.a0@u] PX
2 2 au|ii ii|ua VS -1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@u~V.a0@u,S.c0@a~V.a1@a] 0 [] [] PX
2 2 au|ii ii|ua VS 1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@a~V.a1@a] 1 [S.c1@u] [V.a0@u] PX
2 2 au|ii ii|ua VS 1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@u~V.a0@u,S.c0@a~V.a1@a] 0 [] [] PX
2 2 au|ii ii|ua VS -1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@a~V.a1@a] 1 [S.c1@u] [V.a0@u] PX
2 2 au|ii ii|au VS -1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@a~V.a0@a] 1 [S.c1@u] [V.a1@u] PX
2 2 au|ii ii|au VS 1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@a~V.a0@a,S.c1@u~V.a1@u] 0 [] [] PX
2 2 au|ii ii|au VS 1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@a~V.a0@a] 1 [S.c1@u] [V.a1@u] PX
2 2 au|ii ii|au VS -1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@a~V.a0@a,S.c1@u~V.a1@u] 0 [] [] PX
2 2 au|iu ii|ia VS -1 [S.a0@i~V.c0@i,V.a0@i~V.c1@i,S.c0@a~V.a1@a] 1 [S.c1@u] [S.a1@u] PX
2 2 au|iu ii|ia VS 1 [V.a0@i~V.c0@i,S.a0@i~V.c1@i,S.c0@a~V.a1@a] 1 [S.c1@u] [S.a1@u] PX
2 2 au|iu ii|ai VS 1 [S.a0@i~V.c0@i,V.a1@i~V.c1@i,S.c0@a~V.a0@a] 1 [S.c1@u] [S.a1@u] PX
2 2 au|iu ii|ai VS -1 [V.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@a~V.a0@a] 1 [S.c1@u] [S.a1@u] PX
2 2 au|iu iu|ua VS -1 [S.a0@i~V.c0@i,S.c0@a~V.a1@a] 2 [V.c1@u,S.c1@u] [V.a0@u,S.a1@u] PX
2 2 au|iu iu|ua VS -1 [S.a0@i~V.c0@i,S.c1@u~V.a0@u,S.c0@a~V.a1@a] 1 [V.c1@u] [S.a1@u] PX
2 2 au|iu iu|au VS 1 [S.a0@i~V.c0@i,S.c0@a~V.a0@a,S.c1@u~V.a1@u] 1 [V.c1@u] [S.a1@u] PX
2 2 au|iu iu|au VS 1 [S.a0@i~V.c0@i,S.c0@a~V.a0@a] 2 [V.c1@u,S.c1@u] [V.a1@u,S.a1@u] PX
2 2 au|iu ui|ua VS 1 [S.a0@i~V.c1@i,S.c0@a~V.a1@a] 2 [V.c0@u,S.c1@u] [V.a0@u,S.a1@u] PX
2 2 au|iu ui|ua VS 1 [S.a0@i~V.c1@i,S.c1@u~V.a0@u,S.c0@a~V.a1@a] 1 [V.c0@u] [S.a1@u] PX
2 2 au|iu ui|au VS -1 [S.a0@i~V.c1@i,S.c0@a~V.a0@a,S.c1@u~V.a1@u] 1 [V.c0@u] [S.a1@u] PX
2 2 au|iu ui|au VS -1 [S.a0@i~V.c1@i,S.c0@a~V.a0@a] 2 [V.c0@u,S.c1@u] [V.a1@u,S.a1@u] PX
2 2 au|ui ii|ia VS 1 [S.a1@i~V.c0@i,V.a0@i~V.c1@i,S.c0@a~V.a1@a] 1 [S.c1@u] [S.a0@u] PX
2 2 au|ui ii|ia VS -1 [V.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@a~V.a1@a] 1 [S.c1@u] [S.a0@u] PX
2 2 au|ui ii|ai VS -1 [S.a1@i~V.c0@i,V.a1@i~V.c1@i,S.c0@a~V.a0@a] 1 [S.c1@u] [S.a0@u] PX
2 2 au|ui ii|ai VS 1 [V.a1@i~V.c0@i,S.a1@i~V.c1@i,S.c0@a~V.a0@a] 1 [S.c1@u] [S.a0@u] PX
2 2 au|ui iu|ua VS 1 [S.a1@i~V.c0@i,S.c0@a~V.a1@a] 2 [V.c1@u,S.c1@u] [V.a0@u,S.a0@u] PX
2 2 au|ui iu|ua VS 1 [S.a1@i~V.c0@i,S.c1@u~V.a0@u,S.c0@a~V.a1@a] 1 [V.c1@u] [S.a0@u] PX
2 2 au|ui iu|au VS -1 [S.a1@i~V.c0@i,S.c0@a~V.a0@a,S.c1@u~V.a1@u] 1 [V.c1@u] [S.a0@u] PX
2 2 au|ui iu|au VS -1 [S.a1@i~V.c0@i,S.c0@a~V.a0@a] 2 [V.c1@u,S.c1@u] [V.a1@u,S.a0@u] PX
2 2 au|ui ui|ua VS -1 [S.a1@i~V.c1@i,S.c0@a~V.a1@a] 2 [V.c0@u,S.c1@u] [V.a0@u,S.a0@u] PX
2 2 au|ui ui|ua VS -1 [S.a1@i~V.c1@i,S.c1@u~V.a0@u,S.c0@a~V.a1@a] 1 [V.c0@u] [S.a0@u] PX
2 2 au|ui ui|au VS 1 [S.a1@i~V.c1@i,S.c0@a~V.a0@a,S.c1@u~V.a1@u] 1 [V.c0@u] [S.a0@u] PX
2 2 au|ui ui|au VS 1 [S.a1@i~V.c1@i,S.c0@a~V.a0@a] 2 [V.c0@u,S.c1@u] [V.a1@u,S.a0@u] PX
2 2 au|uu iu|ia VS 1 [V.a0@i~V.c0@i,S.c0@a~V.a1@a] 2 [V.c1@u,S.c1@u] [S.a0@u,S.a1@u] PX
2 2 au|uu iu|ai VS -1 [V.a1@i~V.c0@i,S.c0@a~V.a0@a] 2 [V.c1@u,S.c1@u] [S.a0@u,S.a1@u] PX
2 2 au|uu ui|ia VS -1 [V.a0@i~V.c1@i,S.c0@a~V.a1@a] 2 [V.c0@u,S.c1@u] [S.a0@u,S.a1@u] PX
2 2 au|uu ui|ai VS 1 [V.a1@i~V.c1@i,S.c0@a~V.a0@a] 2 [V.c0@u,S.c1@u] [S.a0@u,S.a1@u] PX
2 2 au|uu uu|ua VS -1 [S.c1@u~V.a0@u,S.c0@a~V.a1@a] 2 [V.c0@u,V.c1@u] [S.a0@u,S.a1@u] PX
2 2 au|uu uu|ua VS 1 [S.c0@a~V.a1@a] 3 [V.c0@u,V.c1@u,S.c1@u] [V.a0@u,S.a0@u,S.a1@u] PX
2 2 au|uu uu|au VS 1 [S.c0@a~V.a0@a,S.c1@u~V.a1@u] 2 [V.c0@u,V.c1@u] [S.a0@u,S.a1@u] PX
2 2 au|uu uu|au VS -1 [S.c0@a~V.a0@a] 3 [V.c0@u,V.c1@u,S.c1@u] [V.a1@u,S.a0@u,S.a1@u] PX
2 2 aa|ii ii|aa VS 1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c1@a~V.a0@a,S.c0@a~V.a1@a] 0 [] [] PX
2 2 aa|ii ii|aa VS 1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c0@a~V.a0@a,S.c1@a~V.a1@a] 0 [] [] PX
2 2 aa|ii ii|aa VS -1 [S.a0@i~V.c0@i,S.a1@i~V.c1@i,S.c1@a~V.a0@a,S.c0@a~V.a1@a] 0 [] [] PX
2 2 aa|ii ii|aa VS -1 [S.a1@i~V.c0@i,S.a0@i~V.c1@i,S.c0@a~V.a0@a,S.c1@a~V.a1@a] 0 [] [] PX
2 2 aa|iu iu|aa VS 1 [S.a0@i~V.c0@i,S.c0@a~V.a0@a,S.c1@a~V.a1@a] 1 [V.c1@u] [S.a1@u] PX
2 2 aa|iu iu|aa VS -1 [S.a0@i~V.c0@i,S.c1@a~V.a0@a,S.c0@a~V.a1@a] 1 [V.c1@u] [S.a1@u] PX
2 2 aa|iu ui|aa VS -1 [S.a0@i~V.c1@i,S.c0@a~V.a0@a,S.c1@a~V.a1@a] 1 [V.c0@u] [S.a1@u] PX
2 2 aa|iu ui|aa VS 1 [S.a0@i~V.c1@i,S.c1@a~V.a0@a,S.c0@a~V.a1@a] 1 [V.c0@u] [S.a1@u] PX
2 2 aa|ui iu|aa VS -1 [S.a1@i~V.c0@i,S.c0@a~V.a0@a,S.c1@a~V.a1@a] 1 [V.c1@u] [S.a0@u] PX
2 2 aa|ui iu|aa VS 1 [S.a1@i~V.c0@i,S.c1@a~V.a0@a,S.c0@a~V.a1@a] 1 [V.c1@u] [S.a0@u] PX
2 2 aa|ui ui|aa VS 1 [S.a1@i~V.c1@i,S.c0@a~V.a0@a,S.c1@a~V.a1@a] 1 [V.c0@u] [S.a0@u] PX
2 2 aa|ui ui|aa VS -1 [S.a1@i~V.c1@i,S.c1@a~V.a0@a,S.c0@a~V.a1@a] 1 [V.c0@u] [S.a0@u] PX
2 2 aa|uu uu|aa VS 1 [S.c0@a~V.a0@a,S.c1@a~V.a1@a] 2 [V.c0@u,V.c1@u] [S.a0@u,S.a1@u] PX
2 2 aa|uu uu|aa VS -1 [S.c1@a~V.a0@a,S.c0@a~V.a1@a] 2 [V.c0@u,V.c1@u] [S.a0@u,S.a1@u] PX
)TERMS";

// ---------------------------------------------------------------------------
// Reading the table.
// ---------------------------------------------------------------------------

struct Slot {
  char operand = 'S';  // 'S' or 'V'
  char kind = 'c';     // 'c' creation, 'a' annihilation
  int position = 0;
  char space = 'i';  // 'i', 'u', 'a'

  bool operator<(const Slot& other) const {
    return std::tie(operand, kind, position) <
           std::tie(other.operand, other.kind, other.position);
  }
};

Slot parse_slot(const std::string& text) {
  Slot slot;
  slot.operand = text[0];
  slot.kind = text[2];
  slot.position = text[3] - '0';
  slot.space = text[text.size() - 1];
  return slot;
}

/// Two operands x {creation, annihilation} x two positions: eight slots, so
/// assignments live in a flat array rather than a map on the hot path.
constexpr int kSlotCount = 8;

int slot_index(const Slot& slot) {
  return (slot.operand == 'V' ? 4 : 0) + (slot.kind == 'a' ? 2 : 0) +
         slot.position;
}

using SlotValues = std::array<int, kSlotCount>;

std::vector<std::string> split_list(std::string text) {
  if (text.size() >= 2) text = text.substr(1, text.size() - 2);
  std::vector<std::string> out;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ','))
    if (!item.empty()) out.push_back(item);
  return out;
}

struct TermRecord {
  int rank_s = 0, rank_v = 0;
  std::string s_block, v_block;
  bool s_first = true;
  double wick_sign = 1.0;
  std::vector<std::pair<Slot, Slot>> deltas;
  int residual_rank = 0;
  std::vector<Slot> residual_cre, residual_ann;
};

std::vector<TermRecord> read_reference_terms() {
  std::stringstream input(kReferenceTerms);
  std::vector<TermRecord> out;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream fields(line);
    std::string ordering, deltas, cre, ann, flags;
    TermRecord term;
    fields >> term.rank_s >> term.rank_v >> term.s_block >> term.v_block >>
        ordering >> term.wick_sign >> deltas >> term.residual_rank >> cre >>
        ann >> flags;
    term.s_first = ordering == "SV";
    for (const std::string& pair : split_list(deltas)) {
      const auto separator = pair.find('~');
      term.deltas.emplace_back(parse_slot(pair.substr(0, separator)),
                               parse_slot(pair.substr(separator + 1)));
    }
    for (const std::string& slot : split_list(cre))
      term.residual_cre.push_back(parse_slot(slot));
    for (const std::string& slot : split_list(ann))
      term.residual_ann.push_back(parse_slot(slot));
    // The emitting tool lists annihilators by particle index; the operator
    // string they came from holds them reversed, and the sign belongs to the
    // string order.
    std::reverse(term.residual_ann.begin(), term.residual_ann.end());
    out.push_back(std::move(term));
  }
  return out;
}

/// Mirrors project_matchings: scale * perm_sign * normA * normB.
double kernel_prefactor(const TermRecord& term) {
  const double scale = term.s_first ? 0.5 : -0.5;
  const double norm_s = term.rank_s == 2 ? 0.25 : 1.0;
  const double norm_v = term.rank_v == 2 ? 0.25 : 1.0;
  return scale * norm_s * norm_v * term.wick_sign;
}

// ---------------------------------------------------------------------------
// A deterministic random system.
// ---------------------------------------------------------------------------

struct System {
  int norb = 0, n_so = 0, n_inactive = 0;
  Eigen::MatrixXd h1, f;
  Eigen::VectorXd g, eps;
  std::vector<double> v;
  sw::SpinOrbitalPartition part;
  sw::SpinBlockedTwoBody blocked;
  std::vector<int> inactive_so, active_so, virtual_so;
};

System build_system(int n_inactive, int n_active, int n_virtual) {
  System sys;
  sys.norb = n_inactive + n_active + n_virtual;
  sys.n_so = 2 * sys.norb;
  sys.n_inactive = n_inactive;
  const int norb = sys.norb;

  std::mt19937 generator(0xC0FFEEu);
  std::uniform_real_distribution<double> coupling(-0.09, 0.09);

  sys.h1 = Eigen::MatrixXd::Zero(norb, norb);
  sys.eps = Eigen::VectorXd::Zero(sys.n_so);
  for (int o = 0; o < norb; ++o) {
    const double energy = -1.0 + 0.55 * o;
    sys.h1(o, o) = energy;
    sys.eps(2 * o) = sys.eps(2 * o + 1) = energy;
  }
  for (int p = 0; p < norb; ++p)
    for (int q = p + 1; q < norb; ++q)
      sys.h1(p, q) = sys.h1(q, p) = coupling(generator);

  sys.g = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(norb) * norb * norb *
                                norb);
  for (int p = 0; p < norb; ++p)
    for (int q = p; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = r; s < norb; ++s) {
          const double value = coupling(generator);
          const int permutations[8][4] = {
              {p, q, r, s}, {q, p, r, s}, {p, q, s, r}, {q, p, s, r},
              {r, s, p, q}, {s, r, p, q}, {r, s, q, p}, {s, r, q, p}};
          for (const auto& index : permutations)
            sys.g(sw::idx4(index[0], index[1], index[2], index[3], norb)) =
                value;
        }

  std::vector<int> inactive, active, virtuals;
  for (int o = 0; o < n_inactive; ++o) inactive.push_back(o);
  for (int o = 0; o < n_active; ++o) active.push_back(n_inactive + o);
  for (int o = 0; o < n_virtual; ++o)
    virtuals.push_back(n_inactive + n_active + o);
  sys.part = sw::make_partition(norb, active, inactive, virtuals);

  sys.blocked = sw::build_two_body_blocked(sys.g, norb);
  sys.f = sw::spin_orbital_one_body(sys.h1, sys.h1, norb);

  sys.v.assign(
      static_cast<std::size_t>(sys.n_so) * sys.n_so * sys.n_so * sys.n_so, 0.0);
  for (int P = 0; P < sys.n_so; ++P)
    for (int Q = 0; Q < sys.n_so; ++Q)
      for (int R = 0; R < sys.n_so; ++R)
        for (int S = 0; S < sys.n_so; ++S)
          sys.v[sw::idx4(P, Q, R, S, sys.n_so)] =
              sw::so_v_from_blocked(sys.blocked, P, Q, R, S);

  for (int P = 0; P < sys.n_so; ++P) {
    if (sys.part.is_inactive[P]) sys.inactive_so.push_back(P);
    if (sys.part.is_active[P]) sys.active_so.push_back(P);
    if (sys.part.is_virtual[P]) sys.virtual_so.push_back(P);
  }
  return sys;
}

// ---------------------------------------------------------------------------
// Determinant algebra over the active space, for the density-fold check.
// ---------------------------------------------------------------------------

struct Ladder {
  std::uint64_t mask = 0;
  int sign = 0;
  bool ok = false;
};

Ladder apply_ladder(std::uint64_t mask, int orbital, bool creation) {
  const std::uint64_t bit = std::uint64_t{1} << orbital;
  const bool occupied = (mask & bit) != 0;
  if (creation == occupied) return {0, 0, false};  // Pauli
  const int below = std::popcount(mask & (bit - 1));
  return {mask ^ bit, (below & 1) ? -1 : 1, true};
}

/// a+_{cre[0]} ... a_{ann[last]} applied to `mask`, rightmost operator first.
Ladder apply_string(std::uint64_t mask, const std::vector<int>& cre,
                    const std::vector<int>& ann) {
  int sign = 1;
  for (auto it = ann.rbegin(); it != ann.rend(); ++it) {
    const Ladder step = apply_ladder(mask, *it, false);
    if (!step.ok) return {0, 0, false};
    mask = step.mask;
    sign *= step.sign;
  }
  for (auto it = cre.rbegin(); it != cre.rend(); ++it) {
    const Ladder step = apply_ladder(mask, *it, true);
    if (!step.ok) return {0, 0, false};
    mask = step.mask;
    sign *= step.sign;
  }
  return {mask, sign, true};
}

/// Action of a scalar + one-body + abab two-body operator on one determinant.
/// The two-body block is the antisymmetric completion of the stored abab
/// spatial block, and the operator is
/// (1/4) sum_{PQRS} v[PQRS] a+_P a+_Q a_R a_S, inverting the four-fold
/// antisymmetrization RetainedOperator::add applies on the way in.
std::vector<double> apply_operator(int n_active_so, double scalar,
                                   const Eigen::MatrixXd& one_body,
                                   const Eigen::VectorXd& abab,
                                   std::uint64_t reference) {
  std::vector<double> out(std::size_t{1} << n_active_so, 0.0);
  out[reference] += scalar;

  for (int P = 0; P < n_active_so; ++P)
    for (int Q = 0; Q < n_active_so; ++Q) {
      if (one_body(P, Q) == 0.0) continue;
      const Ladder result = apply_string(reference, {P}, {Q});
      if (result.ok) out[result.mask] += one_body(P, Q) * result.sign;
    }

  const int n_spatial = n_active_so / 2;
  const auto block = [&](int p, int q, int r, int s) {
    return abab(sw::idx4(p, q, r, s, n_spatial));
  };
  const auto element = [&](int P, int Q, int R, int S) -> double {
    const int p = P >> 1, q = Q >> 1, r = R >> 1, s = S >> 1;
    const int sp = P & 1, sq = Q & 1, sr = R & 1, ss = S & 1;
    if (sp + sq != sr + ss) return 0.0;  // spin is conserved
    if (sp == sq) return block(p, q, r, s) - block(p, q, s, r);
    if (sp == 0) return sr == 0 ? block(p, q, r, s) : -block(p, q, s, r);
    return sr == 1 ? block(q, p, s, r) : -block(q, p, r, s);
  };

  for (int P = 0; P < n_active_so; ++P)
    for (int Q = 0; Q < n_active_so; ++Q)
      for (int R = 0; R < n_active_so; ++R)
        for (int S = 0; S < n_active_so; ++S) {
          const double value = element(P, Q, R, S);
          if (value == 0.0) continue;
          const Ladder result = apply_string(reference, {P, Q}, {R, S});
          if (result.ok) out[result.mask] += 0.25 * value * result.sign;
        }
  return out;
}

// ---------------------------------------------------------------------------
// Evaluating the table.
// ---------------------------------------------------------------------------

struct Evaluated {
  double scalar = 0.0;
  Eigen::MatrixXd one_body;
  Eigen::VectorXd two_body;  // active spatial abab block
};

class Evaluator {
 public:
  Evaluator(const System& system, const sw::RegularizerOptions& regularizer)
      : _system(system), _regularizer(regularizer) {
    _compact.assign(system.n_so, -1);
    for (std::size_t k = 0; k < system.active_so.size(); ++k)
      _compact[system.active_so[k]] = static_cast<int>(k);
  }

  bool changes_external_occupation(int P, int Q) const {
    const auto& part = _system.part;
    return (part.is_inactive[P] - part.is_inactive[Q]) != 0 ||
           (part.is_virtual[P] - part.is_virtual[Q]) != 0;
  }
  bool changes_external_occupation(int P, int Q, int R, int S) const {
    const auto& part = _system.part;
    return (part.is_inactive[P] + part.is_inactive[Q] - part.is_inactive[R] -
            part.is_inactive[S]) != 0 ||
           (part.is_virtual[P] + part.is_virtual[Q] - part.is_virtual[R] -
            part.is_virtual[S]) != 0;
  }

  /// Sums the whole table into scalar, one-body and two-body accumulators.
  Evaluated run(const std::vector<TermRecord>& terms) {
    Evaluated out;
    const int n_active = static_cast<int>(_system.active_so.size());
    const int n_spatial = n_active / 2;
    out.one_body = Eigen::MatrixXd::Zero(n_active, n_active);
    out.two_body = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_spatial) *
                                         n_spatial * n_spatial * n_spatial);
    for (const TermRecord& term : terms) {
      if (term.residual_rank > 2) continue;  // folded, checked separately
      accumulate(term, out);
    }
    return out;
  }

  /// Action of the unfolded rank-three terms on a reference determinant.
  std::vector<double> rank3_on_reference(const std::vector<TermRecord>& terms,
                                         std::uint64_t reference) {
    return operator_on_reference(terms, reference, 3, 3);
  }

  /// Same for the rank <= 2 terms, to self-test `apply_operator`.
  std::vector<double> low_rank_on_reference(
      const std::vector<TermRecord>& terms, std::uint64_t reference) {
    return operator_on_reference(terms, reference, 0, 2);
  }

 private:
  const std::vector<int>& domain(char space) const {
    switch (space) {
      case 'i':
        return _system.inactive_so;
      case 'a':
        return _system.virtual_so;
      default:
        return _system.active_so;
    }
  }

  double s1(int P, int Q) const {
    if (!changes_external_occupation(P, Q)) return 0.0;
    return _system.f(P, Q) * sw::regularized_inverse(
                                 _system.eps(P) - _system.eps(Q), _regularizer);
  }
  double s2(int P, int Q, int R, int S) const {
    if (!changes_external_occupation(P, Q, R, S)) return 0.0;
    const double delta =
        _system.eps(P) + _system.eps(Q) - _system.eps(R) - _system.eps(S);
    return _system.v[sw::idx4(P, Q, R, S, _system.n_so)] *
           sw::regularized_inverse(delta, _regularizer);
  }
  double v1(int P, int Q) const {
    return changes_external_occupation(P, Q) ? _system.f(P, Q) : 0.0;
  }
  double v2(int P, int Q, int R, int S) const {
    return changes_external_occupation(P, Q, R, S)
               ? _system.v[sw::idx4(P, Q, R, S, _system.n_so)]
               : 0.0;
  }

  double operand_value(char operand, int rank, const std::string& block,
                       const SlotValues& value) const {
    int index[4] = {0, 0, 0, 0};
    int n = 0;
    for (int k = 0; k < rank; ++k)
      index[n++] = value[slot_index(Slot{operand, 'c', k, block[k]})];
    const int annihilation_offset = rank + 1;  // skip the '|'
    for (int k = 0; k < rank; ++k)
      index[n++] = value[slot_index(
          Slot{operand, 'a', k, block[annihilation_offset + k]})];
    const bool is_generator = operand == 'S';
    if (rank == 1)
      return is_generator ? s1(index[0], index[1]) : v1(index[0], index[1]);
    return is_generator ? s2(index[0], index[1], index[2], index[3])
                        : v2(index[0], index[1], index[2], index[3]);
  }

  /// Each delta is one summed index, each residual slot one free active index.
  template <class Sink>
  void for_each_assignment(const TermRecord& term, const Sink& sink) {
    std::vector<std::pair<int, int>> variables;  // slot indices
    std::vector<char> spaces;
    for (const auto& delta : term.deltas) {
      variables.emplace_back(slot_index(delta.first), slot_index(delta.second));
      spaces.push_back(delta.first.space);
    }
    for (const Slot& slot : term.residual_cre) {
      variables.emplace_back(slot_index(slot), slot_index(slot));
      spaces.push_back(slot.space);
    }
    for (const Slot& slot : term.residual_ann) {
      variables.emplace_back(slot_index(slot), slot_index(slot));
      spaces.push_back(slot.space);
    }

    SlotValues value{};
    const double prefactor = kernel_prefactor(term);
    std::vector<int> choice(variables.size(), 0);

    const std::function<void(std::size_t)> loop = [&](std::size_t depth) {
      if (depth == variables.size()) {
        for (std::size_t k = 0; k < variables.size(); ++k) {
          value[variables[k].first] = choice[k];
          value[variables[k].second] = choice[k];
        }
        const double contribution =
            prefactor * operand_value('S', term.rank_s, term.s_block, value) *
            operand_value('V', term.rank_v, term.v_block, value);
        if (contribution != 0.0) sink(value, contribution);
        return;
      }
      for (int orbital : domain(spaces[depth])) {
        choice[depth] = orbital;
        loop(depth + 1);
      }
    };
    loop(0);
  }

  /// Mirrors RetainedOperator::add_abab: only (even, odd, even, odd) is stored.
  void add_abab(Evaluated& out, int i, int j, int k, int l,
                double coefficient) const {
    if ((i & 1) == 0 && (j & 1) == 1 && (k & 1) == 0 && (l & 1) == 1) {
      const int n = static_cast<int>(_system.active_so.size()) / 2;
      out.two_body(sw::idx4(i >> 1, j >> 1, k >> 1, l >> 1, n)) += coefficient;
    }
  }

  void accumulate(const TermRecord& term, Evaluated& out) {
    for_each_assignment(term, [&](const SlotValues& value,
                                  double contribution) {
      if (term.residual_rank == 0) {
        out.scalar += contribution;
        return;
      }
      if (term.residual_rank == 1) {
        const int row = _compact[value[slot_index(term.residual_cre[0])]];
        const int column = _compact[value[slot_index(term.residual_ann[0])]];
        out.one_body(row, column) += contribution;
        return;
      }
      // Ascending spin-orbital order with a sign, which is what the
      // kernel's normal_order produces before RetainedOperator::add.
      int c0 = _compact[value[slot_index(term.residual_cre[0])]];
      int c1 = _compact[value[slot_index(term.residual_cre[1])]];
      int a0 = _compact[value[slot_index(term.residual_ann[0])]];
      int a1 = _compact[value[slot_index(term.residual_ann[1])]];
      if (c0 == c1 || a0 == a1) return;  // Pauli
      double weight = contribution;
      if (c0 > c1) {
        std::swap(c0, c1);
        weight = -weight;
      }
      if (a0 > a1) {
        std::swap(a0, a1);
        weight = -weight;
      }
      add_abab(out, c0, c1, a0, a1, +weight);
      add_abab(out, c1, c0, a0, a1, -weight);
      add_abab(out, c0, c1, a1, a0, -weight);
      add_abab(out, c1, c0, a1, a0, +weight);
    });
  }

  std::vector<double> operator_on_reference(
      const std::vector<TermRecord>& terms, std::uint64_t reference,
      int min_rank, int max_rank) {
    const int n = static_cast<int>(_system.active_so.size());
    std::vector<double> out(std::size_t{1} << n, 0.0);
    for (const TermRecord& term : terms) {
      if (term.residual_rank < min_rank || term.residual_rank > max_rank)
        continue;
      for_each_assignment(
          term, [&](const SlotValues& value, double contribution) {
            std::vector<int> cre, ann;
            for (const Slot& slot : term.residual_cre)
              cre.push_back(_compact[value[slot_index(slot)]]);
            for (const Slot& slot : term.residual_ann)
              ann.push_back(_compact[value[slot_index(slot)]]);
            const Ladder result = apply_string(reference, cre, ann);
            if (result.ok) out[result.mask] += contribution * result.sign;
          });
    }
    return out;
  }

  const System& _system;
  sw::RegularizerOptions _regularizer;
  std::vector<int> _compact;
};

/// downfold_blocked returns the block-diagonal mean field plus the commutator.
/// These strip the former so the commutator can be compared on its own.
struct BlockDiagonalBaseline {
  double scalar = 0.0;
  Eigen::MatrixXd one_body;
  Eigen::VectorXd two_body;
};

BlockDiagonalBaseline block_diagonal_baseline(const System& sys,
                                              const Evaluator& evaluator) {
  BlockDiagonalBaseline baseline;
  Eigen::VectorXd occupation = Eigen::VectorXd::Zero(sys.n_so);
  for (int P = 0; P < sys.n_so; ++P)
    if (sys.part.is_inactive[P]) occupation(P) = 1.0;

  const auto bd_v = [&](int P, int Q, int R, int S) {
    return evaluator.changes_external_occupation(P, Q, R, S)
               ? 0.0
               : sys.v[sw::idx4(P, Q, R, S, sys.n_so)];
  };
  for (int P = 0; P < sys.n_so; ++P)
    if (!evaluator.changes_external_occupation(P, P))
      baseline.scalar += sys.f(P, P) * occupation(P);
  for (int P = 0; P < sys.n_so; ++P)
    for (int Q = 0; Q < sys.n_so; ++Q)
      baseline.scalar -= 0.5 * bd_v(P, Q, P, Q) * occupation(P) * occupation(Q);

  const int n_active = static_cast<int>(sys.active_so.size());
  baseline.one_body = Eigen::MatrixXd::Zero(n_active, n_active);
  for (int ci = 0; ci < n_active; ++ci)
    for (int cj = 0; cj < n_active; ++cj) {
      const int i = sys.active_so[ci], j = sys.active_so[cj];
      double fold = 0.0;
      for (int b = 0; b < sys.n_so; ++b)
        fold += bd_v(i, b, b, j) * occupation(b);
      baseline.one_body(ci, cj) =
          (evaluator.changes_external_occupation(i, j) ? 0.0 : sys.f(i, j)) +
          fold;
    }

  const int n_spatial = n_active / 2;
  baseline.two_body = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(n_spatial) * n_spatial * n_spatial * n_spatial);
  for (int p = 0; p < n_spatial; ++p)
    for (int q = 0; q < n_spatial; ++q)
      for (int r = 0; r < n_spatial; ++r)
        for (int s = 0; s < n_spatial; ++s)
          baseline.two_body(sw::idx4(p, q, r, s, n_spatial)) =
              bd_v(sys.active_so[2 * p], sys.active_so[2 * q + 1],
                   sys.active_so[2 * r], sys.active_so[2 * s + 1]);
  return baseline;
}

struct Case {
  int inactive, active, virtuals;
};

// Kept small so the suite stays fast; the derivation is size independent, and
// the point of comparing at all is that it holds where the determinant-space
// oracle cannot run.
const std::vector<Case>& default_cases() {
  static const std::vector<Case> cases{{1, 2, 1}, {2, 3, 3}, {2, 4, 3}};
  return cases;
}

}  // namespace

// ---------------------------------------------------------------------------
// The projected commutator, term by term.
// ---------------------------------------------------------------------------
TEST(Swpt2SymbolicTest, TableIsComplete) {
  EXPECT_EQ(static_cast<int>(read_reference_terms().size()),
            kProjectedTermCount);
}

TEST(Swpt2SymbolicTest, MatchesIndependentSymbolicDerivation) {
  const std::vector<TermRecord> terms = read_reference_terms();
  ASSERT_EQ(static_cast<int>(terms.size()), kProjectedTermCount);

  for (const Case& test_case : default_cases()) {
    const System sys =
        build_system(test_case.inactive, test_case.active, test_case.virtuals);
    const sw::RegularizerOptions regularizer;  // bare denominators
    Evaluator evaluator(sys, regularizer);

    const Evaluated expected = evaluator.run(terms);
    const auto produced = sw::downfold_blocked(sys.f, sys.blocked, sys.eps,
                                               sys.part, regularizer, 0.0);
    const BlockDiagonalBaseline baseline =
        block_diagonal_baseline(sys, evaluator);

    const std::string label = "inactive=" + std::to_string(test_case.inactive) +
                              " active=" + std::to_string(test_case.active) +
                              " virtual=" + std::to_string(test_case.virtuals);
    EXPECT_NEAR(produced.e - baseline.scalar, expected.scalar, 1e-12) << label;
    EXPECT_LT((produced.f_active - baseline.one_body - expected.one_body)
                  .cwiseAbs()
                  .maxCoeff(),
              1e-12)
        << label;
    EXPECT_LT((produced.v_abab - baseline.two_body - expected.two_body)
                  .cwiseAbs()
                  .maxCoeff(),
              1e-12)
        << label;
  }
}

// ---------------------------------------------------------------------------
// A zero reference density switches project_matchings from the
// needs_coincidence shortcut to the folding gate, while gamma == 0 leaves the
// fold itself contributing nothing. The answer must not move, which is the
// claim the shortcut rests on.
// ---------------------------------------------------------------------------
TEST(Swpt2SymbolicTest, CoincidenceShortcutAgreesWithFoldingGate) {
  for (const Case& test_case : default_cases()) {
    const System sys =
        build_system(test_case.inactive, test_case.active, test_case.virtuals);
    const sw::RegularizerOptions regularizer;
    const auto shortcut = sw::downfold_blocked(sys.f, sys.blocked, sys.eps,
                                               sys.part, regularizer, 0.0);
    const auto gated = sw::downfold_blocked(
        sys.f, sys.blocked, sys.eps, sys.part, regularizer, 0.0, {},
        Eigen::MatrixXd::Zero(sys.norb, sys.norb));

    EXPECT_NEAR(gated.e, shortcut.e, 1e-12);
    EXPECT_LT((gated.f_active - shortcut.f_active).cwiseAbs().maxCoeff(),
              1e-12);
    EXPECT_LT((gated.v_abab - shortcut.v_abab).cwiseAbs().maxCoeff(), 1e-12);
  }
}

// ---------------------------------------------------------------------------
// fold_onto_density replaces a three-body term A by A - {A}, keeping every
// 1-RDM contraction. {A} does not annihilate the reference determinant, but its
// only surviving component there is the all-quasi-creator piece, which is a
// pure triple excitation. So on an idempotent reference the folded (<= 2-body)
// operator reproduces A exactly up to double excitations, and loses exactly the
// triple.
// ---------------------------------------------------------------------------
TEST(Swpt2SymbolicTest, DensityFoldIsExactThroughDoubleExcitations) {
  const std::vector<TermRecord> terms = read_reference_terms();

  // Only a four-orbital active space leaves enough empty spin-orbitals for a
  // triple excitation to exist at all, so it is the case that exercises what
  // the truncation drops; the small one guards the cheap path.
  const std::vector<Case> cases{{1, 2, 1}, {2, 4, 3}};
  for (const Case& test_case : cases) {
    const System sys =
        build_system(test_case.inactive, test_case.active, test_case.virtuals);
    const sw::RegularizerOptions regularizer;
    Evaluator evaluator(sys, regularizer);
    const int n_active = static_cast<int>(sys.active_so.size());

    // Two doubly occupied active orbitals: a three-body operator annihilates
    // any determinant carrying fewer than three active electrons, which would
    // make this test pass vacuously.
    Eigen::MatrixXd density = Eigen::MatrixXd::Zero(sys.norb, sys.norb);
    std::uint64_t reference = 0;
    int occupied_active = 0;
    for (int o = 0; o < sys.norb; ++o) {
      if (sys.part.is_inactive[2 * o]) density(o, o) = 2.0;
      if (sys.part.is_active[2 * o] && occupied_active < 2) {
        density(o, o) = 2.0;
        const int compact = 2 * (o - sys.n_inactive);
        reference |=
            (std::uint64_t{1} << compact) | (std::uint64_t{1} << (compact + 1));
        ++occupied_active;
      }
    }

    const auto gated = sw::downfold_blocked(
        sys.f, sys.blocked, sys.eps, sys.part, regularizer, 0.0, {},
        Eigen::MatrixXd::Zero(sys.norb, sys.norb));
    const auto folded = sw::downfold_blocked(
        sys.f, sys.blocked, sys.eps, sys.part, regularizer, 0.0, {}, density);

    const std::vector<double> from_kernel = apply_operator(
        n_active, folded.e - gated.e, folded.f_active - gated.f_active,
        folded.v_abab - gated.v_abab, reference);
    const std::vector<double> from_table =
        evaluator.rank3_on_reference(terms, reference);

    double through_doubles = 0.0;
    double signal = 0.0;
    int nonzero = 0;
    for (std::size_t k = 0; k < from_kernel.size(); ++k) {
      const int level =
          std::popcount(static_cast<std::uint64_t>(k) ^ reference) / 2;
      if (level <= 2)
        through_doubles =
            std::max(through_doubles, std::abs(from_kernel[k] - from_table[k]));
      signal = std::max(signal, std::abs(from_table[k]));
      if (std::abs(from_table[k]) > 1e-12) ++nonzero;
    }

    const std::string label = "inactive=" + std::to_string(test_case.inactive) +
                              " active=" + std::to_string(test_case.active) +
                              " virtual=" + std::to_string(test_case.virtuals);
    EXPECT_GT(nonzero, 0) << "vacuous: the reference is annihilated; " << label;
    EXPECT_GT(signal, 1e-6) << "vacuous: nothing to compare; " << label;
    EXPECT_LT(through_doubles, 1e-12) << label;
  }
}

// Guards the abab reconstruction used above, so a bug in the test helper cannot
// be mistaken for a kernel defect. Size independent, so one system suffices.
TEST(Swpt2SymbolicTest, AbabReconstructionRoundTrips) {
  const std::vector<TermRecord> terms = read_reference_terms();
  const System sys = build_system(1, 2, 1);
  const sw::RegularizerOptions regularizer;
  Evaluator evaluator(sys, regularizer);
  const int n_active = static_cast<int>(sys.active_so.size());

  std::uint64_t reference = 0;
  int occupied_active = 0;
  for (int o = 0; o < sys.norb; ++o)
    if (sys.part.is_active[2 * o] && occupied_active < 2) {
      const int compact = 2 * (o - sys.n_inactive);
      reference |=
          (std::uint64_t{1} << compact) | (std::uint64_t{1} << (compact + 1));
      ++occupied_active;
    }

  const Evaluated low = evaluator.run(terms);
  const std::vector<double> direct =
      evaluator.low_rank_on_reference(terms, reference);
  const std::vector<double> reconstructed = apply_operator(
      n_active, low.scalar, low.one_body, low.two_body, reference);

  double reconstruction = 0.0;
  double signal = 0.0;
  for (std::size_t k = 0; k < direct.size(); ++k) {
    reconstruction =
        std::max(reconstruction, std::abs(direct[k] - reconstructed[k]));
    signal = std::max(signal, std::abs(direct[k]));
  }
  EXPECT_GT(signal, 1e-6) << "vacuous: nothing to reconstruct";
  EXPECT_LT(reconstruction, 1e-12);
}
