/// ducc_active_openshell.cpp — Open-shell (spin-traced) active-space DUCC
///
/// Companion to ducc_active_test.cpp. That program produces the downfolded
/// active-space gamma operators in the FULL SPIN-ORBITAL basis (indices run
/// over 2N spin-orbitals). This program takes the SAME downfolded gamma
/// operators and applies SeQuant's general `spintrace` (spinfree = false) to
/// reduce them to SPIN-BLOCK (spatial) form, valid for restricted references
/// including restricted open-shell (ROHF, high-spin) — exactly the regime MACIS
/// supports.
///
/// Closed-shell `closed_shell_spintrace` collapses to a single spin-free
/// expression but is only valid for doubly-occupied singlet references. The
/// general `spintrace` with `spinfree = false` instead emits one expression per
/// external spin-case (alpha-alpha, alpha-beta, beta-beta, ...), which maps
/// one-to-one onto qdk-chemistry's native three-spin-block Hamiltonian
/// representation (h1_alpha/h1_beta; h2_aaaa/h2_aabb/h2_bbbb).
///
/// Pipeline (identical to ducc_active_test.cpp through step 5):
///   1. Build H over the COMPLETE space p
///   2. FWickTheorem normal-orders w.r.t. Fermi vacuum
///   3. Resolve s/delta Kronecker tensors
///   4. filter_and_project_active: keep active-only operator terms
///   5. strip_normalop -> gamma coefficient with active free legs
///   6. NEW: wrap gamma as a ResultExpr and call spintrace(.., spinfree=false)
///      -> a vector of per-spin-case ResultExprs (spatial spin blocks)
///   7. Export each spin block as numpy einsum. Per-spin index spaces are
///      tagged so the generated tensor names encode the spin block, e.g.
///      gamma_XX (1-body alpha), gamma_XxXx (2-body alpha-beta). Shapes use
///      spatial dims (nocc/nact), so the einsum runs on N-dim tensors.
///
/// Output: /tmp/ducc_openshell_einsum.txt

#include <SeQuant/core/context.hpp>
#include <SeQuant/core/export/export.hpp>
#include <SeQuant/core/export/python_einsum.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/expressions/result_expr.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
#include <SeQuant/core/expressions/variable.hpp>
#include <SeQuant/core/io/shorthands.hpp>
#include <SeQuant/core/op.hpp>
#include <SeQuant/core/reserved.hpp>
#include <SeQuant/core/utility/expr.hpp>
#include <SeQuant/core/wick.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/spin.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

using namespace sequant;
using namespace sequant::mbpt;
using OpType = NormalOperator<Statistics::FermiDirac>;

// ───────────────────────── shared helpers (verbatim) ─────────────────────────

// Resolve overlap (s) and Kronecker (delta) tensors in a single Product
ExprPtr resolve_deltas_in_product(const ExprPtr& term,
                                  std::shared_ptr<IndexSpaceRegistry> isr) {
  using sequant::reserved::kronecker_label;
  using sequant::reserved::overlap_label;

  if (!term->is<Product>()) return term->clone();
  auto& prod = term->as<Product>();

  container::map<Index, Index> repl;
  ExprPtr non_delta;

  for (size_t i = 0; i < prod.size(); ++i) {
    auto& factor = prod.at(i);
    if (factor->is<Tensor>()) {
      auto& t = factor->as<Tensor>();
      if (t.label() == overlap_label() || t.label() == kronecker_label()) {
        Index b = t.bra().at(0);
        Index k = t.ket().at(0);
        const auto& common = isr->intersection(b.space(), k.space());
        if (k.space().type() == common.type()) {
          repl[b] = k;
        } else {
          repl[k] = b;
        }
        continue;
      }
    }
    non_delta = non_delta ? non_delta * factor : factor->clone();
  }

  if (prod.scalar() != 1.0) {
    non_delta = non_delta ? ex<Constant>(prod.scalar()) * non_delta
                          : ex<Constant>(prod.scalar());
  }
  if (!non_delta) non_delta = ex<Constant>(1);

  auto resolve_chain = [&](Index x) {
    int guard = 0;
    while (repl.count(x) && guard++ < 100) {
      Index next = repl[x];
      if (next == x) break;
      x = next;
    }
    return x;
  };
  container::map<Index, Index> final_repl;
  for (auto& [from, to] : repl) final_repl[from] = resolve_chain(to);

  if (final_repl.empty()) return non_delta;
  return transform_expr(non_delta, final_repl);
}

ExprPtr resolve_deltas(const ExprPtr& expr,
                       std::shared_ptr<IndexSpaceRegistry> isr) {
  if (expr->is<Sum>()) {
    ExprPtr result;
    for (auto& term : *expr) {
      auto cleaned = resolve_deltas_in_product(term, isr);
      result = result ? result + cleaned : cleaned;
    }
    simplify(result);
    return result;
  }
  auto r = resolve_deltas_in_product(expr, isr);
  simplify(r);
  return r;
}

// GENERAL FILTER (works for BCH order >= 1): keep ONLY terms whose
// NormalOperator legs are all ACTIVE; project complete-p legs onto active x.
ExprPtr filter_and_project_active(const ExprPtr& expr,
                                  std::shared_ptr<IndexSpaceRegistry> isr) {
  const auto& active = isr->retrieve(L"x");
  const auto active_type = active.type();

  auto process_term = [&](const ExprPtr& term) -> ExprPtr {
    container::map<Index, Index> repl;
    bool drop = false;
    term->visit(
        [&](const ExprPtr& e) {
          if (e->is<OpType>()) {
            auto& nop = e->as<OpType>();
            for (auto it = nop.cbegin(); it != nop.cend(); ++it) {
              Index leg = it->index();
              auto leg_type = leg.space().type();
              auto inter = leg_type.intersection(active_type);
              if (!inter) {
                drop = true;
              } else if (leg_type == inter) {
                // leg subset of active: keep
              } else {
                if (leg.ordinal().has_value())
                  repl[leg] = Index(active, leg.ordinal().value());
              }
            }
          }
        },
        true);
    if (drop) return ex<Constant>(0);
    if (repl.empty()) return term->clone();
    return transform_expr(term, repl);
  };

  if (expr->is<Sum>()) {
    ExprPtr result;
    for (auto& term : *expr) {
      auto r = process_term(term);
      if (r->is<Constant>()) continue;
      result = result ? result + r : r;
    }
    if (!result) return ex<Constant>(0);
    simplify(result);
    return result;
  }
  auto r = process_term(expr);
  if (r->is<Constant>()) return ex<Constant>(0);
  simplify(r);
  return r;
}

// Strip NormalOperator, extract bra/ket indices, return coefficient expression
ExprPtr strip_normalop(const ExprPtr& expr, Tensor& result_tensor) {
  auto strip_one = [&](const ExprPtr& term) -> ExprPtr {
    if (!term->is<Product>()) return term->clone();
    auto& prod = term->as<Product>();
    ExprPtr coeff;
    std::vector<Index> bra_idx, ket_idx;
    for (size_t i = 0; i < prod.size(); ++i) {
      auto& factor = prod.at(i);
      if (factor->is<OpType>()) {
        auto& nop = factor->as<OpType>();
        size_t nc = nop.ncreators(), idx = 0;
        for (auto it = nop.cbegin(); it != nop.cend(); ++it, ++idx) {
          if (idx < nc)
            bra_idx.push_back(it->index());
          else
            ket_idx.push_back(it->index());
        }
        continue;
      }
      coeff = coeff ? coeff * factor : factor->clone();
    }
    if (prod.scalar() != 1.0)
      coeff = coeff ? ex<Constant>(prod.scalar()) * coeff
                    : ex<Constant>(prod.scalar());
    if (!coeff) coeff = ex<Constant>(1);
    if (result_tensor.braket_indices().empty() && !bra_idx.empty())
      result_tensor = Tensor(L"gamma", bra(bra_idx), ket(ket_idx));
    return coeff;
  };

  if (expr->is<Sum>()) {
    ExprPtr result;
    for (auto& term : *expr) {
      auto c = strip_one(term);
      result = result ? result + c : c;
    }
    return result;
  }
  return strip_one(expr);
}

ExprPtr extract_by_rank(const ExprPtr& expr, int target_rank) {
  ExprPtr result;
  if (!expr->is<Sum>()) {
    int rank = -1;
    expr->visit(
        [&](const ExprPtr& e) {
          if (e->is<OpType>()) rank = (int)e->as<OpType>().ncreators();
        },
        true);
    return (rank == target_rank) ? expr->clone() : ex<Constant>(0);
  }
  for (auto& term : *expr) {
    int rank = -1;
    term->visit(
        [&](const ExprPtr& e) {
          if (e->is<OpType>()) rank = (int)e->as<OpType>().ncreators();
        },
        true);
    if (rank == target_rank) result = result ? result + term : term->clone();
  }
  return result ? result : ex<Constant>(0);
}

// ───────────────────────── spin-aware einsum export ─────────────────────────

// Collect every IndexSpace that occurs in an expression (and a result tensor).
void collect_spaces(const ExprPtr& expr, std::set<IndexSpace>& spaces) {
  if (!expr) return;
  expr->visit(
      [&](const ExprPtr& e) {
        if (e->is<Tensor>()) {
          for (const Index& idx : e->as<Tensor>().const_indices())
            spaces.insert(idx.space());
        }
      },
      true);
}

// Configure a NumPyEinsumGeneratorContext for a spin-traced ResultExpr.
// Each spin-labeled space gets:
//   - a SHAPE = spatial dim ("nocc" for occupied-type, "nact" for active-type)
//   - a TAG encoding (space, spin): occupied alpha='M' beta='m';
//     active alpha='X' beta='x'. Tags are concatenated into tensor names, so
//     e.g. gamma_XX (1-body alpha), v_mXmX (occ-beta/act-alpha), etc.
void configure_spin_context(NumPyEinsumGeneratorContext& ctx,
                            const ResultExpr& re,
                            std::shared_ptr<IndexSpaceRegistry> reg) {
  const auto occ_type = reg->retrieve(L"m").type();
  const auto act_type = reg->retrieve(L"x").type();

  std::set<IndexSpace> spaces;
  collect_spaces(re.expression(), spaces);
  for (const Index& idx : re.indices()) spaces.insert(idx.space());

  for (const IndexSpace& sp : spaces) {
    const bool is_occ = (sp.type() == occ_type);
    const bool is_act = (sp.type() == act_type);
    // spin: alpha or beta (spin-free spaces should not occur post-spintrace)
    Spin s = Spin::any;
    try {
      s = to_spin(sp.qns());
    } catch (...) {
      s = Spin::any;
    }
    const bool alpha = (s == Spin::alpha);

    if (is_occ) {
      ctx.set_shape(sp, "nocc");
      ctx.set_tag(sp, alpha ? "M" : "m");
    } else if (is_act) {
      ctx.set_shape(sp, "nact");
      ctx.set_tag(sp, alpha ? "X" : "x");
    } else {
      // fallback: treat anything else as active-sized
      ctx.set_shape(sp, "nact");
      ctx.set_tag(sp, alpha ? "Y" : "y");
    }
  }
}

// Export a single spin-case ResultExpr as numpy einsum source.
std::string export_spin_block(const ResultExpr& re,
                              std::shared_ptr<IndexSpaceRegistry> reg) {
  NumPyEinsumGeneratorContext ctx;
  ctx.enable_rewriting(false);
  configure_spin_context(ctx, re, reg);

  auto tree = to_export_tree(re);
  NumPyEinsumGenerator gen;
  export_expression(tree, gen, ctx);
  return gen.get_generated_code();
}

// Spin of an index (alpha/beta) from its space quantum numbers.
Spin index_spin(const Index& idx) {
  try {
    return to_spin(idx.space().qns());
  } catch (...) {
    return Spin::any;
  }
}

// Spin-trace a downfolded gamma operator into explicit spin blocks.
//
// Uses the general `spintrace(expr, groups, spinfree=false)`, which keeps spin
// labels and PRESERVES operator-coefficient normalization (unlike
// open_shell_spintrace, whose (k,k)-moment convention introduces a spurious
// factor on pure-external 1-body terms). The single spin-labeled sum is then
// split by the spin pattern of the external (active free) legs into one
// ResultExpr per spin block (alpha-alpha, alpha-beta, ...).
std::vector<ResultExpr> spintrace_blocks(
    const ExprPtr& gamma_clean, std::shared_ptr<IndexSpaceRegistry> reg) {
  const auto active_type = reg->retrieve(L"x").type();

  Tensor result_tensor;
  auto coeff = strip_normalop(gamma_clean, result_tensor);
  simplify(coeff);
  if (result_tensor.braket_indices().empty()) return {};

  std::vector<Index> bra_idx(result_tensor.bra().begin(),
                             result_tensor.bra().end());
  std::vector<Index> ket_idx(result_tensor.ket().begin(),
                             result_tensor.ket().end());
  std::vector<Index> externals(bra_idx);
  externals.insert(externals.end(), ket_idx.begin(), ket_idx.end());

  std::vector<std::size_t> ext_ord;
  for (const Index& e : externals) ext_ord.push_back(e.ordinal().value());

  // particle groups {bra_i, ket_i}
  container::svector<container::svector<Index>> groups;
  for (std::size_t j = 0; j < bra_idx.size(); ++j)
    groups.push_back(container::svector<Index>{bra_idx[j], ket_idx[j]});

  auto traced = spintrace(coeff, groups, /*spinfree_index_spaces=*/false);
  simplify(traced);

  // Split terms by the spin pattern of the external legs (active-type indices,
  // identified by ordinal).
  std::map<std::vector<int>, ExprPtr> blocks;
  auto add_term = [&](const ExprPtr& term) {
    std::map<std::size_t, Spin> ord_spin;
    term->visit(
        [&](const ExprPtr& e) {
          if (e->is<Tensor>()) {
            for (const Index& idx : e->as<Tensor>().const_indices()) {
              if (idx.space().type() == active_type && idx.ordinal().has_value())
                ord_spin[idx.ordinal().value()] = index_spin(idx);
            }
          }
        },
        true);
    std::vector<int> key;
    for (std::size_t o : ext_ord)
      key.push_back(ord_spin[o] == Spin::beta ? 1 : 0);
    blocks[key] = blocks[key] ? blocks[key] + term->clone() : term->clone();
  };
  if (traced->is<Sum>())
    for (const auto& term : *traced) add_term(term);
  else
    add_term(traced);

  std::vector<ResultExpr> out;
  for (auto& [key, expr] : blocks) {
    std::vector<Index> sbra, sket;
    for (std::size_t i = 0; i < externals.size(); ++i) {
      Index labeled = key[i] == 1 ? make_spinbeta(externals[i])
                                  : make_spinalpha(externals[i]);
      if (i < bra_idx.size())
        sbra.push_back(labeled);
      else
        sket.push_back(labeled);
    }
    Tensor lhs(L"gamma", bra(std::move(sbra)), ket(std::move(sket)));
    simplify(expr);
    out.emplace_back(lhs, expr);
  }
  return out;
}

// Spin-trace one downfolded gamma operator and append the einsum for every
// spin block to `out`.
void spintrace_and_export(const ExprPtr& gamma_clean, int rank,
                          std::shared_ptr<IndexSpaceRegistry> reg,
                          std::ostream& out) {
  auto blocks = spintrace_blocks(gamma_clean, reg);
  if (blocks.empty()) {
    out << "# (rank " << rank << ": empty / no active operator)\n";
    return;
  }
  out << "# ==== gamma rank " << rank << ": " << blocks.size()
      << " spin block(s) ====\n";
  for (const auto& re : blocks) {
    try {
      out << export_spin_block(re, reg) << "\n";
    } catch (const std::exception& e) {
      out << "# export failed: " << e.what() << "\n";
    }
  }
}

int main() {
  auto reg = make_sr_spaces();
  set_default_context({.index_space_registry_shared_ptr = reg,
                       .vacuum = Vacuum::SingleProduct});

  // Build H with complete-space p operators (identical to spin-orbital pipeline)
  auto h_coeff = ex<Tensor>(L"h", bra{L"p_1"}, ket{L"p_2"}, Symmetry::Nonsymm);
  auto H1_ops = fcrex(L"p_1") * fannx(L"p_2");
  auto v_coeff = ex<Tensor>(L"v", bra{L"p_1", L"p_3"}, ket{L"p_2", L"p_4"},
                            Symmetry::Antisymm);
  auto H2_ops = fcrex(L"p_1") * fcrex(L"p_3") * fannx(L"p_4") * fannx(L"p_2");

  auto H1_no = FWickTheorem{H1_ops}.full_contractions(false).compute();
  auto H2_no = FWickTheorem{H2_ops}.full_contractions(false).compute();
  simplify(H1_no);
  simplify(H2_no);

  auto H_N = h_coeff * H1_no + ex<Constant>(rational(1, 4)) * v_coeff * H2_no;
  simplify(H_N);

  auto g1_raw = extract_by_rank(H_N, 1);
  auto g2_raw = extract_by_rank(H_N, 2);

  auto g1_clean = filter_and_project_active(resolve_deltas(g1_raw, reg), reg);
  auto g2_clean = filter_and_project_active(resolve_deltas(g2_raw, reg), reg);

  // Show the spin-block structure (symbolic proof).
  auto show_cases = [&](const ExprPtr& gamma_clean, const std::wstring& name) {
    auto blocks = spintrace_blocks(gamma_clean, reg);
    std::wcout << L"=== " << name << L" spin blocks (" << blocks.size()
               << L") ===" << std::endl;
    for (const auto& re : blocks)
      std::wcout << to_latex(re.result_as_tensor(L"gamma")) << L" = "
                 << to_latex(re.expression()) << std::endl;
  };
  show_cases(g1_clean, L"gamma_1");
  std::wcout << std::endl;
  show_cases(g2_clean, L"gamma_2");

  std::ofstream ofs("/tmp/ducc_openshell_einsum.txt");
  ofs << "# Open-shell (spin-traced) active-space DUCC einsum\n";
  ofs << "# tags: occ alpha=M beta=m ; active alpha=X beta=x\n";
  ofs << "# shapes: nocc (occupied spatial), nact (active spatial)\n\n";
  spintrace_and_export(g1_clean, 1, reg, ofs);
  spintrace_and_export(g2_clean, 2, reg, ofs);
  ofs.close();

  std::wcout << L"\n[einsum written to /tmp/ducc_openshell_einsum.txt]"
             << std::endl;
  return 0;
}
