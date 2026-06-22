/// ducc_active_test.cpp — Active-space SeQuant->Python integration (4 spaces)
///
/// Spaces (make_sr_spaces):
///   o = inactive occupied, i = active occupied (hole), a = active virtual
///   (particle), g = inactive virtual. m={o,i}=occupied, x={i,a}=active,
///   p={o,i,a,g}=complete.
///
/// Pipeline:
///   1. Build H = h^p_q {a+_p a_q} + 1/4 v^{pq}_{rs} {a+_p a+_q a_s a_r}
///      over COMPLETE space p
///   2. FWickTheorem normal-orders w.r.t. Fermi vacuum (fills m={o,i})
///   3. Resolve s/delta Kronecker tensors via index replacement
///   4. Relabel the FREE (NormalOperator) legs from complete p -> active x.
///      This projects the downfolded Hamiltonian onto the active space.
///      The COEFFICIENT keeps occupied-m summations (incl. inactive frozen
///      core mean field) — these are the "contractions with external indices".
///   5. Strip NormalOperator -> gamma coefficient (active free legs)
///   6. Export gamma_1, gamma_2 as numpy einsum
///   7. Python: build active-space integrals, evaluate, gamma->chi,
///      FCI vs PySCF CASCI on the original Hamiltonian.

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
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

using namespace sequant;
using namespace sequant::mbpt;
using OpType = NormalOperator<Statistics::FermiDirac>;

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

// Relabel the FREE (NormalOperator) legs from complete p -> active x.
// Each NormalOp leg index keeps its ordinal but moves to the active space.
ExprPtr relabel_free_to_active(const ExprPtr& expr,
                               std::shared_ptr<IndexSpaceRegistry> isr) {
  const auto& active = isr->retrieve(L"x");

  auto relabel_term = [&](const ExprPtr& term) -> ExprPtr {
    container::map<Index, Index> repl;
    term->visit([&](const ExprPtr& e) {
      if (e->is<OpType>()) {
        auto& nop = e->as<OpType>();
        for (auto it = nop.cbegin(); it != nop.cend(); ++it) {
          Index old = it->index();
          if (old.ordinal().has_value()) {
            repl[old] = Index(active, old.ordinal().value());
          }
        }
      }
    }, true);
    if (repl.empty()) return term->clone();
    return transform_expr(term, repl);
  };

  if (expr->is<Sum>()) {
    ExprPtr result;
    for (auto& term : *expr) {
      auto r = relabel_term(term);
      result = result ? result + r : r;
    }
    simplify(result);
    return result;
  }
  auto r = relabel_term(expr);
  simplify(r);
  return r;
}

// GENERAL FILTER (works for BCH order >= 1): keep ONLY terms whose
// NormalOperator legs are all ACTIVE after Wick contraction. For each free
// (NormalOp) leg of a term:
//   - if leg space has NO overlap with active {i,a}  -> strictly inactive (o,g)
//       => DROP the whole term (operator acts on external space, not active)
//   - if leg space is a subset of active (i, a, or x) -> keep as-is
//   - if leg space is broader than active (complete p) -> project to active x
// At level 0 (complete-p legs) nothing is dropped and every leg is projected to
// active, reproducing relabel_free_to_active. At BCH order >= 1, sigma_ext
// introduces inactive (o,g) legs; terms with uncontracted inactive legs are
// dropped, leaving the active-only downfolded operators.
ExprPtr filter_and_project_active(const ExprPtr& expr,
                                  std::shared_ptr<IndexSpaceRegistry> isr) {
  const auto& active = isr->retrieve(L"x");
  const auto active_type = active.type();

  auto process_term = [&](const ExprPtr& term) -> ExprPtr {
    container::map<Index, Index> repl;
    bool drop = false;
    term->visit([&](const ExprPtr& e) {
      if (e->is<OpType>()) {
        auto& nop = e->as<OpType>();
        for (auto it = nop.cbegin(); it != nop.cend(); ++it) {
          Index leg = it->index();
          auto leg_type = leg.space().type();
          auto inter = leg_type.intersection(active_type);
          if (!inter) {
            drop = true;  // strictly inactive (o, g)
          } else if (leg_type == inter) {
            // leg subset of active (i, a, or x): keep as-is
          } else {
            // leg broader than active (complete p): project to active x
            if (leg.ordinal().has_value())
              repl[leg] = Index(active, leg.ordinal().value());
          }
        }
      }
    }, true);
    if (drop) return ex<Constant>(0);  // dropped marker
    if (repl.empty()) return term->clone();
    return transform_expr(term, repl);
  };

  if (expr->is<Sum>()) {
    ExprPtr result;
    for (auto& term : *expr) {
      auto r = process_term(term);
      if (r->is<Constant>()) continue;  // dropped term
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
          if (idx < nc) bra_idx.push_back(it->index());
          else ket_idx.push_back(it->index());
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
    expr->visit([&](const ExprPtr& e) {
      if (e->is<OpType>()) rank = (int)e->as<OpType>().ncreators();
    }, true);
    return (rank == target_rank) ? expr->clone() : ex<Constant>(0);
  }
  for (auto& term : *expr) {
    int rank = -1;
    term->visit([&](const ExprPtr& e) {
      if (e->is<OpType>()) rank = (int)e->as<OpType>().ncreators();
    }, true);
    if (rank == target_rank) result = result ? result + term : term->clone();
  }
  return result ? result : ex<Constant>(0);
}

std::string export_einsum(const Tensor& result_tensor, ExprPtr coeff,
                          const IndexSpaceRegistry& reg) {
  ResultExpr result_expr(result_tensor, coeff);
  auto tree = to_export_tree(result_expr);

  NumPyEinsumGeneratorContext ctx;
  ctx.enable_rewriting(false);
  // Active (x) free legs, occupied (m) summation legs.
  auto active = reg.retrieve(L"x");
  auto occupied = reg.retrieve(L"m");
  ctx.set_shape(active, "nact");
  ctx.set_shape(occupied, "nocc");
  ctx.set_tag(active, "x");
  ctx.set_tag(occupied, "m");

  NumPyEinsumGenerator gen;
  export_expression(tree, gen, ctx);
  return gen.get_generated_code();
}

// Export a SCALAR (rank-0) coefficient as numpy einsum.
//
// Proper BCH contractions (order >= 1) fully contract H_N against sigma_ext, so
// all indices pair legs of DIFFERENT tensors -> proper inter-tensor contractions
// (NO self-traces). The generator's rank-0 Tensor result handles these correctly,
// INCLUDING multi-step contractions with intermediate tensors, e.g.
//     I_mmee += np.einsum('ma,bc->mcab', f_me, t_em)
//     g0     += 1/2 * np.einsum('mabc,bcma->', I_mmee, t_eemm)
//
// SELF-TRACES (h^m_m, v^{mn}_{mn}) arise ONLY in the level-0 reference energy
// <Phi|H|Phi> (= SCF energy). binarize mishandles double self-traces
// ('mama->mmaa'); we do NOT extract those via einsum -- the SCF energy is a
// trivial constant, subtracted manually. Thus this exporter is correct for ALL
// BCH order >= 1 scalar contributions.
std::string export_scalar(ExprPtr coeff, const IndexSpaceRegistry& reg) {
  Tensor scalar_result(L"g0", bra{}, ket{});  // rank-0 result
  ResultExpr result_expr(scalar_result, coeff);
  auto tree = to_export_tree(result_expr);

  NumPyEinsumGeneratorContext ctx;
  ctx.enable_rewriting(false);
  ctx.set_shape(reg.retrieve(L"m"), "nocc");
  ctx.set_shape(reg.retrieve(L"a"), "nvir");
  ctx.set_shape(reg.retrieve(L"x"), "nact");
  ctx.set_tag(reg.retrieve(L"m"), "m");
  ctx.set_tag(reg.retrieve(L"a"), "e");
  ctx.set_tag(reg.retrieve(L"x"), "x");

  NumPyEinsumGenerator gen;
  export_expression(tree, gen, ctx);
  return gen.get_generated_code();
}

int main() {
  auto reg = make_sr_spaces();
  set_default_context({.index_space_registry_shared_ptr = reg,
                       .vacuum = Vacuum::SingleProduct});

  // Build H with complete-space p operators
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
  auto g0_raw = extract_by_rank(H_N, -1);  // scalar (rank-0) = <Phi|H|Phi>

  // Resolve s/delta, then FILTER active-only operator terms (general; works
  // for BCH order >= 1) and project complete-p legs onto active.
  auto g1_clean = filter_and_project_active(resolve_deltas(g1_raw, reg), reg);
  auto g2_clean = filter_and_project_active(resolve_deltas(g2_raw, reg), reg);
  // Scalar has no free legs -> only resolve deltas (no relabeling)
  auto g0_clean = resolve_deltas(g0_raw, reg);

  std::wcout << L"=== gamma_1 (active free legs) ===" << std::endl;
  std::wcout << to_latex(g1_clean) << std::endl;
  std::wcout << L"\n=== gamma_2 (active free legs) ===" << std::endl;
  std::wcout << to_latex(g2_clean) << std::endl;

  Tensor g1_result, g2_result;
  auto g1_coeff = strip_normalop(g1_clean, g1_result);
  auto g2_coeff = strip_normalop(g2_clean, g2_result);
  simplify(g1_coeff);
  simplify(g2_coeff);

  std::wcout << L"\n=== gamma_1 result + coeff ===" << std::endl;
  std::wcout << to_latex(g1_result) << L" = " << to_latex(g1_coeff) << std::endl;
  std::wcout << L"\n=== gamma_2 result + coeff ===" << std::endl;
  std::wcout << to_latex(g2_result) << L" = " << to_latex(g2_coeff) << std::endl;
  std::wcout << L"\n=== gamma_0 scalar = <Phi|H|Phi> ===" << std::endl;
  std::wcout << to_latex(g0_clean) << std::endl;

  std::string g1_einsum, g2_einsum, g0_einsum;
  try {
    g1_einsum = export_einsum(g1_result, g1_coeff, *reg);
  } catch (const std::exception& e) {
    g1_einsum = std::string("# g1 export failed: ") + e.what();
  }
  try {
    g2_einsum = export_einsum(g2_result, g2_coeff, *reg);
  } catch (const std::exception& e) {
    g2_einsum = std::string("# g2 export failed: ") + e.what();
  }
  try {
    g0_einsum = export_scalar(g0_clean, *reg);
  } catch (const std::exception& e) {
    g0_einsum = std::string("# g0 export failed: ") + e.what();
  }

  std::ofstream ofs("/tmp/ducc_active_einsum.txt");
  ofs << "# === gamma_0 scalar einsum ===\n" << g0_einsum << "\n";
  ofs << "# === gamma_1 einsum ===\n" << g1_einsum << "\n";
  ofs << "# === gamma_2 einsum ===\n" << g2_einsum << "\n";
  ofs.close();
  std::wcout << L"\n[einsum written to /tmp/ducc_active_einsum.txt]" << std::endl;

  return 0;
}
