// Multiply-add contraction, as the code generator performs it.
//
// PTX's mul, add and sub without a rounding modifier "may be optimized
// aggressively by the code generator": ptxas contracts a mul and the add or
// sub that consumes its product into one fma, rounded once. An RTX 3060 does
// so under these rules, measured from results (a fused d is exactly fma(a, b,
// c), an unfused one the product rounded and then the sum):
//   - neither instruction carries a rounding modifier (.rn included), f32 and
//     f64 alike, .ftz forms too;
//   - the product's every use is such an add or sub: then each one fuses,
//     the multiply duplicated where there are several; a product also used
//     any other way is fused nowhere (a use that is dead code, such as a copy
//     nothing reads, ptxas removes first; this pass does not, so such a pair
//     stays unfused here where the card would fuse it);
//   - either operand order, and both directions of sub: c - a*b is
//     fma(-a, b, c), a*b - c is fma(a, b, -c);
//   - with a product on both sides, the first operand's is fused;
//   - not across a branch.
// Kernels see the difference whenever nvcc leaves contraction to ptxas, as it
// did in PolyBench's ADI (x - y*a in kernels 3 and 6): a one-ulp difference
// in X that the recurrence carried through a third of the output.
//
// Only what was measured is done. Within a basic block, a product whose
// multiply is the only definition of its register in the function, whose
// operands are not redefined before the add, and whose uses are all
// qualifying adds and subs later in the same block; everything else is left
// as written, which is what the ISA's own semantics give anyway.
#include "vgpu/ptx/contract.hpp"

#include <cstdlib>
#include <map>
#include <tuple>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vgpu/ptx/regalloc.hpp"

namespace vgpu::ptx {
namespace {

uint32_t key(const Reg& r) { return r.id * 2 + (r.wide ? 1u : 0u); }

const Reg* reg_of(const Operand& o) {
  const auto* r = std::get_if<RegOperand>(&o);
  return r ? &r->reg : nullptr;
}

bool contractible(const Instr& ins, const OpFloatBin& op) {
  return !ins.has_pred && !op.round_explicit && !op.sat && !op.xorsign_abs && !op.nan_propagate &&
         op.ty.is_float() && (op.ty.bits == 32 || op.ty.bits == 64);
}

bool ends_block(const Instr& ins) {
  return std::holds_alternative<OpBra>(ins.op) || std::holds_alternative<OpBrx>(ins.op) ||
         std::holds_alternative<OpRet>(ins.op) || std::holds_alternative<OpCall>(ins.op) || std::holds_alternative<OpBar>(ins.op);
}

}  // namespace

size_t contract_mul_add(EntryFn& fn) {
  // VGPU_PTX_CONTRACT=0 runs mul and add as written, rounding each.
  if (const char* e = std::getenv("VGPU_PTX_CONTRACT"); e && e[0] == '0') return 0;
  auto& body = fn.body;
  const size_t n = body.size();
  if (n == 0) return 0;

  // Basic blocks: a branch target starts one, and a branch, return, call or
  // barrier ends one.
  std::vector<uint32_t> block(n, 0);
  {
    std::vector<bool> leader(n + 1, false);
    leader[0] = true;
    for (size_t i = 0; i < n; ++i) {
      if (const auto* b = std::get_if<OpBra>(&body[i].op))
        if (b->target < n) leader[b->target] = true;
      if (const auto* b = std::get_if<OpBrx>(&body[i].op))
        for (size_t t : b->targets)
          if (t < n) leader[t] = true;
      if (ends_block(body[i])) leader[i + 1] = true;
    }
    uint32_t id = 0;
    for (size_t i = 0; i < n; ++i) {
      if (i && leader[i]) ++id;
      block[i] = id;
    }
  }

  // Every definition and use in the function, by register.
  std::unordered_map<uint32_t, std::vector<size_t>> defs_of, uses_of;
  std::vector<std::vector<uint32_t>> defs(n), uses(n);
  for (size_t i = 0; i < n; ++i) {
    instr_registers(body[i], defs[i], uses[i]);
    for (uint32_t k : defs[i]) defs_of[k].push_back(i);
    for (uint32_t k : uses[i]) uses_of[k].push_back(i);
  }

  // The multiply whose product an operand is, if it may be fused into
  // instruction `at`; otherwise -1.
  auto fusable_mul = [&](const Operand& o, size_t at, const Type& ty, bool ftz) -> long {
    const Reg* r = reg_of(o);
    if (!r) return -1;
    const auto d = defs_of.find(key(*r));
    if (d == defs_of.end() || d->second.size() != 1) return -1;
    const size_t mi = d->second[0];
    if (mi >= at || block[mi] != block[at]) return -1;
    const auto* mul = std::get_if<OpFloatBin>(&body[mi].op);
    if (!mul || mul->op != FloatBinOp::Mul || !contractible(body[mi], *mul) || mul->ty.bits != ty.bits ||
        mul->ftz != ftz)
      return -1;
    // Every use of the product is a qualifying add or sub after it, in its block.
    for (size_t u : uses_of[key(*r)]) {
      const auto* add = std::get_if<OpFloatBin>(&body[u].op);
      if (u <= mi || block[u] != block[mi] || !add ||
          (add->op != FloatBinOp::Add && add->op != FloatBinOp::Sub) || !contractible(body[u], *add) ||
          add->ty.bits != ty.bits || add->ftz != ftz)
        return -1;
    }
    // The multiply's operands must still hold their values at `at`.
    for (const Operand* src : {&mul->a, &mul->b})
      if (const Reg* sr = reg_of(*src)) {
        const auto sd = defs_of.find(key(*sr));
        if (sd != defs_of.end())
          for (size_t w : sd->second)
            if (w > mi && w < at) return -1;
      }
    return static_cast<long>(mi);
  };

  // Decide every fusion before rewriting any: a product feeding two adds
  // fuses into both, and the second must still see the first as an add.
  struct Fusion { size_t at; size_t mul; bool product_first; };
  std::vector<Fusion> plan;
  for (size_t i = 0; i < n; ++i) {
    const auto* add = std::get_if<OpFloatBin>(&body[i].op);
    if (!add || (add->op != FloatBinOp::Add && add->op != FloatBinOp::Sub) || !contractible(body[i], *add))
      continue;
    // The first operand's product is the one fused when both are products.
    long mi = fusable_mul(add->a, i, add->ty, add->ftz);
    bool product_first = true;
    if (mi < 0) {
      mi = fusable_mul(add->b, i, add->ty, add->ftz);
      product_first = false;
    }
    if (mi < 0) continue;
    plan.push_back({i, static_cast<size_t>(mi), product_first});
  }
  size_t fused = 0;
  for (const Fusion& fu : plan) {
    const size_t i = fu.at;
    const bool product_first = fu.product_first;
    const auto* add = std::get_if<OpFloatBin>(&body[i].op);
    const auto& mul = std::get<OpFloatBin>(body[fu.mul].op);
    OpFma f;
    f.ty = add->ty;
    f.dst = add->dst;
    f.a = mul.a;
    f.b = mul.b;
    f.c = product_first ? add->b : add->a;
    f.ftz = add->ftz;
    if (add->op == FloatBinOp::Sub) {
      if (product_first) f.neg_c = true;   // a*b - c
      else f.neg_ab = true;                // c - a*b
    }
    body[i].op = f;
    ++fused;
  }
  return fused;
}

}  // namespace vgpu::ptx

namespace vgpu::ptx {

void mark_nan_order(EntryFn& fn) {
  auto& body = fn.body;
  std::unordered_map<uint32_t, long> last_def;   // register key -> index of its latest definition so far
  std::map<std::tuple<int, int, long, long, long, int>, long> cse_rank;   // an expression over defined values -> where it was first computed
  std::vector<uint32_t> defs, uses;
  // How late an operand was defined: -1 for a register nothing has written (a parameter, say), a very large number for a constant.
  const auto rank = [&](const Operand& o) -> long {
    if (const Reg* r = reg_of(o)) {
      const auto it = last_def.find(key(*r));
      return it == last_def.end() ? -1 : it->second;
    }
    return 1L << 40;
  };
  // Whether register `k`, written by instruction `at`, is read before it is written again (looking ahead a bounded distance).
  std::vector<uint32_t> fdefs, fuses;
  const auto read_later = [&](size_t at, uint32_t k) {
    for (size_t j = at + 1; j < body.size() && j < at + 4096; ++j) {
      fdefs.clear();
      fuses.clear();
      instr_registers(body[j], fdefs, fuses);
      for (uint32_t u : fuses)
        if (u == k) return true;
      if (!body[j].has_pred)
        for (uint32_t d : fdefs)
          if (d == k) return false;
    }
    return false;
  };
  for (size_t i = 0; i < body.size(); ++i) {
    if (auto* f = std::get_if<OpFloatBin>(&body[i].op); f && f->ty.bits == 64 && f->ty.is_float()) {
      // a - <constant> is a + <the constant negated> to the code generator, which folds the negation into the constant: a NaN constant
      // that is the answer comes back with its sign flipped (an RTX 3060: sub.f64 of 0dFFF8000000000008 returns 0x7ff8000000000008).
      if (f->op == FloatBinOp::Sub && f->round == FRound::Nearest && !f->nan_propagate)
        if (auto* im = std::get_if<ImmFloatBits>(&f->b); im && im->width == 64) {
          im->bits ^= uint64_t{1} << 63;
          f->op = FloatBinOp::Add;
        }
      f->nan_a_first = rank(f->a) > rank(f->b);
    }
    else if (auto* m = std::get_if<OpFma>(&body[i].op); m && m->ty.bits == 64)
      m->nan_swap_ab = rank(m->a) > rank(m->b);
    // ptxas numbers a value once: an expression it has already computed from the same values (a common subexpression), and min or max
    // of a value with itself, give the earlier result's definition, not a new one.
    if (!body[i].has_pred) {
      const auto sorted = [](long x, long y) { return x < y ? std::pair{x, y} : std::pair{y, x}; };
      const Reg* dst = nullptr;
      std::tuple<int, int, long, long, long, int> cse{};
      long same = -2;
      if (const auto* f = std::get_if<OpFloatBin>(&body[i].op); f && f->ty.bits == 64 && f->ty.is_float() && reg_of(f->a) && reg_of(f->b) &&
                                                                    f->round == FRound::Nearest && !f->nan_propagate && !f->sat && !f->xorsign_abs) {
        dst = &f->dst;
        const long ra = rank(f->a), rb = rank(f->b);
        const bool commutes = f->op != FloatBinOp::Sub && f->op != FloatBinOp::Div;
        const auto pr = commutes ? sorted(ra, rb) : std::pair{ra, rb};
        cse = {1, static_cast<int>(f->op), pr.first, pr.second, 0, f->ftz ? 1 : 0};
        if ((f->op == FloatBinOp::Min || f->op == FloatBinOp::Max) && ra == rb && ra >= 0) same = ra;
      } else if (const auto* m = std::get_if<OpFma>(&body[i].op); m && m->ty.bits == 64 && reg_of(m->a) && reg_of(m->b) && reg_of(m->c) &&
                                                                     m->round == FRound::Nearest && !m->sat) {
        dst = &m->dst;
        const auto pr = sorted(rank(m->a), rank(m->b));
        cse = {2, (m->neg_ab ? 1 : 0) | (m->neg_c ? 2 : 0), pr.first, pr.second, rank(m->c), m->ftz ? 1 : 0};
      }
      if (dst && dst->id != kNoReg) {
        const auto seen = cse_rank.find(cse);
        const bool known_operands = std::get<2>(cse) >= 0 && std::get<3>(cse) >= 0;
        if (same >= 0) {
          last_def[key(*dst)] = same;
          continue;
        }
        if (known_operands && seen != cse_rank.end()) {
          last_def[key(*dst)] = seen->second;
          continue;
        }
        if (known_operands && read_later(i, key(*dst))) cse_rank[cse] = static_cast<long>(i);   // (a result nothing reads is dead code, which ptxas removes first)
      }
    }
    defs.clear();
    uses.clear();
    instr_registers(body[i], defs, uses);
    // A register-to-register mov is a copy ptxas propagates: the copy has the rank of what it copies (so a mov of a register onto
    // itself changes nothing, and a copy made late is no later than its source).
    const auto* mv = std::get_if<OpMov>(&body[i].op);
    if (mv && !body[i].has_pred && mv->ty.bits == 64 && reg_of(mv->src) && mv->dst.id != kNoReg) {
      last_def[key(mv->dst)] = rank(mv->src);
      continue;
    }
    for (uint32_t k : defs) last_def[k] = static_cast<long>(i);
  }
}

}  // namespace vgpu::ptx
