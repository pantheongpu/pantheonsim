#include "vgpu/ptx/ast.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace vgpu::ptx {

std::string Type::str() const {
  if (kind == Kind::Pred) return ".pred";
  // bfloat16 is its own kind, and it was missing here: it fell through to the
  // 'b' default and printed as ".b16", so every diagnostic naming a bf16 type
  // said "bits" instead. The switch is exhaustive now so the compiler catches
  // the next kind that is added.
  if (kind == Kind::BF) return ".bf" + std::to_string(bits);
  char k = 'b';
  switch (kind) {
    case Kind::B: k = 'b'; break;
    case Kind::U: k = 'u'; break;
    case Kind::S: k = 's'; break;
    case Kind::F: k = 'f'; break;
    case Kind::BF: break;   // handled above
    case Kind::Pred: break;
  }
  return std::string(".") + k + std::to_string(bits);
}

// Opcode interning. A process-wide table, because instruction ids outlive the
// module that first mentioned an opcode: a launch counts by id and reports by
// name, and the two must agree across modules.
//
// Guarded by a mutex because modules can be loaded from more than one thread,
// but read on the hot path only through the id already stored in the Instr --
// nothing looks a name up while a kernel is running.
uint16_t intern_opcode(const std::string& mnemonic) {
  static std::mutex mu;
  static std::unordered_map<std::string, uint16_t> ids;
  std::lock_guard<std::mutex> lock(mu);
  auto it = ids.find(mnemonic);
  if (it != ids.end()) return it->second;
  auto& names = const_cast<std::vector<std::string>&>(opcode_names());
  if (names.empty()) names.push_back("<none>");  // id 0 means "not set"
  if (names.size() >= 0xFFFF) return 0;          // absurdly many opcodes; stop growing
  const uint16_t id = static_cast<uint16_t>(names.size());
  names.push_back(mnemonic);
  ids.emplace(mnemonic, id);
  return id;
}

const std::vector<std::string>& opcode_names() {
  static std::vector<std::string> names;
  return names;
}

// The narrow-register or immediate operand a MicroOp source can be; false for
// anything else (a 64-bit register, a special register, a symbol).
static bool micro_source(const Operand& o, uint32_t& value, bool& imm) {
  if (const auto* r = std::get_if<RegOperand>(&o)) {
    if (r->reg.wide || r->reg.id == kNoReg) return false;
    value = r->reg.id;
    imm = false;
    return true;
  }
  if (const auto* i = std::get_if<ImmInt>(&o)) {
    value = static_cast<uint32_t>(i->value);
    imm = true;
    return true;
  }
  if (const auto* f = std::get_if<ImmFloatBits>(&o)) {
    value = static_cast<uint32_t>(f->bits);
    imm = true;
    return true;
  }
  return false;
}

void build_micro(EntryFn& fn) {
  fn.micro.assign(fn.body.size(), MicroOp{});
  for (size_t pc = 0; pc < fn.body.size(); ++pc) {
    const Instr& ins = fn.body[pc];
    MicroOp u;
    u.opcode_id = ins.opcode_id;
    fn.micro[pc].opcode_id = ins.opcode_id;   // every entry carries it: the loop counts by it
    if (ins.has_pred) {
      if (ins.pred.wide || ins.pred.id == kNoReg || ins.pred.id > 0xFFFF) continue;
      u.flags |= MicroOp::kHasPred | (ins.pred_negated ? MicroOp::kPredNeg : 0);
      u.pred = static_cast<uint16_t>(ins.pred.id);
    }
    auto source = [&](size_t i, const Operand& o) {
      bool imm = false;
      if (!micro_source(o, u.src[i], imm)) return false;
      if (imm) u.src_imm |= static_cast<uint8_t>(1u << i);
      return true;
    };
    auto narrow_dst = [&](const Reg& r) {
      if (r.wide || r.id == kNoReg) return false;
      u.dst = r.id;
      return true;
    };
    auto wide_dst = [&](const Reg& r) {
      if (!r.wide || r.id == kNoReg) return false;
      u.dst = r.id;
      u.flags |= MicroOp::kDstWide;
      return true;
    };
    // A source of a 64-bit form: a register of either file (a narrow one reads zero-extended), or an integer
    // immediate that is its 32-bit value sign-extended -- the offsets and shift counts of address arithmetic.
    auto source64 = [&](size_t i, const Operand& o) {
      if (const auto* r = std::get_if<RegOperand>(&o)) {
        if (r->reg.id == kNoReg) return false;
        u.src[i] = r->reg.id;
        if (r->reg.wide) u.src_imm |= static_cast<uint8_t>(1u << (3 + i));
        return true;
      }
      if (const auto* im = std::get_if<ImmInt>(&o)) {
        if (im->value < INT32_MIN || im->value > INT32_MAX) return false;
        u.src[i] = static_cast<uint32_t>(static_cast<int32_t>(im->value));
        u.src_imm |= static_cast<uint8_t>(1u << i);
        return true;
      }
      return false;
    };
    bool ok = false;
    if (const auto* op = std::get_if<OpIntBin>(&ins.op)) {
      if (op->ty.bits == 64) {
        // fast_int_bin64's conditions.
        ok = !(op->carry_in || op->carry_out || op->sat || op->op == IntBinOp::Div || op->op == IntBinOp::Rem) &&
             wide_dst(op->dst) && source64(0, op->a) && source64(1, op->b);
        if (ok) {
          u.kind = MicroOp::Int64;
          u.sub = static_cast<uint8_t>(op->op);
          if (op->ty.is_signed()) u.flags |= MicroOp::kSigned;
        }
      } else {
        ok = !(op->carry_in || op->carry_out || op->sat || op->ty.bits != 32 || op->op == IntBinOp::Div ||
               op->op == IntBinOp::Rem) &&
             narrow_dst(op->dst) && source(0, op->a) && source(1, op->b);
        if (ok) {
          u.kind = MicroOp::Int32;
          u.sub = static_cast<uint8_t>(op->op);
          if (op->ty.is_signed()) u.flags |= MicroOp::kSigned;
        }
      }
    } else if (const auto* op = std::get_if<OpCvt>(&ins.op); op && op->dst_ty.is_float() && !op->dst_ty.is_bfloat() &&
                                                             !op->src_ty.is_real() &&
                                                             op->src_ty.kind != Type::Kind::Pred) {
      // An integer of at most 32 bits to f32 or f64, rounded to nearest: exact in a double, so the general path's
      // double-then-float conversion is one rounding. Anything else (a 64-bit source, a directed rounding mode, f16)
      // stays on the general path.
      const Type& st = op->src_ty;
      const Type& dt = op->dst_ty;
      const auto* src = std::get_if<RegOperand>(&op->src);
      ok = src && src->reg.id != kNoReg && !src->reg.wide && op->dst.id != kNoReg && st.bits >= 1 && st.bits <= 32 &&
           (op->round == Round::None || op->round == Round::Rn) &&
           ((dt.bits == 32 && !op->dst.wide) || (dt.bits == 64 && op->dst.wide));
      if (ok) {
        u.kind = MicroOp::CvtI2F;
        u.sub = static_cast<uint8_t>(st.bits);
        u.src[0] = src->reg.id;
        u.dst = op->dst.id;
        if (st.is_signed()) u.flags |= MicroOp::kSigned;
        if (dt.bits == 64) u.flags |= MicroOp::kDstWide;
      }
    } else if (const auto* op = std::get_if<OpCvt>(&ins.op)) {
      // fast_cvt_int's conditions: an integer to an integer, no rounding, no saturation.
      const Type& st = op->src_ty;
      const Type& dt = op->dst_ty;
      const auto* src = std::get_if<RegOperand>(&op->src);
      ok = src && src->reg.id != kNoReg && op->dst.id != kNoReg && !st.is_real() && !dt.is_real() &&
           st.kind != Type::Kind::Pred && dt.kind != Type::Kind::Pred && op->round == Round::None && !op->sat &&
           !(dt.is_signed() && dt.bits < 32) && st.bits >= 1 && st.bits <= 64 && dt.bits >= 1 && dt.bits <= 64;
      if (ok) {
        u.kind = MicroOp::Cvt;
        u.sub = static_cast<uint8_t>(st.bits);
        u.src[1] = dt.bits;
        if (st.is_signed()) u.flags |= MicroOp::kSigned;
        source64(0, op->src);
        u.dst = op->dst.id;
        if (op->dst.wide) u.flags |= MicroOp::kDstWide;
      }
    } else if (const auto* op = std::get_if<OpMulHi>(&ins.op)) {
      ok = op->ty.bits == 32 && !op->ty.is_real() && narrow_dst(op->dst) && source(0, op->a) && source(1, op->b);
      if (ok) {
        u.kind = MicroOp::MulHi32;
        if (op->ty.is_signed()) u.flags |= MicroOp::kSigned;
      }
    } else if (const auto* op = std::get_if<OpNeg>(&ins.op)) {
      ok = op->ty.bits == 32 && !op->ty.is_bfloat() && narrow_dst(op->dst) && source(0, op->src);
      if (ok) {
        u.kind = MicroOp::Sign32;
        u.sub = 1;
        if (op->ty.is_float()) u.flags |= MicroOp::kFloat | (op->ftz ? MicroOp::kFtz : 0);
      }
    } else if (const auto* op = std::get_if<OpAbs>(&ins.op)) {
      ok = op->ty.bits == 32 && !op->ty.is_bfloat() && narrow_dst(op->dst) && source(0, op->src);
      if (ok) {
        u.kind = MicroOp::Sign32;
        u.sub = 0;
        if (op->ty.is_float()) u.flags |= MicroOp::kFloat | (op->ftz ? MicroOp::kFtz : 0);
      }
    } else if (const auto* op = std::get_if<OpMulWide>(&ins.op)) {
      ok = op->src_bits == 32 && wide_dst(op->dst) && source(0, op->a) && source(1, op->b);
      if (ok) {
        u.kind = MicroOp::MulWide32;
        if (op->is_signed) u.flags |= MicroOp::kSigned;
      }
    } else if (const auto* op = std::get_if<OpMadLo>(&ins.op)) {
      ok = !(op->carry_in || op->carry_out || op->ty.bits != 32) && narrow_dst(op->dst) && source(0, op->a) &&
           source(1, op->b) && source(2, op->c);
      if (ok) u.kind = MicroOp::MadLo32;
    } else if (const auto* op = std::get_if<OpMov>(&ins.op)) {
      ok = op->ty.bits == 32 && narrow_dst(op->dst) && source(0, op->src);
      if (ok) u.kind = MicroOp::Mov32;
    } else if (const auto* op = std::get_if<OpSetp>(&ins.op); op && op->ty.bits == 64) {
      // 64-bit integer and f64 comparisons, with registers or small immediates (an f64 literal is 64 bits).
      ok = !op->ty.is_bfloat() && !op->has_bop && !op->has_q && !op->packed && !op->ftz && narrow_dst(op->dst) &&
           source64(0, op->a) && source64(1, op->b);
      if (ok && !op->ty.is_float() &&
          !(op->cmp == CmpOp::Eq || op->cmp == CmpOp::Ne || op->cmp == CmpOp::Lt || op->cmp == CmpOp::Le ||
            op->cmp == CmpOp::Gt || op->cmp == CmpOp::Ge))
        ok = false;
      if (ok && op->ty.is_float() && (u.src_imm & 7)) ok = false;
      if (ok) {
        u.kind = MicroOp::Setp64;
        u.sub = static_cast<uint8_t>(op->cmp);
        if (op->ty.is_float()) u.flags |= MicroOp::kFloat;
        if (op->ty.is_signed()) u.flags |= MicroOp::kSigned;
      }
    } else if (const auto* op = std::get_if<OpSetp>(&ins.op)) {
      ok = op->ty.bits == 32 && !op->ty.is_bfloat() && !op->has_bop && !op->has_q && !op->packed && !op->ftz &&
           narrow_dst(op->dst) && source(0, op->a) && source(1, op->b);
      // The integer comparisons the fast path knows: the six ordered ones.
      if (ok && !op->ty.is_float() &&
          !(op->cmp == CmpOp::Eq || op->cmp == CmpOp::Ne || op->cmp == CmpOp::Lt || op->cmp == CmpOp::Le ||
            op->cmp == CmpOp::Gt || op->cmp == CmpOp::Ge))
        ok = false;
      if (ok) {
        u.kind = MicroOp::Setp32;
        u.sub = static_cast<uint8_t>(op->cmp);
        if (op->ty.is_float()) u.flags |= MicroOp::kFloat;
        if (op->ty.is_signed()) u.flags |= MicroOp::kSigned;
      }
    } else if (const auto* op = std::get_if<OpSelp>(&ins.op)) {
      ok = op->ty.bits == 32 && !op->pred.wide && op->pred.id != kNoReg && narrow_dst(op->dst) &&
           source(0, op->a) && source(1, op->b);
      if (ok) {
        u.kind = MicroOp::Selp32;
        u.src[2] = op->pred.id;
      }
    } else if (const auto* op = std::get_if<OpNot>(&ins.op)) {
      ok = !op->logical && op->ty.bits == 32 && narrow_dst(op->dst) && source(0, op->src);
      if (ok) u.kind = MicroOp::Not32;
    } else if (const auto* op = std::get_if<OpShf>(&ins.op)) {
      ok = narrow_dst(op->dst) && source(0, op->a) && source(1, op->b) && source(2, op->c);
      if (ok) {
        u.kind = MicroOp::Shf32;
        if (op->left) u.flags |= MicroOp::kLeft;
        if (op->wrap) u.flags |= MicroOp::kWrap;
      }
    } else if (const auto* op = std::get_if<OpBra>(&ins.op)) {
      ok = op->target <= 0xFFFFFFFFu;
      if (ok) {
        u.kind = MicroOp::Bra;
        u.src[0] = static_cast<uint32_t>(op->target);
      }
    } else if (const auto* op = std::get_if<OpFma>(&ins.op)) {
      ok = !(op->round != FRound::Nearest || op->sat || op->ftz || op->neg_ab || op->neg_c) && op->ty.bits == 32 &&
           narrow_dst(op->dst) && source(0, op->a) && source(1, op->b) && source(2, op->c);
      if (ok) u.kind = MicroOp::Fma32;
    } else if (const auto* op = std::get_if<OpFloatBin>(&ins.op)) {
      ok = !(op->round != FRound::Nearest || op->nan_propagate || op->sat || op->xorsign_abs || op->ftz ||
             op->approx || op->ty.bits != 32 || !op->ty.is_float()) &&
           narrow_dst(op->dst) && source(0, op->a) && source(1, op->b);
      if (ok) {
        u.kind = MicroOp::Float32;
        u.sub = static_cast<uint8_t>(op->op);
      }
    }
    if (ok) fn.micro[pc] = u;
  }
}

}  // namespace vgpu::ptx
