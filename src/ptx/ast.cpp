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
  fn.micro_consts.clear();
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
    // A source of an f64 form: a 64-bit register, or a constant (kept whole, in the function's table of them).
    auto source_f64 = [&](size_t i, const Operand& o) {
      if (const auto* r = std::get_if<RegOperand>(&o)) {
        if (!r->reg.wide || r->reg.id == kNoReg) return false;
        u.src[i] = r->reg.id;
        u.src_imm |= static_cast<uint8_t>(8u << i);
        return true;
      }
      uint64_t bits = 0;
      if (const auto* f = std::get_if<ImmFloatBits>(&o)) bits = f->bits;
      else if (const auto* im = std::get_if<ImmInt>(&o)) bits = static_cast<uint64_t>(im->value);
      else return false;
      u.src[i] = static_cast<uint32_t>(fn.micro_consts.size());
      fn.micro_consts.push_back(bits);
      u.src_imm |= static_cast<uint8_t>(1u << i);
      return true;
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
      } else if (op->ty.bits == 16) {
        // int_bin's 16-bit cases, the result cut to 16 bits (division is left to the general path).
        ok = !(op->carry_in || op->carry_out || op->sat || op->op == IntBinOp::Div || op->op == IntBinOp::Rem) &&
             !op->ty.is_float() && narrow_dst(op->dst) && source(0, op->a) && source(1, op->b);
        if (ok) {
          u.kind = MicroOp::Int16;
          u.sub = static_cast<uint8_t>(op->op);
          if (op->ty.is_signed()) u.flags |= MicroOp::kSigned;
        }
      } else if ((op->carry_in || op->carry_out) && op->ty.bits == 32 && !op->sat &&
                 (op->op == IntBinOp::Add || op->op == IntBinOp::Sub)) {
        // add/sub with the condition-code carry (exec_carry_add_sub).
        ok = narrow_dst(op->dst) && source(0, op->a) && source(1, op->b);
        if (ok) {
          u.kind = MicroOp::CarryAddSub32;
          u.sub = static_cast<uint8_t>(static_cast<unsigned>(op->op) | (op->carry_in ? 64u : 0u) | (op->carry_out ? 128u : 0u));
        }
      } else {
        ok = !(op->carry_in || op->carry_out || op->sat || op->ty.bits != 32) && narrow_dst(op->dst) &&
             source(0, op->a) && source(1, op->b);
        if (ok) {
          u.kind = MicroOp::Int32;
          u.sub = static_cast<uint8_t>(op->op);
          if (op->ty.is_signed()) u.flags |= MicroOp::kSigned;
        }
      }
    } else if (const auto* op = std::get_if<OpCvt>(&ins.op);
               op && op->src_ty.is_float() && !op->src_ty.is_bfloat() && (op->src_ty.bits == 32 || op->src_ty.bits == 64) &&
               !op->sat && !op->ftz && !op->relu && !op->pzo) {
      // A real to an integer of 32 or 64 bits (convert's float-to-integer tail), or f32 <-> f64 and the round-to-integral
      // forms that keep the type (to nearest, for the narrowing one). Only a register source.
      const Type& st = op->src_ty;
      const Type& dt = op->dst_ty;
      const auto* src = std::get_if<RegOperand>(&op->src);
      const bool src_ok = src && src->reg.id != kNoReg && src->reg.wide == (st.bits == 64) && op->dst.id != kNoReg;
      const bool integral_round = op->round == Round::Rni || op->round == Round::Rzi || op->round == Round::Rmi ||
                                  op->round == Round::Rpi;
      if (src_ok && dt.kind != Type::Kind::Pred && !dt.is_real() && (dt.bits == 32 || dt.bits == 64) &&
          op->dst.wide == (dt.bits == 64)) {
        ok = true;
        u.kind = MicroOp::CvtF2I;
        u.sub = static_cast<uint8_t>(op->round);
        u.src[0] = src->reg.id;
        u.src[1] = dt.bits;
        u.dst = op->dst.id;
        if (st.bits == 64) u.src_imm |= 8;
        if (dt.is_signed()) u.flags |= MicroOp::kSigned;
        if (dt.bits == 64) u.flags |= MicroOp::kDstWide;
      } else if (src_ok && dt.is_float() && !dt.is_bfloat() && (dt.bits == 32 || dt.bits == 64) &&
                 op->dst.wide == (dt.bits == 64) &&
                 ((dt.bits != st.bits && (op->round == Round::None || op->round == Round::Rn)) ||
                  (dt.bits == st.bits && integral_round))) {
        ok = true;
        u.kind = MicroOp::CvtF2F;
        u.sub = static_cast<uint8_t>(op->round);
        u.src[0] = src->reg.id;
        u.src[1] = dt.bits;
        u.dst = op->dst.id;
        if (st.bits == 64) u.src_imm |= 8;
        if (dt.bits == 64) u.flags |= MicroOp::kDstWide;
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
    } else if (const auto* op = std::get_if<OpShfl>(&ins.op)) {
      // shfl.sync with narrow operands. The predicate output, if any, takes the `pred` field, so such a shfl has no guard.
      const bool has_p = op->pred_dst.id != kNoReg;
      ok = narrow_dst(op->dst) && source(0, op->a) && source(1, op->b) && source(2, op->c) &&
           (!has_p || (!op->pred_dst.wide && op->pred_dst.id <= 0xFFFF && !ins.has_pred));
      if (ok) {
        u.kind = MicroOp::Shfl32;
        u.sub = static_cast<uint8_t>(static_cast<unsigned>(op->mode) | (has_p ? 4u : 0u));
        if (has_p) u.pred = static_cast<uint16_t>(op->pred_dst.id);
      }
    } else if (const auto* op = std::get_if<OpVote>(&ins.op)) {
      ok = !op->src.wide && op->src.id != kNoReg && !op->dst.wide && op->dst.id != kNoReg;
      if (ok) {
        u.kind = MicroOp::Vote;
        u.sub = static_cast<uint8_t>(static_cast<unsigned>(op->mode) | (op->negate_src ? 4u : 0u) | (op->ballot ? 8u : 0u));
        u.src[0] = op->src.id;
        u.dst = op->dst.id;
      }
    } else if (const auto* op = std::get_if<OpMovUnpack>(&ins.op); op && op->ty.bits == 64 && op->dsts.size() == 2) {
      // mov.b64 {lo, hi}, %rd: the two halves of a 64-bit register (either may be `_`).
      const auto* src = std::get_if<RegOperand>(&op->src);
      ok = src && src->reg.wide && src->reg.id != kNoReg && !op->dsts[0].wide && !op->dsts[1].wide;
      if (ok) {
        u.kind = MicroOp::Unpack64;
        u.src[0] = src->reg.id;
        u.dst = op->dsts[0].id;
        u.src[1] = op->dsts[1].id;
      }
    } else if (const auto* op = std::get_if<OpMovPack>(&ins.op); op && op->ty.bits == 64 && op->srcs.size() == 2) {
      // mov.b64 %rd, {lo, hi}.
      ok = wide_dst(op->dst) && source(0, op->srcs[0]) && source(1, op->srcs[1]);
      if (ok) u.kind = MicroOp::Pack64;
    } else if (const auto* op = std::get_if<OpPopcClz>(&ins.op)) {
      // popc.b32 and clz.b32, into a narrow register.
      ok = op->ty.bits == 32 && narrow_dst(op->dst) && source(0, op->src);
      if (ok) {
        u.kind = MicroOp::PopcClz32;
        u.sub = op->popc ? 1 : 0;
      }
    } else if (const auto* op = std::get_if<OpMulWide>(&ins.op)) {
      ok = op->src_bits == 32 && wide_dst(op->dst) && source(0, op->a) && source(1, op->b);
      if (ok) {
        u.kind = MicroOp::MulWide32;
        if (op->is_signed) u.flags |= MicroOp::kSigned;
      }
    } else if (const auto* op = std::get_if<OpMadLo>(&ins.op)) {
      ok = op->ty.bits == 32 && narrow_dst(op->dst) && source(0, op->a) && source(1, op->b) && source(2, op->c);
      if (ok && (op->carry_in || op->carry_out)) {
        u.kind = MicroOp::CarryMadLo32;
        u.sub = static_cast<uint8_t>((op->carry_in ? 64u : 0u) | (op->carry_out ? 128u : 0u));
      } else if (ok) {
        u.kind = MicroOp::MadLo32;
      }
    } else if (const auto* op = std::get_if<OpMov>(&ins.op)) {
      if (op->ty.bits == 64) {
        ok = wide_dst(op->dst) && source64(0, op->src);
        if (ok) {
          u.kind = MicroOp::Mov64;
        } else if (wide_dst(op->dst) && (std::holds_alternative<ImmFloatBits>(op->src) || std::holds_alternative<ImmInt>(op->src))) {
          // A constant that does not fit a sign-extended 32 bits (an f64 literal, say).
          u.src_imm = 0;
          ok = source_f64(0, op->src);
          if (ok) u.kind = MicroOp::MovImm64;
        }
      } else if (const auto* sr = std::get_if<SregOperand>(&op->src)) {
        // The ids a kernel starts by reading (the same values sreg_value gives them).
        switch (sr->reg) {
          case Sreg::TidX: case Sreg::TidY: case Sreg::TidZ: case Sreg::NtidX: case Sreg::NtidY: case Sreg::NtidZ:
          case Sreg::CtaidX: case Sreg::CtaidY: case Sreg::CtaidZ: case Sreg::NctaidX: case Sreg::NctaidY:
          case Sreg::NctaidZ: case Sreg::LaneId:
            ok = op->ty.bits == 32 && narrow_dst(op->dst);
            break;
          default: break;
        }
        if (ok) {
          u.kind = MicroOp::MovSreg;
          u.sub = static_cast<uint8_t>(sr->reg);
        }
      } else if (const auto* sy = std::get_if<SymbolOperand>(&op->src)) {
        // nvcc's warpSize: the one predefined constant (resolve_symbol).
        ok = sy->name == "WARP_SZ" && op->ty.bits == 32 && narrow_dst(op->dst);
        if (ok) {
          u.kind = MicroOp::MovSreg;
          u.sub = 255;
        }
      } else if (op->ty.bits == 16) {
        ok = narrow_dst(op->dst) && source(0, op->src);
        if (ok) u.kind = MicroOp::Mov16;
      } else {
        ok = op->ty.bits == 32 && narrow_dst(op->dst) && source(0, op->src);
        if (ok) u.kind = MicroOp::Mov32;
      }
    } else if (const auto* op = std::get_if<OpCvta>(&ins.op)) {
      // Into or out of .global or the generic space, a 64-bit address is its own (no window to add or take away).
      ok = op->ty.bits == 64 && (op->space == Space::Global || op->space == Space::Generic || op->space == Space::Param) &&
           wide_dst(op->dst) && source64(0, op->src);
      if (ok) u.kind = MicroOp::Mov64;
    } else if (const auto* op = std::get_if<OpLd>(&ins.op)) {
      // A scalar kernel parameter, by the position the parser recorded.
      ok = op->space == Space::Param && op->addr.base_kind != Addr::Base::Reg && op->addr.param_index != kNoReg &&
           op->dsts.size() == 1 && op->dsts[0].id != kNoReg && op->addr.offset >= INT32_MIN &&
           op->addr.offset <= INT32_MAX && (op->ty.bits == 8 || op->ty.bits == 16 || op->ty.bits == 32 || op->ty.bits == 64);
      if (ok) {
        u.kind = MicroOp::LdParam;
        u.sub = static_cast<uint8_t>(op->ty.bits / 8);
        u.src[0] = op->addr.param_index;
        u.src[1] = static_cast<uint32_t>(static_cast<int32_t>(op->addr.offset));
        u.dst = op->dsts[0].id;
        if (op->ty.is_signed()) u.flags |= MicroOp::kSigned;
        if (op->dsts[0].wide) u.flags |= MicroOp::kDstWide;
      }
    } else if (const auto* op = std::get_if<OpPredBin>(&ins.op)) {
      const bool a_ok = op->a_imm >= 0 || (!op->a.wide && op->a.id != kNoReg);
      const bool b_ok = op->b_imm >= 0 || (!op->b.wide && op->b.id != kNoReg);
      ok = a_ok && b_ok && narrow_dst(op->dst);
      if (ok) {
        u.kind = MicroOp::PredBin;
        u.sub = static_cast<uint8_t>(op->op);
        if (op->a_imm >= 0) { u.src_imm |= 1; u.src[0] = op->a_imm ? 1 : 0; } else { u.src[0] = op->a.id; }
        if (op->b_imm >= 0) { u.src_imm |= 2; u.src[1] = op->b_imm ? 1 : 0; } else { u.src[1] = op->b.id; }
      }
    } else if (const auto* op = std::get_if<OpNotPred>(&ins.op)) {
      ok = !op->src.wide && op->src.id != kNoReg && narrow_dst(op->dst);
      if (ok) {
        u.kind = MicroOp::NotPred;
        u.src[0] = op->src.id;
      }
    } else if (const auto* op = std::get_if<OpSetp>(&ins.op); op && op->ty.bits == 16 && !op->ty.is_real()) {
      // 16-bit integer comparisons (compare's narrow_s/narrow_u at 16 bits): the six ordered ones.
      ok = !op->has_bop && !op->has_q && !op->packed && narrow_dst(op->dst) && source(0, op->a) && source(1, op->b) &&
           (op->cmp == CmpOp::Eq || op->cmp == CmpOp::Ne || op->cmp == CmpOp::Lt || op->cmp == CmpOp::Le ||
            op->cmp == CmpOp::Gt || op->cmp == CmpOp::Ge);
      if (ok) {
        u.kind = MicroOp::Setp16;
        u.sub = static_cast<uint8_t>(op->cmp);
        if (op->ty.is_signed()) u.flags |= MicroOp::kSigned;
      }
    } else if (const auto* op = std::get_if<OpSetp>(&ins.op); op && op->ty.bits == 64) {
      // 64-bit integer and f64 comparisons, with registers or small immediates (an f64 literal is 64 bits).
      // (an f64 comparison names its registers as 64-bit ones and its constants through the function's table)
      const size_t consts = fn.micro_consts.size();
      ok = !op->ty.is_bfloat() && !op->has_bop && !op->has_q && !op->packed && !op->ftz && narrow_dst(op->dst) &&
           (op->ty.is_float() ? (source_f64(0, op->a) && source_f64(1, op->b)) : (source64(0, op->a) && source64(1, op->b)));
      if (!ok) fn.micro_consts.resize(consts);
      if (ok && !op->ty.is_float() &&
          !(op->cmp == CmpOp::Eq || op->cmp == CmpOp::Ne || op->cmp == CmpOp::Lt || op->cmp == CmpOp::Le ||
            op->cmp == CmpOp::Gt || op->cmp == CmpOp::Ge))
        ok = false;
      if (ok) {
        u.kind = MicroOp::Setp64;
        u.sub = static_cast<uint8_t>(op->cmp);
        if (op->ty.is_float()) u.flags |= MicroOp::kFloat;
        if (op->ty.is_signed()) u.flags |= MicroOp::kSigned;
      }
    } else if (const auto* op = std::get_if<OpSetp>(&ins.op)) {
      ok = op->ty.bits == 32 && !op->ty.is_bfloat() && !op->has_bop && !op->has_q && !op->packed &&
           (!op->ftz || op->ty.is_float()) && narrow_dst(op->dst) && source(0, op->a) && source(1, op->b);
      // The integer comparisons the fast path knows: the six ordered ones.
      if (ok && !op->ty.is_float() &&
          !(op->cmp == CmpOp::Eq || op->cmp == CmpOp::Ne || op->cmp == CmpOp::Lt || op->cmp == CmpOp::Le ||
            op->cmp == CmpOp::Gt || op->cmp == CmpOp::Ge))
        ok = false;
      if (ok) {
        u.kind = MicroOp::Setp32;
        u.sub = static_cast<uint8_t>(op->cmp);
        if (op->ftz) u.flags |= MicroOp::kFtz;
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
    } else if (const auto* op = std::get_if<OpFma>(&ins.op); op && op->ty.bits == 64) {
      const size_t consts = fn.micro_consts.size();
      ok = op->round == FRound::Nearest && !op->sat && !op->ftz && wide_dst(op->dst) && source_f64(0, op->a) &&
           source_f64(1, op->b) && source_f64(2, op->c);
      if (ok) {
        u.kind = MicroOp::Fma64;
        u.sub = static_cast<uint8_t>((op->neg_ab ? 1 : 0) | (op->neg_c ? 2 : 0));
      } else {
        fn.micro_consts.resize(consts);
      }
    } else if (const auto* op = std::get_if<OpFma>(&ins.op)) {
      ok = !(op->round != FRound::Nearest || op->sat) && op->ty.bits == 32 && narrow_dst(op->dst) &&
           source(0, op->a) && source(1, op->b) && source(2, op->c);
      if (ok) {
        u.kind = MicroOp::Fma32;
        u.sub = static_cast<uint8_t>((op->neg_ab ? 1 : 0) | (op->neg_c ? 2 : 0));   // a contracted sub
        if (op->ftz) u.flags |= MicroOp::kFtz;
      }
    } else if (const auto* op = std::get_if<OpFloatBin>(&ins.op); op && op->ty.bits == 64 && op->ty.is_float()) {
      // f64 add, sub, mul and div to nearest (min and max keep the general path's fmin and fmax).
      const size_t consts = fn.micro_consts.size();
      ok = op->round == FRound::Nearest && !op->nan_propagate && !op->sat && !op->xorsign_abs && !op->approx &&
           !op->ftz && op->op != FloatBinOp::Min && op->op != FloatBinOp::Max && wide_dst(op->dst) &&
           source_f64(0, op->a) && source_f64(1, op->b);
      if (ok) {
        u.kind = MicroOp::Float64;
        u.sub = static_cast<uint8_t>(op->op);
      } else {
        fn.micro_consts.resize(consts);
      }
    } else if (const auto* op = std::get_if<OpFloatBin>(&ins.op)) {
      ok = !(op->round != FRound::Nearest || op->nan_propagate || op->sat || op->xorsign_abs ||
             op->approx || op->ty.bits != 32 || !op->ty.is_float()) &&
           narrow_dst(op->dst) && source(0, op->a) && source(1, op->b);
      if (ok) {
        u.kind = MicroOp::Float32;
        u.sub = static_cast<uint8_t>(op->op);
        if (op->ftz) u.flags |= MicroOp::kFtz;
      }
    }
    if (ok) fn.micro[pc] = u;
  }
}

void intern_call_slots(Module& m) {
  std::unordered_map<std::string, uint32_t> ids;
  auto id_of = [&](const std::string& name) {
    return ids.emplace(name, static_cast<uint32_t>(ids.size())).first->second;
  };
  auto number = [&](EntryFn& fn) {
    fn.param_slot_ids.clear();
    for (const std::string& n : fn.param_slot_names) fn.param_slot_ids.push_back(id_of(n));
    fn.retval_slot_id = fn.retval_slot_name.empty() ? kNoSlot : id_of(fn.retval_slot_name);
    for (Instr& ins : fn.body) {
      if (auto* d = std::get_if<OpDeclSlot>(&ins.op)) {
        d->id = id_of(d->name);
      } else if (auto* st = std::get_if<OpStSlot>(&ins.op)) {
        st->id = id_of(st->slot);
      } else if (auto* ld = std::get_if<OpLdSlot>(&ins.op)) {
        ld->id = id_of(ld->slot);
      } else if (auto* c = std::get_if<OpCall>(&ins.op)) {
        c->retval_id = c->retval_slot.empty() ? kNoSlot : id_of(c->retval_slot);
        c->param_ids.clear();
        for (const std::string& n : c->param_slots) c->param_ids.push_back(id_of(n));
      }
    }
  };
  for (EntryFn& fn : m.entries) number(fn);
  for (auto& fn : m.funcs) number(*fn);
}

}  // namespace vgpu::ptx
