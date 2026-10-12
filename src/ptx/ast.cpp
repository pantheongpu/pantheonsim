#include "vgpu/ptx/ast.hpp"

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
    bool ok = false;
    if (const auto* op = std::get_if<OpIntBin>(&ins.op)) {
      ok = !(op->carry_in || op->carry_out || op->sat || op->ty.bits != 32 || op->op == IntBinOp::Div ||
             op->op == IntBinOp::Rem) &&
           narrow_dst(op->dst) && source(0, op->a) && source(1, op->b);
      if (ok) {
        u.kind = MicroOp::Int32;
        u.sub = static_cast<uint8_t>(op->op);
        if (op->ty.is_signed()) u.flags |= MicroOp::kSigned;
      }
    } else if (const auto* op = std::get_if<OpMadLo>(&ins.op)) {
      ok = !(op->carry_in || op->carry_out || op->ty.bits != 32) && narrow_dst(op->dst) && source(0, op->a) &&
           source(1, op->b) && source(2, op->c);
      if (ok) u.kind = MicroOp::MadLo32;
    } else if (const auto* op = std::get_if<OpMov>(&ins.op)) {
      ok = op->ty.bits == 32 && narrow_dst(op->dst) && source(0, op->src);
      if (ok) u.kind = MicroOp::Mov32;
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
