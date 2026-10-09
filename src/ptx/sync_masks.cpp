// A *.sync warp instruction's member mask, when ptxas can see what it is.
//
// nvcc writes __ballot_sync(0xffffffff, p), __reduce_add_sync(0xffff, x) and
// the like as `mov.u32 %r, <constant>;` and the instruction naming %r. ptxas
// sees through that, and a constant mask compiles to the bare instruction: no
// WARPSYNC, no check that the thread is in the mask, the mask not consulted
// (an RTX 3060 runs __ballot_sync(0xffff, p) from all 32 lanes as the 32-lane
// ballot and __reduce_add_sync(0xffff, lane) as 496 in every lane). A mask
// ptxas cannot see through -- a lane-dependent value, a kernel argument --
// compiles to code that checks it at run time and traps a thread the mask
// leaves out ("an illegal instruction was encountered", 715). The interpreter
// tells the two apart by the operand: an immediate is the first kind, a
// register the second. So a register defined once, by a mov of an immediate,
// becomes that immediate here.
#include "vgpu/ptx/sync_masks.hpp"

#include <unordered_map>
#include <vector>

#include "vgpu/ptx/regalloc.hpp"

namespace vgpu::ptx {
namespace {

uint32_t key(const Reg& r) { return r.id * 2 + (r.wide ? 1u : 0u); }

// The member mask operand of a *.sync instruction, or null.
Operand* members_of(Instr& ins) {
  if (auto* op = std::get_if<OpVote>(&ins.op); op && op->has_members) return &op->members;
  if (auto* op = std::get_if<OpShfl>(&ins.op); op && op->has_members) return &op->member_mask;
  if (auto* op = std::get_if<OpMatch>(&ins.op)) return &op->membermask;
  if (auto* op = std::get_if<OpRedux>(&ins.op)) return &op->members;
  if (auto* op = std::get_if<OpBar>(&ins.op); op && op->warp) return &op->id;
  return nullptr;
}

}  // namespace

size_t fold_constant_sync_masks(EntryFn& fn) {
  auto& body = fn.body;
  std::unordered_map<uint32_t, std::vector<size_t>> defs_of;
  bool any = false;
  std::vector<uint32_t> defs, uses;
  for (size_t i = 0; i < body.size(); ++i) {
    if (members_of(body[i])) any = true;
    defs.clear();
    uses.clear();
    instr_registers(body[i], defs, uses);
    for (uint32_t k : defs) defs_of[k].push_back(i);
  }
  if (!any) return 0;
  size_t folded = 0;
  for (Instr& ins : body) {
    Operand* m = members_of(ins);
    if (!m) continue;
    const auto* r = std::get_if<RegOperand>(m);
    if (!r) continue;
    const auto d = defs_of.find(key(r->reg));
    if (d == defs_of.end() || d->second.size() != 1) continue;
    const Instr& def = body[d->second[0]];
    const auto* mov = std::get_if<OpMov>(&def.op);
    if (!mov || def.has_pred) continue;
    const auto* imm = std::get_if<ImmInt>(&mov->src);
    if (!imm) continue;
    *m = ImmInt{static_cast<int64_t>(static_cast<uint32_t>(imm->value))};
    ++folded;
  }
  return folded;
}

}  // namespace vgpu::ptx
