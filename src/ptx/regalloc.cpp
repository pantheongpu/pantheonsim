#include "vgpu/ptx/regalloc.hpp"

#include <algorithm>
#include <map>
#include <vector>

namespace vgpu::ptx {
namespace {

// NVIDIA parts allocate registers in fixed-size groups; rounding up mirrors
// that and keeps the estimate on the conservative side.
constexpr uint32_t kAllocGranularity = 8;

// How many 32-bit registers a value of this type occupies.
uint32_t slots_for(const Type& t) {
  if (t.kind == Type::Kind::Pred) return 0;  // predicates use their own file
  return t.bits > 32 ? 2u : 1u;
}

// Collects every register referenced by an instruction, separating the
// destination (a definition) from the sources (uses).
void collect(const Instr& ins, std::vector<uint32_t>& defs, std::vector<uint32_t>& uses) {
  auto use_operand = [&](const Operand& o) {
    if (const auto* r = std::get_if<RegOperand>(&o)) uses.push_back(r->reg.id);
  };
  auto use_addr = [&](const Addr& a) {
    if (a.base_kind == Addr::Base::Reg && a.base_id != kNoReg) uses.push_back(a.base_id);
  };
  if (ins.has_pred && ins.pred.id != kNoReg) uses.push_back(ins.pred.id);

  std::visit(
      [&](const auto& op) {
        using T = std::decay_t<decltype(op)>;
        if constexpr (requires { op.dst; }) {
          if constexpr (std::is_same_v<std::decay_t<decltype(op.dst)>, Reg>)
            if (op.dst.id != kNoReg) defs.push_back(op.dst.id);
        }
        if constexpr (requires { op.dsts; })
          for (const auto& d : op.dsts) defs.push_back(d.id);
        if constexpr (requires { op.src; }) {
          if constexpr (std::is_same_v<std::decay_t<decltype(op.src)>, Reg>)
            uses.push_back(op.src.id);
          else if constexpr (std::is_same_v<std::decay_t<decltype(op.src)>, Operand>)
            use_operand(op.src);
          else
            for (const auto& s : op.src) use_operand(s);
        }
        if constexpr (requires { op.srcs; })
          for (const auto& s : op.srcs) use_operand(s);
        if constexpr (requires { op.a; }) {
          if constexpr (std::is_same_v<std::decay_t<decltype(op.a)>, Reg>) uses.push_back(op.a.id);
          else if constexpr (std::is_same_v<std::decay_t<decltype(op.a)>, Operand>) use_operand(op.a);
          else for (const auto& r : op.a) uses.push_back(r.id);
        }
        if constexpr (requires { op.b; }) {
          if constexpr (std::is_same_v<std::decay_t<decltype(op.b)>, Reg>) uses.push_back(op.b.id);
          else if constexpr (std::is_same_v<std::decay_t<decltype(op.b)>, Operand>) use_operand(op.b);
          else for (const auto& r : op.b) uses.push_back(r.id);
        }
        if constexpr (requires { op.c; }) {
          if constexpr (std::is_same_v<std::decay_t<decltype(op.c)>, Operand>) use_operand(op.c);
          else for (const auto& r : op.c) uses.push_back(r.id);
        }
        if constexpr (requires { op.d; }) {
          if constexpr (std::is_same_v<std::decay_t<decltype(op.d)>, Operand>) use_operand(op.d);
          else for (const auto& r : op.d) defs.push_back(r.id);
        }
        if constexpr (requires { op.pred; }) uses.push_back(op.pred.id);
        if constexpr (requires { op.pred_dst; })
          if (op.pred_dst.id != kNoReg) defs.push_back(op.pred_dst.id);
        if constexpr (requires { op.member_mask; }) use_operand(op.member_mask);
        if constexpr (requires { op.stride; }) use_operand(op.stride);
        if constexpr (requires { op.addr; }) use_addr(op.addr);
      },
      ins.op);
}

}  // namespace

RegisterUsage analyze_registers(const EntryFn& fn) {
  RegisterUsage out;
  out.local_bytes = fn.local_frame_size;
  if (fn.num_regs == 0 || fn.body.empty()) return out;

  const size_t n = fn.body.size();
  constexpr size_t kNone = static_cast<size_t>(-1);
  std::vector<size_t> first(fn.num_regs, kNone), last(fn.num_regs, 0);

  for (size_t i = 0; i < n; ++i) {
    std::vector<uint32_t> defs, uses;
    collect(fn.body[i], defs, uses);
    for (uint32_t id : defs) {
      if (id >= fn.num_regs) continue;
      if (first[id] == kNone) first[id] = i;
      last[id] = std::max(last[id], i);
    }
    for (uint32_t id : uses) {
      if (id >= fn.num_regs) continue;
      if (first[id] == kNone) first[id] = i;  // used before any def we saw
      last[id] = std::max(last[id], i);
    }
  }

  // A value live anywhere inside a loop stays live for the whole loop, since
  // the back edge can return to the top with it still needed.
  for (size_t i = 0; i < n; ++i) {
    const auto* br = std::get_if<OpBra>(&fn.body[i].op);
    if (!br || br->target > i) continue;  // forward branch: not a loop edge
    size_t lo = br->target, hi = i;
    for (uint32_t id = 0; id < fn.num_regs; ++id) {
      if (first[id] == kNone) continue;
      if (first[id] <= hi && last[id] >= lo) {  // overlaps the loop body
        first[id] = std::min(first[id], lo);
        last[id] = std::max(last[id], hi);
      }
    }
  }

  // Width of each register, from its declaration. Undeclared (inline-asm
  // locals) default to 32-bit.
  std::vector<uint32_t> width(fn.num_regs, 1), is_pred(fn.num_regs, 0);
  for (const auto& [name, id] : fn.reg_ids) {
    if (id >= fn.num_regs) continue;
    auto it = fn.reg_decls.find(name);
    if (it == fn.reg_decls.end()) continue;
    if (it->second.kind == Type::Kind::Pred) is_pred[id] = 1;
    width[id] = slots_for(it->second);
  }

  // Linear scan for the peak of simultaneously live values.
  std::vector<int32_t> delta(n + 1, 0), pred_delta(n + 1, 0);
  for (uint32_t id = 0; id < fn.num_regs; ++id) {
    if (first[id] == kNone) continue;
    if (is_pred[id]) {
      pred_delta[first[id]] += 1;
      pred_delta[last[id] + 1] -= 1;
    } else {
      delta[first[id]] += static_cast<int32_t>(width[id]);
      delta[last[id] + 1] -= static_cast<int32_t>(width[id]);
    }
  }
  int32_t live = 0, peak = 0, plive = 0, ppeak = 0;
  for (size_t i = 0; i <= n; ++i) {
    live += delta[i];
    plive += pred_delta[i];
    peak = std::max(peak, live);
    ppeak = std::max(ppeak, plive);
  }

  out.peak_live = static_cast<uint32_t>(peak);
  out.pred_regs = static_cast<uint32_t>(ppeak);
  // Round up to the hardware's allocation granularity.
  out.regs_per_thread =
      ((static_cast<uint32_t>(peak) + kAllocGranularity - 1) / kAllocGranularity) * kAllocGranularity;
  if (out.regs_per_thread == 0) out.regs_per_thread = kAllocGranularity;
  return out;
}

Occupancy compute_occupancy(uint32_t regs_per_thread, uint32_t threads_per_block,
                            uint32_t static_shared_bytes, uint32_t dynamic_shared_bytes,
                            uint32_t regs_per_sm, uint32_t max_threads_per_sm,
                            uint32_t max_blocks_per_sm, uint32_t shared_per_sm,
                            uint32_t warp_size) {
  Occupancy o;
  if (threads_per_block == 0 || warp_size == 0) return o;
  o.warps_per_block = (threads_per_block + warp_size - 1) / warp_size;
  if (o.warps_per_block == 0) return o;

  uint32_t by_warps = max_threads_per_sm / warp_size / o.warps_per_block;
  uint32_t by_blocks = max_blocks_per_sm;
  uint32_t by_regs = ~0u;
  if (regs_per_thread > 0 && regs_per_sm > 0) {
    uint32_t regs_per_warp = regs_per_thread * warp_size;
    uint32_t warps_by_regs = regs_per_warp ? regs_per_sm / regs_per_warp : ~0u;
    by_regs = warps_by_regs / o.warps_per_block;
  }
  uint32_t by_shared = ~0u;
  uint64_t shared = static_cast<uint64_t>(static_shared_bytes) + dynamic_shared_bytes;
  if (shared > 0 && shared_per_sm > 0)
    by_shared = static_cast<uint32_t>(shared_per_sm / shared);

  o.blocks_per_sm = std::min(std::min(by_warps, by_blocks), std::min(by_regs, by_shared));
  o.warps_per_sm = o.blocks_per_sm * o.warps_per_block;
  o.limited_by = (o.blocks_per_sm == by_regs)     ? "registers"
                 : (o.blocks_per_sm == by_shared) ? "shared memory"
                 : (o.blocks_per_sm == by_warps)  ? "warps"
                                                  : "blocks";
  return o;
}

}  // namespace vgpu::ptx
