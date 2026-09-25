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
        if constexpr (requires { op.dst; }) {
          if constexpr (std::is_same_v<std::decay_t<decltype(op.dst)>, Reg>) {
            if (op.dst.id != kNoReg) defs.push_back(op.dst.id);
          } else if constexpr (std::is_same_v<std::decay_t<decltype(op.dst)>, Addr>) {
            // An address-typed destination defines memory, not a register; the
            // register it names is read to form the address.
            use_addr(op.dst);
          }
        }
        if constexpr (requires { op.dsts; })
          for (const auto& d : op.dsts) defs.push_back(d.id);
        if constexpr (requires { op.src; }) {
          if constexpr (std::is_same_v<std::decay_t<decltype(op.src)>, Reg>)
            uses.push_back(op.src.id);
          else if constexpr (std::is_same_v<std::decay_t<decltype(op.src)>, Operand>)
            use_operand(op.src);
          else if constexpr (std::is_same_v<std::decay_t<decltype(op.src)>, Addr>)
            use_addr(op.src);
          else
            for (const auto& s : op.src) use_operand(s);
        }
        if constexpr (requires { op.src_size; }) use_operand(op.src_size);
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
        if constexpr (std::is_same_v<std::decay_t<decltype(op)>, OpWgmma>) {
          // The accumulator is read as well as written, and the descriptors
          // and the scale-d predicate are ordinary sources.
          for (const auto& r : op.d) uses.push_back(r.id);
          use_operand(op.a_desc);
          use_operand(op.b_desc);
          use_operand(op.scale_d);
        }
      },
      ins.op);
}

}  // namespace

RegisterUsage analyze_registers(const EntryFn& fn) {
  RegisterUsage out;
  out.local_bytes = fn.local_frame_size;
  if (fn.num_regs == 0 || fn.body.empty()) return out;

  const size_t n = fn.body.size();

  // Per-instruction definitions and uses, deduplicated.
  std::vector<std::vector<uint32_t>> defs(n), uses(n);
  for (size_t i = 0; i < n; ++i) {
    std::vector<uint32_t> d, u;
    collect(fn.body[i], d, u);
    auto keep = [&](std::vector<uint32_t>& src, std::vector<uint32_t>& dst) {
      std::sort(src.begin(), src.end());
      src.erase(std::unique(src.begin(), src.end()), src.end());
      for (uint32_t id : src)
        if (id < fn.num_regs) dst.push_back(id);
    };
    keep(d, defs[i]);
    keep(u, uses[i]);
  }

  // Successors. An unpredicated branch goes only to its target; a predicated
  // one may also fall through; ret ends the path.
  auto successors = [&](size_t i, size_t* buf) -> int {
    const Instr& ins = fn.body[i];
    if (std::holds_alternative<OpRet>(ins.op)) return 0;
    if (const auto* br = std::get_if<OpBra>(&ins.op)) {
      int cnt = 0;
      if (br->target < n) buf[cnt++] = br->target;
      if (ins.has_pred && i + 1 < n) buf[cnt++] = i + 1;
      return cnt;
    }
    if (i + 1 < n) { buf[0] = i + 1; return 1; }
    return 0;
  };

  // Backward dataflow liveness. The earlier version approximated a live range
  // as first-definition to last-use and then extended everything that touched
  // a loop across the whole loop body -- which made every value in a
  // grid-stride loop look simultaneously live and reported 328 registers for a
  // kernel ptxas compiles into 14. A real fixed-point solve costs a few passes
  // and gets the answer right:
  //     live_out[i] = union of live_in over successors
  //     live_in[i]  = uses[i] + (live_out[i] - defs[i])
  const size_t words = (fn.num_regs + 63) / 64;
  std::vector<uint64_t> live_in(n * words, 0), live_out(n * words, 0), scratch(words, 0);
  auto bit_set = [](uint64_t* w, uint32_t id) { w[id >> 6] |= 1ull << (id & 63); };
  auto bit_clear = [](uint64_t* w, uint32_t id) { w[id >> 6] &= ~(1ull << (id & 63)); };

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t ii = n; ii-- > 0;) {
      std::fill(scratch.begin(), scratch.end(), 0ull);
      size_t succ[2];
      const int ns = successors(ii, succ);
      for (int k = 0; k < ns; ++k) {
        const uint64_t* in = &live_in[succ[k] * words];
        for (size_t w = 0; w < words; ++w) scratch[w] |= in[w];
      }
      uint64_t* out_i = &live_out[ii * words];
      for (size_t w = 0; w < words; ++w)
        if (out_i[w] != scratch[w]) { out_i[w] = scratch[w]; changed = true; }

      // live_in = uses + (live_out - defs)
      for (size_t w = 0; w < words; ++w) scratch[w] = out_i[w];
      for (uint32_t id : defs[ii]) bit_clear(scratch.data(), id);
      for (uint32_t id : uses[ii]) bit_set(scratch.data(), id);
      uint64_t* in_i = &live_in[ii * words];
      for (size_t w = 0; w < words; ++w)
        if (in_i[w] != scratch[w]) { in_i[w] = scratch[w]; changed = true; }
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

  // Peak pressure: at each instruction, what is live on the way out plus
  // whatever this instruction defines -- a dead definition still needs a
  // register while it is being written.
  uint32_t peak = 0, ppeak = 0;
  for (size_t i = 0; i < n; ++i) {
    std::fill(scratch.begin(), scratch.end(), 0ull);
    const uint64_t* out_i = &live_out[i * words];
    for (size_t w = 0; w < words; ++w) scratch[w] = out_i[w];
    for (uint32_t id : defs[i]) bit_set(scratch.data(), id);
    uint32_t live = 0, plive = 0;
    for (size_t w = 0; w < words; ++w) {
      uint64_t bits = scratch[w];
      while (bits) {
        const uint32_t id = static_cast<uint32_t>(w * 64 + __builtin_ctzll(bits));
        bits &= bits - 1;
        if (is_pred[id]) ++plive;
        else live += width[id];
      }
    }
    peak = std::max(peak, live);
    ppeak = std::max(ppeak, plive);
  }

  out.peak_live = peak;
  out.pred_regs = ppeak;
  // Round up to the hardware's allocation granularity.
  out.regs_per_thread = ((peak + kAllocGranularity - 1) / kAllocGranularity) * kAllocGranularity;
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
