// The value a SASS atomic (ATOM, ATOMG, ATOMS, RED) leaves in memory, given the one it finds: the rules of the SASS executor's
// read-modify-write (src/sass/exec_ops.inc exec_mem), and the faster forms of the common ones that a warp's batch of atomics runs.
// `op` is Instr::f[0] (0 ADD, 1 MIN, 2 MAX, 3 INC, 4 DEC, 5 AND, 6 OR, 7 XOR, 8 EXCH) and `type` Instr::f[4] (0 U32, 1 S32, 2 U64,
// 3 F32, 4 F16x2, 5 S64, 6 F64, 7 BF16x2); 32-bit values are the low 32 bits of a 64-bit word.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "vgpu/exec/numerics.hpp"
#include "vgpu/ptx/ast.hpp"

namespace vgpu::exec::sassatom {

inline float as_f32(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline uint32_t f32_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
inline double as_f64(uint64_t b) { double d; std::memcpy(&d, &b, 8); return d; }
inline uint64_t f64_bits(double d) { uint64_t b; std::memcpy(&b, &d, 8); return b; }

inline bool is64(unsigned type) { return type == 2 || type == 5 || type == 6; }

// The general rule. `cas`: compare-and-swap (`v` is the compare value, `swap` the new one); `popc_inc`: ATOMS.POPC.INC. False for an
// operation that is none of the above (the caller faults); `out` is then unset. The result is cut to 32 bits for a 32-bit type.
inline bool apply(unsigned op, unsigned type, bool cas, bool popc_inc, uint64_t old, uint64_t v, uint64_t swap, uint64_t& out) {
  uint64_t nw = old;
  if (cas) {
    if (old == v) nw = swap;
  } else if (type == 4 || type == 7) {   // F16x2, BF16x2 (and their vectors): add, min or max on each half
    const ptx::AtomOp fop = op == 1 ? ptx::AtomOp::Min : op == 2 ? ptx::AtomOp::Max : ptx::AtomOp::Add;
    nw = exec::atom_word(fop, type == 7 ? 2 : 1, static_cast<uint32_t>(old), static_cast<uint32_t>(v));
  } else {
    switch (popc_inc ? 3 : op) {
      case 0:   // ADD, by the PTX engine's rules (an RTX 3060's)
        if (type == 3) {
          // .ftz on inputs and result; a NaN result is the canonical one.
          const auto fz = [](float x) { return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.0f, x) : x; };
          const float r = fz(fz(as_f32(static_cast<uint32_t>(old))) + fz(as_f32(static_cast<uint32_t>(v))));
          nw = r != r ? 0x7fffffffu : f32_bits(r);
        } else if (type == 6) {
          // A NaN comes through as it is: the one added, else the one in memory.
          const double x = as_f64(old), y = as_f64(v);
          nw = std::isnan(y) ? v : std::isnan(x) ? old : f64_bits(x + y);
        } else {
          nw = old + v;
        }
        break;
      case 1:   // MIN
        if (type == 1) nw = static_cast<uint32_t>(std::min(static_cast<int32_t>(old), static_cast<int32_t>(v)));
        else if (type == 5) nw = static_cast<uint64_t>(std::min(static_cast<int64_t>(old), static_cast<int64_t>(v)));
        else nw = std::min(old, v);
        break;
      case 2:   // MAX
        if (type == 1) nw = static_cast<uint32_t>(std::max(static_cast<int32_t>(old), static_cast<int32_t>(v)));
        else if (type == 5) nw = static_cast<uint64_t>(std::max(static_cast<int64_t>(old), static_cast<int64_t>(v)));
        else nw = std::max(old, v);
        break;
      case 3: nw = popc_inc ? old + 1 : (static_cast<uint32_t>(old) >= static_cast<uint32_t>(v) ? 0 : old + 1); break;   // INC
      case 4: nw = (old == 0 || static_cast<uint32_t>(old) > static_cast<uint32_t>(v)) ? v : old - 1; break;             // DEC
      case 5: nw = old & v; break;
      case 6: nw = old | v; break;
      case 7: nw = old ^ v; break;
      case 8: nw = v; break;   // EXCH
      default: return false;
    }
  }
  out = is64(type) ? nw : (nw & 0xffffffffull);
  return true;
}

// The steps of the common operations, each a function of (old, v, swap) with the operation known when it is compiled. Their result is
// not cut to 32 bits (the caller's mask does that); each gives `apply`'s value for the operation it is named for.
struct CasStep { uint64_t operator()(uint64_t old, uint64_t cmp, uint64_t swap) const { return old == cmp ? swap : old; } };
struct AddStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return old + x; } };
struct AddF32Step {
  uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const {
    // .ftz on inputs and result; a NaN result is the canonical one (the general rule works it out on floats).
    const auto fz = [](uint32_t b) { return (b & 0x7f800000u) == 0 ? b & 0x80000000u : b; };
    const float sum = as_f32(fz(static_cast<uint32_t>(old))) + as_f32(fz(static_cast<uint32_t>(x)));
    return uint64_t{sum != sum ? 0x7fffffffu : fz(f32_bits(sum))};
  }
};
struct AddF64Step {
  uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const {
    // A NaN comes through as it is: the one added, else the one in memory (the general rule).
    const double a = as_f64(old), b = as_f64(x);
    return b != b ? x : a != a ? old : f64_bits(a + b);
  }
};
struct MinS32Step { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return static_cast<uint32_t>(std::min(static_cast<int32_t>(old), static_cast<int32_t>(x))); } };
struct MaxS32Step { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return static_cast<uint32_t>(std::max(static_cast<int32_t>(old), static_cast<int32_t>(x))); } };
struct MinS64Step { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return static_cast<uint64_t>(std::min(static_cast<int64_t>(old), static_cast<int64_t>(x))); } };
struct MaxS64Step { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return static_cast<uint64_t>(std::max(static_cast<int64_t>(old), static_cast<int64_t>(x))); } };
// INC and DEC compare the low 32 bits whatever the type is (the general rule).
struct IncStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return static_cast<uint32_t>(old) >= static_cast<uint32_t>(x) ? 0 : old + 1; } };
struct DecStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return (old == 0 || static_cast<uint32_t>(old) > static_cast<uint32_t>(x)) ? x : old - 1; } };
struct MinStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return std::min(old, x); } };
struct MaxStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return std::max(old, x); } };
struct AndStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return old & x; } };
struct OrStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return old | x; } };
struct XorStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return old ^ x; } };
struct ExchStep { uint64_t operator()(uint64_t, uint64_t x, uint64_t) const { return x; } };

// Which step stands for (op, type, cas); General when none does (the halves, the signed min and max, INC, DEC, F64 and the rest: `apply`).
enum class Step : uint8_t { General, Cas, Add, AddF32, AddF64, Min, Max, MinS32, MaxS32, MinS64, MaxS64, Inc, Dec, And, Or, Xor, Exch };
inline Step step_for(unsigned op, unsigned type, bool cas) {
  if (cas) return Step::Cas;
  if (type == 4 || type == 7) return Step::General;   // (every operation on the halves works on each half)
  switch (op) {
    case 0: return type == 3 ? Step::AddF32 : type == 6 ? Step::AddF64 : (type == 0 || type == 1 || type == 2 || type == 5) ? Step::Add : Step::General;
    case 1: return (type == 0 || type == 2) ? Step::Min : type == 1 ? Step::MinS32 : type == 5 ? Step::MinS64 : Step::General;
    case 2: return (type == 0 || type == 2) ? Step::Max : type == 1 ? Step::MaxS32 : type == 5 ? Step::MaxS64 : Step::General;
    case 3: return Step::Inc;
    case 4: return Step::Dec;
    case 5: return Step::And;
    case 6: return Step::Or;
    case 7: return Step::Xor;
    case 8: return Step::Exch;
    default: return Step::General;
  }
}

// A warp's atomics on words already loaded: `cur[slot[l]]` is lane l's word (lanes that share an address share a slot), and each lane of `ex`,
// in lane order, finds in it what the lanes before left. old_of[l] is what lane l found; cur takes the new values (cut to `keep`). Returns
// whether any lane changed a word, or -1 (nothing touched) for the operations without a step. `nslots` is the number of distinct words
// (one: every lane on the same word). One call per warp, a loop per step.
namespace detail {
template <class StepFn>
[[gnu::noinline]] inline int global_lanes(uint32_t ex, unsigned nslots, const unsigned* __restrict slot, uint64_t* __restrict cur,
                                          const uint64_t* __restrict v, const uint64_t* __restrict swp, uint64_t keep,
                                          uint64_t* __restrict old_of, StepFn step) {
  uint64_t changed = 0;
  if (nslots == 1) {   // every lane on one word (a counter, a maximum): the word stays in a register from lane to lane
    uint64_t c = cur[0];
    for (uint32_t m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(__builtin_ctz(m));
      const uint64_t nw = step(c, v[l], swp ? swp[l] : 0) & keep;
      old_of[l] = c;
      changed |= nw ^ c;
      c = nw;
    }
    cur[0] = c;
    return changed != 0;
  }
  for (uint32_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctz(m));
    const uint64_t old = cur[slot[l]];
    const uint64_t nw = step(old, v[l], swp ? swp[l] : 0) & keep;
    old_of[l] = old;
    changed |= nw ^ old;
    cur[slot[l]] = nw;
  }
  return changed != 0;
}
}  // namespace detail

inline int global_batch(unsigned op, unsigned type, bool cas, uint32_t ex, unsigned nslots, const unsigned* slot, uint64_t* cur,
                        const uint64_t* v, const uint64_t* swp, uint64_t* old_of) {
  const uint64_t keep = is64(type) ? ~uint64_t{0} : 0xffffffffull;
  const auto go = [&](auto step) { return detail::global_lanes(ex, nslots, slot, cur, v, swp, keep, old_of, step); };
  switch (step_for(op, type, cas)) {
    case Step::Cas: return go(CasStep{});
    case Step::Add: return go(AddStep{});
    case Step::AddF32: return go(AddF32Step{});
    case Step::AddF64: return go(AddF64Step{});
    case Step::Min: return go(MinStep{});
    case Step::Max: return go(MaxStep{});
    case Step::MinS32: return go(MinS32Step{});
    case Step::MaxS32: return go(MaxS32Step{});
    case Step::MinS64: return go(MinS64Step{});
    case Step::MaxS64: return go(MaxS64Step{});
    case Step::Inc: return go(IncStep{});
    case Step::Dec: return go(DecStep{});
    case Step::And: return go(AndStep{});
    case Step::Or: return go(OrStep{});
    case Step::Xor: return go(XorStep{});
    case Step::Exch: return go(ExchStep{});
    default: return -1;
  }
}

// Any operation of the general rule (`apply`), as a step: the halves, and whatever else has no loop of its own.
struct GeneralStep {
  unsigned op, type;
  bool cas;
  uint64_t operator()(uint64_t old, uint64_t x, uint64_t swap) const {
    uint64_t out = old;
    apply(op, type, cas, false, old, x, swap, out);
    return out;
  }
};

// A warp's atomics on plain device memory without a lock: the distinct words `uaddr[0..nu)` each get one compare-and-swap loop
// (MemoryManager::rmw_batch with lock_free) whose step runs the lanes of that word, `lane_mask[k]`, one after another in lane order from
// the value the swap replaces -- so a lane finds what the lanes before it left, and the word ends as the locked form leaves it; a retry
// (another thread got in between) runs them again from the new value, and old_of[l] is what the last run gave. What threads see: each
// word's update is atomic; the warp's words are not one step (CUDA promises none). Integer operations give the same old values and
// final words at any thread count only when the order does not matter (add, and, or, xor, min, max); an exchange, compare-and-swap, inc
// or dec, or a float add (rounding), take the order the threads reach the word in -- as on the card. Returns whether any lane changed a
// word (0 or 1), or -1 having done nothing (an operation not in the table, or the memory declined: a fault armed, an unaligned or out of
// allocation word, host-mapped memory) and the caller takes the locked form. `nu` is 1 to 32; the caller has checked op <= 8.
template <class Mem>
[[gnu::noinline]] inline int global_lockfree(Mem& mem, unsigned op, unsigned type, bool cas, unsigned nu, const uint64_t* uaddr, const uint32_t* lane_mask,
                           const uint64_t* v, const uint64_t* swp, uint64_t* old_of) {
  const uint64_t keep = is64(type) ? ~uint64_t{0} : 0xffffffffull;
  const uint32_t size = is64(type) ? 8 : 4;
  uint8_t* ptr[32];
  if (!mem.atomic_word_ptrs(uaddr, nu, size, ptr)) return -1;
  const auto go = [&](auto step) -> int {
    uint64_t any = 0;
    for (unsigned k = 0; k < nu; ++k) {
      uint64_t acc = 0;
      const auto compute = [&](uint64_t old) -> uint64_t {
        uint64_t c = old;
        acc = 0;
        for (uint32_t m = lane_mask[k]; m; m &= m - 1) {
          const unsigned l = static_cast<unsigned>(__builtin_ctz(m));
          const uint64_t nw = step(c, v[l], swp ? swp[l] : 0) & keep;
          old_of[l] = c;
          acc |= nw ^ c;
          c = nw;
        }
        return c;
      };
      if (uint8_t* p = ptr[k]) {   // (a word in a real chunk: the swap loop of MemoryManager::atomic_rmw_slot, inline)
        if constexpr (std::is_same_v<decltype(step), AddStep>) {
          // An integer add needs no loop: the word's lanes add up to one fetch-and-add, and lane l finds the word's old value plus what the
          // lanes before it added (the same values the chain gives, with no retry when other threads add to the word).
          uint64_t total = 0, any_add = 0;
          uint64_t before[32];
          for (uint32_t m = lane_mask[k]; m; m &= m - 1) {
            const unsigned l = static_cast<unsigned>(__builtin_ctz(m));
            before[l] = total;
            total += v[l] & keep;
            any_add |= v[l] & keep;
          }
          const uint64_t base = size == 4 ? uint64_t{__atomic_fetch_add(reinterpret_cast<uint32_t*>(p), static_cast<uint32_t>(total), __ATOMIC_SEQ_CST)}
                                          : __atomic_fetch_add(reinterpret_cast<uint64_t*>(p), total, __ATOMIC_SEQ_CST);
          for (uint32_t m = lane_mask[k]; m; m &= m - 1) {
            const unsigned l = static_cast<unsigned>(__builtin_ctz(m));
            old_of[l] = (base + before[l]) & keep;
          }
          acc = any_add;
          any |= acc;
          continue;
        } else if (size == 4) {
          uint32_t* w = reinterpret_cast<uint32_t*>(p);
          uint32_t old = __atomic_load_n(w, __ATOMIC_RELAXED);
          for (;;) {
            const uint32_t now = static_cast<uint32_t>(compute(old));
            if (now == old) break;   // nothing to change: the load was the whole operation
            if (__atomic_compare_exchange_n(w, &old, now, true, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) break;
          }
        } else {
          uint64_t* w = reinterpret_cast<uint64_t*>(p);
          uint64_t old = __atomic_load_n(w, __ATOMIC_RELAXED);
          for (;;) {
            const uint64_t now = compute(old);
            if (now == old) break;
            if (__atomic_compare_exchange_n(w, &old, now, true, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) break;
          }
        }
      } else {   // (an untouched or uniform chunk: made real only where the result differs, and always for a store that changes nothing)
        uint64_t found = 0;
        if (!mem.atomic_rmw(uaddr[k], size, compute, &found, /*always_store=*/true)) {
          // (declined after the pointers were taken -- a fault armed in between: the plain pair, which is all that is left)
          found = mem.load_scalar(uaddr[k], size);
          mem.store_scalar(uaddr[k], size, compute(found));
        }
      }
      any |= acc;
    }
    return any != 0;
  };
  switch (step_for(op, type, cas)) {
    case Step::Cas: return go(CasStep{});
    case Step::Add: return go(AddStep{});
    case Step::AddF32: return go(AddF32Step{});
    case Step::AddF64: return go(AddF64Step{});
    case Step::Min: return go(MinStep{});
    case Step::Max: return go(MaxStep{});
    case Step::MinS32: return go(MinS32Step{});
    case Step::MaxS32: return go(MaxS32Step{});
    case Step::MinS64: return go(MinS64Step{});
    case Step::MaxS64: return go(MaxS64Step{});
    case Step::Inc: return go(IncStep{});
    case Step::Dec: return go(DecStep{});
    case Step::And: return go(AndStep{});
    case Step::Or: return go(OrStep{});
    case Step::Xor: return go(XorStep{});
    case Step::Exch: return go(ExchStep{});
    default: return go(GeneralStep{op, type, cas});
  }
}

// A warp's atomics on one block's shared bytes: for each lane of `ex` in lane order, the word (4 or 8 bytes, by `type`) at sh + soff[l]
// is read into old_of[l] and replaced by the step's value (a lane finds what the lanes before it left, also when two share a word or
// overlap). The common operations (step_for) run as a loop of their own with the word width and the step known when it is compiled;
// false, nothing touched, for the rest (the caller's lane-by-lane path has them, and the fault for an operation that is none).
// The caller has checked that every lane's word lies inside the shared bytes.
namespace detail {
template <class W, class StepFn>
[[gnu::noinline]] inline void shared_lanes(uint8_t* __restrict sh, const uint32_t* __restrict soff, uint32_t ex, const uint64_t* __restrict v,
                                           const uint64_t* __restrict swp, uint64_t* __restrict old_of, StepFn step) {
  for (uint32_t m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(__builtin_ctz(m));
    W cur;
    std::memcpy(&cur, sh + soff[l], sizeof cur);
    old_of[l] = cur;
    const W nw = static_cast<W>(step(uint64_t{cur}, v[l], swp ? swp[l] : 0));   // (no exchange value but a compare-and-swap's)
    std::memcpy(sh + soff[l], &nw, sizeof nw);
  }
}
}  // namespace detail

inline bool shared_batch(unsigned op, unsigned type, bool cas, uint8_t* sh, const uint32_t* soff, uint32_t ex, const uint64_t* v,
                         const uint64_t* swp, uint64_t* old_of) {
  const Step s = step_for(op, type, cas);
  if (s == Step::General) return false;
  const bool wide = is64(type);
  const auto go = [&](auto step) {
    if (wide) detail::shared_lanes<uint64_t>(sh, soff, ex, v, swp, old_of, step);
    else detail::shared_lanes<uint32_t>(sh, soff, ex, v, swp, old_of, step);
  };
  switch (s) {
    case Step::Cas: go(CasStep{}); break;
    case Step::Add: go(AddStep{}); break;
    case Step::AddF32: go(AddF32Step{}); break;
    case Step::AddF64: go(AddF64Step{}); break;
    case Step::Min: go(MinStep{}); break;
    case Step::Max: go(MaxStep{}); break;
    case Step::MinS32: go(MinS32Step{}); break;
    case Step::MaxS32: go(MaxS32Step{}); break;
    case Step::MinS64: go(MinS64Step{}); break;
    case Step::MaxS64: go(MaxS64Step{}); break;
    case Step::Inc: go(IncStep{}); break;
    case Step::Dec: go(DecStep{}); break;
    case Step::And: go(AndStep{}); break;
    case Step::Or: go(OrStep{}); break;
    case Step::Xor: go(XorStep{}); break;
    case Step::Exch: go(ExchStep{}); break;
    default: return false;
  }
  return true;
}

// Whether a warp's atomics may be batched: none of the distinct addresses can be in memory the host maps (managed, pinned, IPC), whose atomics
// are the CPU's own and need no stripe lock. Asked before any lock is taken: a batch that locked 26 stripes and then declined cost TSan a
// multiple of the per-lane path. `mem.host_map_possible` is a range test without a lock.
template <class Mem>
inline bool batchable(const Mem& mem, const uint64_t* addrs, unsigned count) {
  for (unsigned i = 0; i < count; ++i)
    if (mem.host_map_possible(addrs[i])) return false;
  return true;
}

}  // namespace vgpu::exec::sassatom
