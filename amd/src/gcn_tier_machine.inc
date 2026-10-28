// The part of the AMD tier that needs the interpreter's Machine (included inside it, before step()): global loads and stores
// of whole dwords. It is global_access with the work the instruction's name asked for done at decode (gcn_tier.cpp's
// classify sets Inst::fast.mach and .words) and the address registers read as rows; the lane loop, the run detection
// (memory_run), the load memo and every memory call are the interpreter's own, in the same order, so a fault in lane 17 happens
// after lane 16's store as before.
//
// Everything step() counts for such an instruction is counted here, before the lanes run (an access that faults has been counted).
int tier_global(Wave& w, Group& g, const Inst& in) {
  const bool is_store = in.fast.mach == 2 || in.fast.mach == 5;
  const bool sub_dword = in.fast.mach >= 4;   // a byte or a short (global_load_ubyte ..., global_store_byte, global_store_short)
  // A wave's first plain store of a round lets the group's other waves run first (Wave::store_gate, in step()).
  if (is_store && w.store_gate && w.exec != 0 && g.waves.size() > 1) {
    w.store_gate = false;
    w.soft_yield = true;
    return 2;
  }
  InstructionCounts& n = stats.counts;
  ++n.mix[static_cast<int>(in.mix)];
  n.mops[static_cast<int>(in.mops_type)] += in.mops;
  w.pc += in.size;
  ++stats.instructions;
  ++n.vmem;
  ++n.flat;
  if (is_store) ++n.flat_write, ++n.vmem_wr;
  else ++n.flat_read, ++n.vmem_rd, issued_load(w);
  if (!w.exec) return 1;
  const uint32_t words = in.fast.words;
  const uint32_t access_bytes = sub_dword ? words : 4 * words;
  const uint64_t saddr_base = in.has_saddr ? scalar_field(w, in.saddr, true) : 0;
  const uint64_t offset = static_cast<uint64_t>(static_cast<int64_t>(in.offset));
  const uint32_t* const lo = w.vgpr[in.src[0].index].data();
  const uint32_t* const hi = in.has_saddr ? lo : w.vgpr[in.src[0].index + 1].data();
  bool try_runs = !sub_dword && (words == 1 || words == 2 || words == 4) && d.peers.empty() && runs_on;
  // Lanes whose addresses make no run go to memory together (wave_scalars), once: a byte, a short, a word or a pair a lane.
  bool try_batch = (sub_dword || words <= 2) && d.peers.empty() && runs_on;
  uint64_t memo_addr = 0;
  bool memo = false;
  uint32_t memo_words[kMaxWords];
  uint32_t buf[4];
  uint64_t memo_value = 0;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    if (!(w.exec >> lane & 1)) continue;
    const uint64_t addr = (in.has_saddr ? saddr_base + lo[lane] : static_cast<uint64_t>(hi[lane]) << 32 | lo[lane]) + offset;
    if (try_runs) {
      const uint32_t made = memory_run(w, in, !is_store, lane, addr, words, access_bytes, !is_store && memo && memo_addr == addr, memo_words);
      if (made) {
        if (!is_store) memo_addr = addr + uint64_t{4 * words} * (made - 1), memo = true;
        lane += made - 1;
        continue;
      }
      try_runs = false;
    }
    if (try_batch && (is_store || !memo)) {
      try_batch = false;
      uint64_t lane_addr[kLanes];
      for (uint64_t m = w.exec >> lane << lane; m; m &= m - 1) {
        const uint32_t l = static_cast<uint32_t>(__builtin_ctzll(m));
        lane_addr[l] = (in.has_saddr ? saddr_base + lo[l] : static_cast<uint64_t>(hi[l]) << 32 | lo[l]) + offset;
      }
      const Narrow nr{words, in.fast.sub != 0};
      if (wave_scalars(w, is_store, lane, lane_addr, is_store ? in.src[1] : in.dst[0], sub_dword ? 1 : words, sub_dword ? &nr : nullptr))
        return 1;
    }
    if (sub_dword) {
      if (is_store) {
        store(addr, words, w.vgpr[in.src[1].index][lane]);
      } else {
        if (!memo || memo_addr != addr) memo_value = load(addr, words), memo_addr = addr, memo = true;
        w.vgpr[in.dst[0].index][lane] = widen(memo_value, Narrow{words, in.fast.sub != 0});
      }
    } else if (is_store) {
      for (uint32_t k = 0; k < words; ++k) buf[k] = w.vgpr[in.src[1].index + k][lane];
      store_words(addr, words, buf);
    } else {
      if (!memo || memo_addr != addr) {
        load_words(addr, words, memo_words);
        memo_addr = addr, memo = true;
      }
      for (uint32_t k = 0; k < words; ++k) w.vgpr[in.dst[0].index + k][lane] = memo_words[k];
    }
  }
  return 1;
}

// s_barrier: the wave stops at it until the work-group's others arrive (run_round releases them).
int tier_barrier(Wave& w, const Inst& in) {
  InstructionCounts& n = stats.counts;
  ++n.mix[static_cast<int>(in.mix)];
  n.mops[static_cast<int>(in.mops_type)] += in.mops;
  w.pc += in.size;
  ++stats.instructions;
  w.at_barrier = true;
  ++stats.barriers;
  return 2;
}

// s_endpgm: the wave is done (step() counts it as it counts any instruction, and returns false).
int tier_endpgm(Wave& w, const Inst& in) {
  InstructionCounts& n = stats.counts;
  ++n.mix[static_cast<int>(in.mix)];
  n.mops[static_cast<int>(in.mops_type)] += in.mops;
  w.pc += in.size;
  ++stats.instructions;
  w.done = true;
  return 2;
}

// A scalar memory instruction: scalar_load does the work, as step() would have it done (counted, the pc moved on first).
int tier_smem(Wave& w, const Inst& in) {
  InstructionCounts& n = stats.counts;
  ++n.mix[static_cast<int>(in.mix)];
  n.mops[static_cast<int>(in.mops_type)] += in.mops;
  w.pc += in.size;
  ++stats.instructions;
  ++n.smem;
  scalar_load(w, in);
  return 1;
}

// v_pk_fma_f16, v_pk_add_f16, v_pk_mul_f16: the whole-wave arithmetic of packed_math.hpp, for the plain case (it says whether this is one: nothing
// has happened if not). Op::sub is 0 for the fused multiply-add, 1 for the add, 2 for the multiply, 3 for the minimum, 4 for the maximum.
int tier_pkf16(Wave& w, const Inst& in) {
  if (!packed_f16_wave(w, in, in.fast.sub == 0 ? packed::Op::Fma : in.fast.sub == 1 ? packed::Op::Add : in.fast.sub == 2 ? packed::Op::Mul
                                : in.fast.sub == 3 ? packed::Op::Min : packed::Op::Max))
    return 0;
  InstructionCounts& n = stats.counts;
  ++n.mix[static_cast<int>(in.mix)];
  n.mops[static_cast<int>(in.mops_type)] += in.mops;
  w.pc += in.size;
  ++stats.instructions;
  ++n.valu;
  return 1;
}

// v_accvgpr_read_b32 and v_accvgpr_write_b32: a register of one bank (or a scalar or a constant) into a register of the other, for the lanes EXEC
// names. The Machine's own lane_src and write_lane (the accumulation bank grows where a write reaches past it).
int tier_acc(Wave& w, const Inst& in) {
  InstructionCounts& n = stats.counts;
  ++n.mix[static_cast<int>(in.mix)];
  n.mops[static_cast<int>(in.mops_type)] += in.mops;
  w.pc += in.size;
  ++stats.instructions;
  ++n.valu;
  const InvariantSources invariant(*this, w, in);
  for (uint64_t m = w.exec; m; m &= m - 1) {
    const uint32_t lane = static_cast<uint32_t>(__builtin_ctzll(m));
    write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane));
  }
  return 1;
}

// A write-back or an invalidate of the caches (buffer_wbl2, buffer_inv, buffer_gl0_inv, ...): a fence, counted as a vector memory instruction, as step_interp has it.
int tier_fence(Wave& w, const Inst& in) {
  InstructionCounts& n = stats.counts;
  ++n.mix[static_cast<int>(in.mix)];
  n.mops[static_cast<int>(in.mops_type)] += in.mops;
  w.pc += in.size;
  ++stats.instructions;
  ++n.vmem;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  return 1;
}

// VOPD (RDNA3 and RDNA4's two instructions issued as one): each half by its own tier handler, as step_interp's Vopd case runs them -- X first, then Y
// with X's destination as it was where Y reads it (both read their sources before either writes). A half that hands back (a NaN where the rule is the
// interpreter's) hands the whole instruction back, with the registers as they were. Op::off0 and ::words are X's destination and its width in
// registers; Op::sub says that Y reads it.
// (Out of line: its 1 KB of saved rows, in the frame of step() (tier_mach is inlined there), moved the frames below it -- step_interp's, and the stack buffers
// of the matrix tiles under it -- and cost mfma_f64 3.5%: the tile kernel's loads and its stores alias in the low 12 bits of their addresses.)
[[gnu::noinline]] int tier_vopd(Wave& w, Group& g, const Inst& in) {
  if (__builtin_expect(!w.tier.ok, 0)) make_tier_regs(w, g);
  gcn::tier::Regs& r = w.tier.r;
  const Inst& x = in.dual[0];
  const Inst& y = in.dual[1];
  const uint32_t xd = static_cast<uint32_t>(in.fast.off0), xw = in.fast.words;
  alignas(32) uint32_t before[2][64], after[2][64];
  for (uint32_t k = 0; k < xw; ++k) std::memcpy(before[k], w.vgpr[xd + k].data(), sizeof before[k]);
  r.target = 0;
  if (!x.fast.run(x.fast, r)) return 0;
  const bool y_reads_x = in.fast.sub != 0;
  if (y_reads_x)
    for (uint32_t k = 0; k < xw; ++k) {
      std::memcpy(after[k], w.vgpr[xd + k].data(), sizeof after[k]);
      std::memcpy(w.vgpr[xd + k].data(), before[k], sizeof before[k]);
    }
  if (!y.fast.run(y.fast, r)) {
    for (uint32_t k = 0; k < xw; ++k) std::memcpy(w.vgpr[xd + k].data(), before[k], sizeof before[k]);
    return 0;
  }
  if (y_reads_x)
    for (uint32_t k = 0; k < xw; ++k) std::memcpy(w.vgpr[xd + k].data(), after[k], sizeof after[k]);
  InstructionCounts& n = stats.counts;
  for (const Inst& half : in.dual) ++n.mix[static_cast<int>(half.mix)];
  w.pc += in.size;
  ++stats.instructions;
  ++n.valu;
  return 1;
}

// The instructions the tier hands to the Machine, by Op::mach: 3 s_barrier, 6 s_endpgm, 7 scalar memory, 8 packed halves, 10 the accumulation register moves, 11 a cache fence, 12 a VOPD pair, else a global access.
int tier_mach(Wave& w, Group& g, const Inst& in) {
  switch (in.fast.mach) {
    case 3: return tier_barrier(w, in);
    case 6: return tier_endpgm(w, in);
    case 7: return tier_smem(w, in);
    case 8: return tier_pkf16(w, in);
    case 10: return tier_acc(w, in);
    case 11: return tier_fence(w, in);
    case 12: return tier_vopd(w, g, in);
    default: return tier_global(w, g, in);
  }
}
