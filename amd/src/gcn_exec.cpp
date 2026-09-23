#include "vgpu/amd_exec.hpp"

#include <cstring>
#include <vector>

#include "vgpu/amd_gcn.hpp"
#include "vgpu/error.hpp"

namespace vgpu::amd {
namespace {

using gcn::Inst;
using gcn::Operand;
using gcn::OperandKind;

constexpr uint32_t kSgprs = 102;      // s0 through s101
constexpr uint32_t kVgprs = 256;
constexpr uint32_t kLanes = 64;

float as_float(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
uint32_t as_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

// One wavefront: its own scalar registers, VCC, EXEC and SCC, and 64 lanes of
// vector registers.
struct Wave {
  uint32_t sgpr[kSgprs] = {};
  uint32_t vgpr[kVgprs][kLanes] = {};
  uint64_t vcc = 0, exec = 0;
  bool scc = false;
  uint64_t pc = 0;
  bool done = false;
  bool at_barrier = false;
  uint32_t first_lane = 0;   // this wave's first work-item in the group
};

// The work-group the waves share: its LDS, and how many waves are still to
// reach the barrier.
struct Group {
  std::vector<uint8_t> lds;
  std::vector<Wave> waves;
};

struct Machine {
  const Dispatch& d;
  MemoryManager& mem;
  DispatchStats stats;

  // ---- Register access ----------------------------------------------------

  uint32_t sgpr(const Wave& w, uint32_t i) const {
    if (i >= kSgprs) throw Error::make(Err::Internal, "scalar register ", i, " is past the file");
    return w.sgpr[i];
  }
  void set_sgpr(Wave& w, uint32_t i, uint32_t v) {
    if (i >= kSgprs) throw Error::make(Err::Internal, "scalar register ", i, " is past the file");
    w.sgpr[i] = v;
  }
  uint64_t sgpr64(const Wave& w, uint32_t i) const { return sgpr(w, i) | static_cast<uint64_t>(sgpr(w, i + 1)) << 32; }
  void set_sgpr64(Wave& w, uint32_t i, uint64_t v) {
    set_sgpr(w, i, static_cast<uint32_t>(v));
    set_sgpr(w, i + 1, static_cast<uint32_t>(v >> 32));
  }

  // A scalar operand's value: a register, a special register, or a constant.
  uint64_t scalar(const Wave& w, const Operand& o) const {
    switch (o.kind) {
      case OperandKind::Sgpr: return o.width >= 2 ? sgpr64(w, o.index) : sgpr(w, o.index);
      case OperandKind::Vcc: return w.vcc;
      case OperandKind::Exec: return w.exec;
      case OperandKind::Inline:
      case OperandKind::Literal: return static_cast<uint64_t>(o.value);
      case OperandKind::M0: return 0;
      case OperandKind::Vgpr:
      case OperandKind::None: break;
    }
    throw Error::make(Err::Internal, "a scalar operand this does not read");
  }
  void write_scalar(Wave& w, const Operand& o, uint64_t v) {
    switch (o.kind) {
      case OperandKind::Sgpr:
        if (o.width >= 2) set_sgpr64(w, o.index, v);
        else set_sgpr(w, o.index, static_cast<uint32_t>(v));
        return;
      case OperandKind::Vcc: w.vcc = v; return;
      case OperandKind::Exec: w.exec = v; return;
      default: break;
    }
    throw Error::make(Err::Internal, "a scalar destination this does not write");
  }

  // A source as one lane sees it: a vector register's lane, or the same
  // scalar value for every lane.
  uint32_t lane_src(const Wave& w, const Operand& o, uint32_t lane) const {
    if (o.kind == OperandKind::Vgpr) return w.vgpr[o.index][lane];
    return static_cast<uint32_t>(scalar(w, o));
  }
  uint64_t lane_src64(const Wave& w, const Operand& o, uint32_t lane) const {
    if (o.kind == OperandKind::Vgpr)
      return w.vgpr[o.index][lane] | static_cast<uint64_t>(w.vgpr[o.index + 1][lane]) << 32;
    return scalar(w, o);
  }
  void write_lane(Wave& w, const Operand& o, uint32_t lane, uint32_t v) { w.vgpr[o.index][lane] = v; }
  void write_lane64(Wave& w, const Operand& o, uint32_t lane, uint64_t v) {
    w.vgpr[o.index][lane] = static_cast<uint32_t>(v);
    w.vgpr[o.index + 1][lane] = static_cast<uint32_t>(v >> 32);
  }

  // ---- The instructions ---------------------------------------------------

  void scalar_alu(Wave& w, const Inst& in) {
    const std::string& op = in.name;
    const uint64_t a = in.src.empty() ? 0 : scalar(w, in.src[0]);
    const uint64_t b = in.src.size() > 1 ? scalar(w, in.src[1]) : 0;
    if (op == "s_mov_b32" || op == "s_mov_b64") {
      write_scalar(w, in.dst[0], a);
    } else if (op == "s_movk_i32") {
      write_scalar(w, in.dst[0], static_cast<uint64_t>(static_cast<int64_t>(in.simm)));
    } else if (op == "s_add_u32") {
      const uint64_t sum = static_cast<uint32_t>(a) + static_cast<uint64_t>(static_cast<uint32_t>(b));
      write_scalar(w, in.dst[0], static_cast<uint32_t>(sum));
      w.scc = sum >> 32;                           // the carry out
    } else if (op == "s_addc_u32") {
      const uint64_t sum = static_cast<uint32_t>(a) + static_cast<uint64_t>(static_cast<uint32_t>(b)) + w.scc;
      write_scalar(w, in.dst[0], static_cast<uint32_t>(sum));
      w.scc = sum >> 32;
    } else if (op == "s_and_b64") {
      const uint64_t v = a & b;
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_or_b64") {
      const uint64_t v = a | b;
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_lshl_b64") {
      const uint64_t v = a << (b & 63);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_and_saveexec_b64") {
      // Divergence, as the compiler writes it: keep EXEC, narrow it to the
      // lanes the condition took.
      const uint64_t saved = w.exec;
      w.exec = a & saved;
      write_scalar(w, in.dst[0], saved);
      w.scc = w.exec != 0;
    } else {
      throw Error::make(Err::Unsupported, "scalar instruction ", op, " is decoded but not implemented");
    }
  }

  void scalar_load(Wave& w, const Inst& in) {
    const uint64_t base = scalar(w, in.src[0]) + static_cast<uint64_t>(in.offset);
    const uint32_t words = in.dst[0].width;
    for (uint32_t i = 0; i < words; ++i)
      set_sgpr(w, in.dst[0].index + i, static_cast<uint32_t>(mem.load_scalar(base + 4 * i, 4)));
  }

  void vector_alu(Wave& w, const Inst& in) {
    const std::string& op = in.name;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;   // EXEC says which lanes write
      if (op == "v_mov_b32_e32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane));
      } else if (op == "v_add_f32_e32") {
        write_lane(w, in.dst[0], lane,
                   as_bits(as_float(lane_src(w, in.src[0], lane)) + as_float(lane_src(w, in.src[1], lane))));
      } else if (op == "v_mul_f32_e32") {
        write_lane(w, in.dst[0], lane,
                   as_bits(as_float(lane_src(w, in.src[0], lane)) * as_float(lane_src(w, in.src[1], lane))));
      } else if (op == "v_lshlrev_b32_e32") {
        // The "rev" forms shift the second source by the first.
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[1], lane) << (lane_src(w, in.src[0], lane) & 31));
      } else if (op == "v_lshrrev_b32_e32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[1], lane) >> (lane_src(w, in.src[0], lane) & 31));
      } else if (op == "v_ashrrev_i32_e32") {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(static_cast<int32_t>(lane_src(w, in.src[1], lane)) >>
                                         (lane_src(w, in.src[0], lane) & 31)));
      } else if (op == "v_lshlrev_b64") {
        write_lane64(w, in.dst[0], lane, lane_src64(w, in.src[1], lane) << (lane_src(w, in.src[0], lane) & 63));
      } else if (op == "v_lshl_add_u32") {
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) << (lane_src(w, in.src[1], lane) & 31)) +
                       lane_src(w, in.src[2], lane));
      } else if (op == "v_lshl_add_u64") {
        write_lane64(w, in.dst[0], lane,
                     (lane_src64(w, in.src[0], lane) << (lane_src(w, in.src[1], lane) & 63)) +
                         lane_src64(w, in.src[2], lane));
      } else {
        throw Error::make(Err::Unsupported, "vector instruction ", op, " is decoded but not implemented");
      }
    }
  }

  void compare(Wave& w, const Inst& in) {
    const std::string& op = in.name;
    uint64_t result = 0;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;   // an inactive lane's bit reads 0
      const uint32_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
      bool set = false;
      if (op == "v_cmp_gt_i32_e32") set = static_cast<int32_t>(a) > static_cast<int32_t>(b);
      else if (op == "v_cmp_gt_u32_e32") set = a > b;
      else if (op == "v_cmp_eq_u32_e32") set = a == b;
      else throw Error::make(Err::Unsupported, "comparison ", op, " is decoded but not implemented");
      if (set) result |= uint64_t{1} << lane;
    }
    write_scalar(w, in.dst[0], result);
  }

  // LDS, which the work-group shares.
  void lds_access(Wave& w, const Inst& in, Group& g) {
    const std::string& op = in.name;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      const uint32_t addr = lane_src(w, in.src[0], lane);
      const auto at = [&](uint64_t offset) {
        const uint64_t a = addr + offset;
        if (a + 4 > g.lds.size())
          throw Error::make(Err::InvalidValue, "an LDS access at ", a, " is past the ", g.lds.size(),
                            " bytes the kernel reserved");
        return a;
      };
      if (op == "ds_write_b32") {
        const uint32_t v = lane_src(w, in.src[1], lane);
        std::memcpy(&g.lds[at(static_cast<uint64_t>(in.offset))], &v, 4);
      } else if (op == "ds_read_b32") {
        uint32_t v = 0;
        std::memcpy(&v, &g.lds[at(static_cast<uint64_t>(in.offset))], 4);
        write_lane(w, in.dst[0], lane, v);
      } else if (op == "ds_read2st64_b32") {
        // Two dwords, each offset by its own count of 64 dwords.
        uint32_t v0 = 0, v1 = 0;
        std::memcpy(&v0, &g.lds[at(uint64_t{static_cast<uint32_t>(in.offset)} * 64 * 4)], 4);
        std::memcpy(&v1, &g.lds[at(uint64_t{static_cast<uint32_t>(in.offset1)} * 64 * 4)], 4);
        write_lane(w, in.dst[0], lane, v0);
        w.vgpr[in.dst[0].index + 1][lane] = v1;
      } else {
        throw Error::make(Err::Unsupported, "LDS instruction ", op, " is decoded but not implemented");
      }
    }
  }

  void global_access(Wave& w, const Inst& in) {
    const std::string& op = in.name;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      // The address is a 64-bit one in a register pair, or a scalar base with
      // a 32-bit offset per lane.
      const uint64_t addr = (in.has_saddr ? sgpr64(w, in.saddr) + lane_src(w, in.src[0], lane)
                                          : lane_src64(w, in.src[0], lane)) +
                            static_cast<uint64_t>(static_cast<int64_t>(in.offset));
      if (op == "global_load_dword") {
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(mem.load_scalar(addr, 4)));
      } else if (op == "global_store_dword") {
        mem.store_scalar(addr, 4, lane_src(w, in.src[1], lane));
      } else {
        throw Error::make(Err::Unsupported, "memory instruction ", op, " is decoded but not implemented");
      }
    }
  }

  // Runs one instruction. Returns false when the wave has stopped or parked
  // at a barrier, so the group can run another wave.
  bool step(Wave& w, Group& g) {
    const CodeObject& o = *d.object;
    const Inst in = gcn::decode(o.text, w.pc - o.text_addr, w.pc);
    w.pc += in.size;
    ++stats.instructions;
    switch (in.enc) {
      case gcn::Enc::Sop1:
      case gcn::Enc::Sop2:
      case gcn::Enc::Sopk:
        scalar_alu(w, in);
        return true;
      case gcn::Enc::Smem:
        scalar_load(w, in);
        return true;
      case gcn::Enc::Vop1:
      case gcn::Enc::Vop2:
      case gcn::Enc::Vop3:
        vector_alu(w, in);
        return true;
      case gcn::Enc::Vopc:
        compare(w, in);
        return true;
      case gcn::Enc::Ds:
        lds_access(w, in, g);
        return true;
      case gcn::Enc::Flat:
        global_access(w, in);
        return true;
      case gcn::Enc::Sopp: break;
      default:
        throw Error::make(Err::Unsupported, gcn::enc_name(in.enc), " is decoded but not implemented");
    }
    // The program-flow instructions.
    if (in.name == "s_endpgm") {
      w.done = true;
      return false;
    }
    if (in.name == "s_nop" || in.name == "s_waitcnt") return true;   // nothing is out of order here
    if (in.name == "s_barrier") {
      w.at_barrier = true;
      ++stats.barriers;
      return false;
    }
    if (in.name == "s_branch") {
      w.pc = in.target;
      return true;
    }
    if (in.name == "s_cbranch_execz") {
      if (!w.exec) w.pc = in.target;
      return true;
    }
    if (in.name == "s_cbranch_execnz") {
      if (w.exec) w.pc = in.target;
      return true;
    }
    throw Error::make(Err::Unsupported, "instruction ", in.name, " is decoded but not implemented");
  }
};

}  // namespace

DispatchStats execute(const Dispatch& d, MemoryManager& mem) {
  if (!d.object || !d.kernel) throw Error::make(Err::InvalidValue, "a dispatch needs a kernel");
  const Kernel& k = *d.kernel;
  if (d.wave_size != kLanes)
    throw Error::make(Err::InvalidValue, "a CDNA wavefront is ", kLanes, " lanes, not ", d.wave_size);
  const uint64_t threads = uint64_t{d.group_size[0]} * d.group_size[1] * d.group_size[2];
  if (!threads) throw Error::make(Err::InvalidValue, "a work-group has no work-items");
  if (k.max_flat_workgroup_size && threads > k.max_flat_workgroup_size)
    throw Error::make(Err::InvalidValue, "a work-group of ", threads, " work-items is past the ",
                      k.max_flat_workgroup_size, " this kernel allows");

  Machine m{d, mem, {}};
  const uint32_t waves_per_group = static_cast<uint32_t>((threads + kLanes - 1) / kLanes);
  for (uint32_t gz = 0; gz < d.groups[2]; ++gz)
    for (uint32_t gy = 0; gy < d.groups[1]; ++gy)
      for (uint32_t gx = 0; gx < d.groups[0]; ++gx) {
        Group group;
        group.lds.assign(k.group_segment, 0);
        group.waves.resize(waves_per_group);
        for (uint32_t i = 0; i < waves_per_group; ++i) {
          Wave& w = group.waves[i];
          w.pc = k.entry;
          w.first_lane = i * kLanes;
          // The lanes this wave has of the work-group, which is short in the
          // last wave when the group is not a multiple of 64.
          const uint64_t left = threads - w.first_lane;
          w.exec = left >= kLanes ? ~uint64_t{0} : (uint64_t{1} << left) - 1;
          // What the hardware leaves in registers before the first
          // instruction: the user SGPRs the descriptor asked for, then the
          // work-group's id, and each lane's id in v0 (and v1, v2 where the
          // group has those dimensions).
          uint32_t at = 0;
          if (k.private_segment_buffer) at += 4;
          if (k.dispatch_ptr) at += 2;
          if (k.queue_ptr) at += 2;
          if (k.kernarg_segment_ptr) {
            m.set_sgpr64(w, at, d.kernarg);
            at += 2;
          }
          if (k.dispatch_id) at += 2;
          if (k.flat_scratch_init) at += 2;
          m.set_sgpr(w, at, gx);
          m.set_sgpr(w, at + 1, gy);
          m.set_sgpr(w, at + 2, gz);
          for (uint32_t lane = 0; lane < kLanes; ++lane) {
            const uint64_t flat = w.first_lane + lane;
            w.vgpr[0][lane] = static_cast<uint32_t>(flat % d.group_size[0]);
            w.vgpr[1][lane] = static_cast<uint32_t>(flat / d.group_size[0] % d.group_size[1]);
            w.vgpr[2][lane] = static_cast<uint32_t>(flat / d.group_size[0] / d.group_size[1]);
          }
          ++m.stats.waves;
        }

        // The group's waves run until every one has stopped. A wave parked at
        // a barrier waits for the others to reach it, as the hardware makes
        // it wait.
        for (bool working = true; working;) {
          working = false;
          for (Wave& w : group.waves) {
            if (w.done || w.at_barrier) continue;
            working = true;
            while (m.step(w, group)) {
            }
          }
          if (!working) {
            // Every wave is stopped or waiting: release the barrier.
            bool any = false;
            for (Wave& w : group.waves)
              if (w.at_barrier) {
                w.at_barrier = false;
                any = working = true;
              }
            if (!any) break;
          }
        }
      }
  return m.stats;
}

}  // namespace vgpu::amd
