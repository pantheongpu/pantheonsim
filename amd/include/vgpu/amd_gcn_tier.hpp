// The AMD interpreter's pre-decoded tier: the hottest instruction forms of the gfx9 (CDNA) and gfx10/11/12 (RDNA)
// instruction sets, resolved when they are decoded into a handler and operand classes, so that executing one is a
// handler call over register rows rather than the interpreter's name ladders, per-lane operand reads and per-lane
// EXEC tests (amd/src/gcn_tier.cpp; the NVIDIA engine's twin is src/sass/tier.cpp).
//
// The contract is the one the whole tier keeps: results, counters and the order of effects are those of the
// interpreter, bit for bit. A handler therefore either does the instruction completely or does nothing and says so
// (returns false), and the interpreter then runs the instruction as it always did. Anything unusual -- a modifier, a
// register the handler does not model, a NaN where the order of two NaN payloads would decide the bits -- is that
// second case. amd/tests/unit/test_gcn_tier_same_bits.cpp runs every instruction shape of the ISA corpora both ways.
//
// VGPU_AMD_TIER=0 turns the tier off (it is on by default); VGPU_AMD_TIER_STATS=1 lists what the interpreter ran anyway.
#pragma once

#include <cstdint>

namespace vgpu::amd::gcn {

namespace tier {

// The registers of a wave an instruction can reach, as pointers into it (so this header needs nothing of the
// interpreter's types). Built by the interpreter for each instruction.
struct Regs {
  uint32_t (*vgpr)[64];   // the vector register rows: vgpr[r][lane]
  uint32_t* sgpr;          // the scalar registers
  uint64_t* vcc;
  uint64_t* exec;
  uint64_t* pc;
  bool* scc;
  uint32_t lanes;          // 64, or 32 for a wave32 wave
  uint32_t* loads;         // Wave::loads_in_flight
  bool* soft_yield;        // Wave::soft_yield
  uint32_t nwaves;         // the waves in the work-group
  bool* store_gate;        // Wave::store_gate
  uint8_t* lds;            // the work-group's LDS
  uint64_t lds_size;
  uint64_t target;         // the instruction's branch target (Inst::target), as the interpreter has it
};

struct Op;
// A handler's answer: 0, it did nothing and the interpreter runs the instruction; 1, done; 2, done, and the wave gives up its turn
// (step() returns false); 3, not done and not counted, the wave gives up its turn and runs the instruction on its next one (a plain
// store's first of a round, Wave::store_gate).
using Fn = int (*)(const Op&, Regs&);
using AnyFn = void (*)();   // a lane kernel (its type is the handler's business)

// What an operand is, as a handler reads it.
enum SrcKind : uint8_t {
  kNone = 0,
  kVgpr,      // vector register `index`
  kConst,     // a constant: `value` (the 32 bits a lane sees), `value64` (what a 64-bit consumer sees)
  kSgpr,      // scalar register `index` (a 32-bit read of it; a pair read when `wide`)
  kVcc,       // VCC (wide: all 64 bits; else the low 32)
  kVccHi,
  kExec,      // EXEC
  kExecLo,
  kExecHi,
  kNull,      // RDNA's null register: reads zero, a write goes nowhere
};

struct Src {
  uint8_t kind = kNone;
  bool wide = false;   // the operand covers two registers (scalar reads give the pair; writes write both)
  bool neg = false, abs = false;
  bool h16 = false;    // the modifiers act on bit 15 (a half in the low 16 bits), not on bit 31
  uint8_t sel = 6;     // SDWA: the part of the 32 bits the instruction reads (0 to 3 a byte, 4 and 5 a half, 6 all)
  bool sext = false;   // ... and whether it is taken with its sign
  uint32_t index = 0;
  uint32_t value = 0;
  uint64_t value64 = 0;
};

// What the instruction counts as in DispatchStats::counts, besides the mix.
enum Unit : uint8_t { kUnitNone = 0, kUnitValu, kUnitSalu, kUnitBranch, kUnitLds };

struct Op {
  Fn run = nullptr;            // null: the interpreter's
  AnyFn kern = nullptr;        // the lane kernel the handler calls
  Src src[3];
  Src dst, dst2;
  uint8_t unit = kUnitNone;
  uint8_t omod = 0;
  uint8_t nan = 0;             // a float result that is a NaN: 0 the interpreter decides (the handler hands back), 1 the kernel's is the interpreter's, else
                               // the NaN lanes are worked out again by the rule the interpreter's instruction uses (nan_order.hpp; see gcn_tier.cpp's fix_nan)
  uint8_t dsel = 6, dunused = 0;   // SDWA: the part of the destination written (dst_sel), and what the rest becomes (0 zeroes, 1 the sign, 2 kept)
  bool clamp = false;
  uint32_t sub = 0;            // handler-specific (a comparison's test, a shift count, ...)
  int32_t off0 = 0, off1 = 0;  // a memory instruction's offsets
  uint8_t words = 0;           // ... and how many registers it moves
  uint8_t mach = 0;            // the Machine runs it: 1 a global load, 2 a global store (name work done at decode), 3 s_barrier, 6 s_endpgm, 7 scalar memory
  bool store = false;          // a plain store (it takes Wave::store_gate)
};

// Whether the tier is on (VGPU_AMD_TIER; set_enabled for tests, which must not call it while another thread runs a
// dispatch).
bool enabled();
void set_enabled(bool on);
// How many instructions a handler gave back to the interpreter since the process began (a NaN where it does not take the decision).
void count_bails(bool on);   // (for tests: off by default)
void note_bail();
uint64_t bails();
// VGPU_AMD_TIER_STATS=1: at exit, the instructions the interpreter ran (those with no handler, and those a handler handed back), by name.
bool stats();
void note_interp(const char* name, bool handed_back);

}  // namespace tier

}  // namespace vgpu::amd::gcn
