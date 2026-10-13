// The pre-decoded execution tier of the SASS executor (docs/perf-design/01-whole-kernel-translation.md, route a).
//
// What it is. The interpreter (exec.cpp, exec_ops.inc) looks at an Instr's operand kinds, modifiers and form every
// time it executes the instruction, although the decoded Instr never changes. The tier decides those things once,
// when a code section is first run: for each instruction it builds a MOp (a micro-op), a pointer to a handler for
// the form plus the operands resolved to register-file rows, constant rows, or the few things that can only be
// read when the instruction runs (a constant-bank word, a uniform register, a register with a modifier).
// Runner::tier_run (tier_ops.inc) then runs the straight-line instructions of a warp's group through the array.
//
// What it is not. It does not change what happens, only how fast: the tier takes the place of
// Runner::execute for one instruction at a time, at the very place step_warp's straight-line loop calls it, so the
// warp scheduling (256-instruction chunks, the step count, the group selection, YIELD/BSYNC/spin handling) is the
// interpreter's own. An instruction the tier has no handler for -- or any condition it cannot prove safe (a lane
// mask forced by BMOV.PQUAD MACTIVE, a register past the warp's file) -- goes to Runner::execute, which is the
// existing code. Divergent warps work: a handler takes the executing lane mask like the interpreter's does.
//
// Off by default: VGPU_TIER=1 turns it on (docs/cli.md). Every handler is held to the bits of the interpreter's path
// for the same form (tests: nvidia/tests/e2e/run_sass_tier.sh, tests/unit/test_tier.cpp).
//
// This header is the data the analysis (tier.cpp: Instr -> MOp, no execution) and the handlers (tier_ops.inc,
// compiled into exec.cpp because they need the executor's private state) share.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "vgpu/sass/sass.hpp"

namespace vgpu::sass::tier {

// 32 lanes of constant, the shape a source row has in the register file.
struct alignas(32) Row {
  uint32_t v[32];
};

// How a source is found when the instruction runs.
enum class SrcKind : uint8_t {
  Row,    // a register's 32 lanes in the warp: R + off, off = register * 32
  Pool,   // a constant row of the program: pool[off] (an immediate, RZ, a folded modifier)
  Dyn,    // needs work at run time: see Dyn
};
enum class Dyn : uint8_t {
  Reg,     // a register with a modifier: off = register
  Bank0,   // a word of constant bank 0 (kernel parameters, launch shape): off = byte offset
  UReg,    // a uniform register: off = its number
};
// The modifiers a Dyn source carries, applied to its value as the form's domain says (apply()).
enum Mod : uint8_t { kNeg = 1, kNot = 2, kAbs = 4 };

struct Src {
  SrcKind kind = SrcKind::Pool;
  Dyn dyn = Dyn::Reg;
  uint8_t mod = 0;
  uint8_t pad = 0;
  uint32_t off = 0;
};

// Which handler runs an instruction. None: the interpreter does.
enum class Form : uint8_t {
  None,
  Mov,       // MOV Rd, src
  Iadd3,
  Iadd3x,    // IADD3 with carry-outs and/or carry-ins
  Imad,
  Imad32,    // IMAD with a carry-in or a carry-out
  ImadWide,  // IMAD.WIDE: a 64-bit result
  ImadHi,    // IMAD.HI
  IsetpEx,   // ISETP.EX
  Popc,
  I2fp,      // I2FP.F32 from a 32-bit integer
  Ldg,       // global loads and stores of 1, 2, 4 or 8 bytes
  Stg,
  Lds,       // shared memory
  Sts,
  Ldl,       // local memory
  Stl,
  Lop3,
  Lea,
  Shf,
  Isetp,
  Fsetp,
  Fsel,
  Ffma,
  Fadd,
  Fmul,
  Dop,       // DADD, DMUL, DFMA
  Count_,
};
const char* form_name(Form f);

struct MOp {
  // `ctx` is exec.cpp's TCtx: what a handler needs of the warp and the launch.
  using Fn = void (*)(void* ctx, const MOp&, uint32_t ex);
  Fn fn = nullptr;               // null: the interpreter executes it
  const Instr* ins = nullptr;    // for the interpreter, faults and statistics
  Src s[3];                      // sources, in the order the handler takes them
  Src p;                         // a predicate source (Dyn::Reg: a per-lane P<n>; Dyn::UReg: a uniform UP<n>), if any
  Src p2;                        // a second one
  int64_t imm = 0;               // a memory operand's offset
  uint32_t a = 0, b = 0;         // form-specific immediates (a LOP3's table, an IADD3's +1 count, a shift, ...)
  uint32_t c = 0;
  uint16_t dst = 0;              // destination register row, or kNoReg for RZ
  uint16_t dst2 = 0;             // second destination (a predicate number, or a register)
  uint16_t maxreg = 0;           // one past the highest register this touches (the warp's file must hold it)
  Form form = Form::None;
  uint8_t cls = 0;               // Instr::cls
  uint8_t guard = 7;             // @P<n>; 7 (PT) with guard_not clear is "always"
  bool guard_not = false;
  bool guard_uniform = false;
  bool unguarded = true;
  bool never = false;            // @!PT: the guard is false for every lane, so it does nothing but count
  bool straight = false;         // Instr::straight
  uint8_t aux = 0;               // form-specific flags
};
inline constexpr uint16_t kNoReg = 0xffff;

// The tier of one code section: a MOp per instruction, and the constant rows they point at.
struct Program {
  std::vector<MOp> ops;
  std::vector<Row> pool;         // pool[0] is all zero
  std::map<uint32_t, uint32_t> const_index;   // value -> pool row, while the program is built
  uint16_t max_reg = 0;          // the largest MOp::maxreg: a warp whose file holds this needs no per-op check
  size_t native = 0;             // instructions with a handler
};

// The analysis half. Fills in everything of `op` but `fn` (the executor binds handlers by `form`).
// `sm` is the architecture the code was decoded for. Never throws on an instruction it does not take: Form::None.
void analyze(const Instr& ins, int sm, Program& prog, MOp& op);

// Whether VGPU_TIER asks for the tier (read once per process). The executor also requires its fast paths
// (VGPU_FASTPATH) and the SIMD kernels (VGPU_SIMD) on, and no per-instruction trace.
bool requested();

}  // namespace vgpu::sass::tier
