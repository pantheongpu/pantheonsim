// Throwaway prototype (r7-design, NOT for merging): a closure-compiled
// ("direct-threaded") executor for a restricted subset of SASS.
//
// Every decoded instruction becomes one MOp: a function pointer plus operand
// *row indices* into a per-warp array of 32-lane rows (registers, then the
// constants and special registers the kernel uses, so every source is just a
// row). Basic blocks are runs of MOps; a block starts with a header MOp that
// adds the block's precomputed instruction counts. The prototype follows the
// convergent case only: when a branch would split the warp it throws
// Unsupported (a real design would fall back to the reference engine's
// per-lane-pc scheduler for the warp).
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/sass/exec.hpp"

namespace r7 {

struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

using Mask = uint32_t;
constexpr Mask kAll = 0xffffffffu;

struct BlockInfo;
struct MOp;

struct Ctx {
  uint32_t* R = nullptr;     // rows of 32 lanes
  Mask P[8] = {0, 0, 0, 0, 0, 0, 0, kAll};
  Mask ex = 0;               // lanes executing (alive and not exited)
  bool done = false;
  vgpu::exec::LaunchStats* st = nullptr;
  uint64_t steps = 0, max_steps = 0;
  vgpu::MemoryManager* mem = nullptr;
  const BlockInfo* infos = nullptr;
  const MOp* ops = nullptr;
  const uint32_t* K = nullptr;       // launch constants (bank-0 words) for generated regions
};

using Fn = const MOp* (*)(Ctx&, const MOp&);

struct MOp {
  Fn fn = nullptr;
  Fn alt = nullptr;                 // guarded ops: the real handler
  uint32_t d = 0, a = 0, b = 0, c = 0, e = 0;   // row indices
  uint32_t imm = 0, imm2 = 0;
  uint8_t g = 7, gneg = 0, cls = 0, k0 = 0, k1 = 0, k2 = 0, k3 = 0;
  int32_t target = -1;              // branch target: index into the op array
  uint32_t pc = 0;                  // the SASS byte address, for diagnostics
  void* user = nullptr;             // a generated region function
};

struct BlockInfo {
  uint32_t n_total = 0;             // instructions in the block
  uint32_t n_free = 0;              // of them guard-free
  uint32_t cls_free[static_cast<size_t>(vgpu::exec::InstClass::Count)] = {};
};

struct Program {
  std::vector<MOp> ops;
  std::vector<BlockInfo> infos;
  uint32_t nregs = 0;               // register rows
  uint32_t nrows = 0;               // total rows
  std::vector<uint32_t> const_rows; // row i (from first_const) holds this value, broadcast
  std::vector<int32_t> const_k;     // per constant row: index into launch_consts when the value came from bank 0, else -1
  std::vector<uint32_t> launch_consts;
  uint32_t first_const = 0;
  uint32_t zero_row = 0, sink_row = 0, first_spec = 0;
  uint32_t row_tid_x = 0, row_laneid = 0;
  uint32_t first_ur = 0;            // uniform registers as broadcast rows: UR n is row first_ur + n
  std::vector<uint32_t> ctaid_rows; // CTAID.X/Y/Z broadcast rows (written per block)
  uint32_t row_tid_y = 0, row_tid_z = 0;
  double compile_seconds = 0;
  size_t n_sass = 0;
};

// Per-launch inputs the compiler folds in.
struct LaunchInfo {
  std::array<uint32_t, 3> grid{1, 1, 1}, block{1, 1, 1};
  std::vector<uint8_t> bank0;       // constant bank 0 as the runner builds it
  uint64_t max_steps = 1ull << 30;
};

Program compile(const vgpu::sass::Module& m, const vgpu::sass::CubinKernel& k, const LaunchInfo& li);
vgpu::exec::LaunchStats run(const Program& p, const LaunchInfo& li, vgpu::MemoryManager& mem);
// Bank 0 as the runner's build_bank0 fills it for sm_86 (fields the subset reads).
std::vector<uint8_t> make_bank0(const vgpu::sass::Module& m, const vgpu::sass::CubinKernel& k, const LaunchInfo& li,
                                const std::vector<std::vector<uint8_t>>& args);

enum class OpKind { Header, Guard, Mov, Iadd3, Iadd3Plain, Imad, ImadM, ImadWide, Lop3, Lop3Xor3, Shf, Lea, Isetp, Ffma, Fadd, Fmul, I2fp, Ldg, Stg, Bra, Exit, Region, Other };
OpKind kind_of(const MOp& op);
Fn region_handler();
// Generated code (codegen.cpp): C++ source for the program's straight-line ALU runs and the patched program.
// fused: one lane loop per run (values stay in registers); otherwise one loop per op with the row array
// round-tripped between ops (what a template JIT without register allocation produces).
struct Generated {
  std::string source;
  Program program;               // ops replaced by region ops (fn = region handler, user set by bind())
  std::vector<std::string> names;   // region function symbol for each region op, in op order
  std::vector<size_t> region_ops;   // indices into program.ops
  size_t regions = 0, fused_ops = 0;
};
Generated generate(const Program& p, bool fused, bool launch_as_params);
void bind(Generated& g, void* dl_handle);
bool supports(const vgpu::sass::Module& m, const vgpu::sass::CubinKernel& k, const vgpu::sass::Instr& ins, std::string* why);
void set_fast_counters(bool on);
void set_bulk_mem(bool on);

}  // namespace r7
