// PTX abstract syntax — the subset VirtualGPU implements.
//
// This is intentionally NOT the full PTX ISA. Instructions are added
// incrementally, driven by real kernels; anything unrecognized is rejected at
// module load with Err::UnsupportedPtx naming the instruction, line, and
// kernel (never silently misbehaving).
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace vgpu::ptx {

// Scalar type of an operand/instruction, e.g. ".s32" -> {Kind::S, 32}.
struct Type {
  // BF is bfloat16: the same exponent range as f32 with a 7-bit mantissa,
  // so it is a distinct kind rather than an f16 with different bits.
  enum class Kind { B, U, S, F, BF, Pred };
  Kind kind = Kind::B;
  uint32_t bits = 32;

  uint32_t bytes() const { return bits / 8; }
  bool is_float() const { return kind == Kind::F; }
  bool is_bfloat() const { return kind == Kind::BF; }
  // Any real-valued type, for code paths that treat them alike.
  bool is_real() const { return kind == Kind::F || kind == Kind::BF; }
  bool is_signed() const { return kind == Kind::S; }
  std::string str() const;
};

// Special (read-only) registers.
enum class Sreg : uint8_t {
  TidX, TidY, TidZ,
  NtidX, NtidY, NtidZ,
  CtaidX, CtaidY, CtaidZ,
  NctaidX, NctaidY, NctaidZ,
  LaneId,
  // The lane-mask family: a 32-bit mask of the lanes whose id compares the
  // given way with this lane's. Warp-aggregated algorithms are built on these
  // -- CUB's radix sort ranks a lane among its peers with
  // popc(%lanemask_le & match_mask), and a wrong value there is an address, not
  // a number.
  LaneMaskEq, LaneMaskLt, LaneMaskLe, LaneMaskGt, LaneMaskGe,
  // One warp per 32 lanes of the block, in this engine's scheduling.
  WarpId, NWarpId,
  // %envreg<32>: driver-set registers. PTX says they read as zero unless a
  // driver has set them, and nothing here sets them -- so zero is the value,
  // not a stand-in for one. ggml's soft_max reads a pair of them, splices them
  // into a 64-bit value and branches on whether it is zero, which is precisely
  // the "not set" path.
  EnvReg,
};

// A virtual register reference. `id` is a dense per-kernel index assigned at
// parse time so the interpreter can index a flat register file instead of
// hashing a name on every operand read -- the difference between ~25M and
// ~100M+ warp-instructions per second. `name` is kept for diagnostics.
inline constexpr uint32_t kNoReg = 0xFFFFFFFFu;
struct Reg {
  std::string name;
  uint32_t id = kNoReg;
  // True for registers declared wider than 32 bits. The interpreter keeps two
  // register files -- 32-bit and 64-bit lanes -- so the common narrow case
  // moves half as much memory per instruction. `id` indexes whichever file
  // this flag selects.
  bool wide = false;
};

// Compare and print a Reg by name, so diagnostics and tests read naturally.
inline bool operator==(const Reg& r, const std::string& s) { return r.name == s; }
inline bool operator==(const Reg& r, const char* s) { return r.name == s; }
inline bool operator==(const Reg& a, const Reg& b) { return a.id == b.id; }
inline std::ostream& operator<<(std::ostream& os, const Reg& r) { return os << r.name; }

struct RegOperand { Reg reg; };                     // %r1, %rd4, %f2, %p1 ...
struct ImmInt { int64_t value = 0; };                   // 42, -1, 0x10
struct ImmFloatBits { uint64_t bits = 0; uint32_t width = 0; };  // 0f3F800000 / 0d...
// `index` is meaningful only for %envregN, where the number selects one of a
// bank of driver-supplied words rather than naming a distinct quantity.
struct SregOperand { Sreg reg = Sreg::TidX; uint32_t index = 0; };
// A bare identifier naming a module .global variable or a function-local
// depot ("mov.u64 %rd, $str;" / "mov.u64 %SPL, __local_depot0;").
struct SymbolOperand { std::string name; };

using Operand = std::variant<RegOperand, ImmInt, ImmFloatBits, SregOperand, SymbolOperand>;

// A memory operand: [base + offset]. The base may be a register, a kernel
// parameter, a call-argument slot, or a variable named directly -- PTX allows
// `st.shared.u32 [myvar+8], %r` without first materializing an address.
struct Addr {
  enum class Base { Reg, EntryParam, CallSlot, Symbol };
  Base base_kind = Base::Reg;
  std::string base;
  uint32_t base_id = kNoReg;  // interned when base_kind == Reg
  bool base_wide = false;     // which register file base_id indexes
  int64_t offset = 0;
};

enum class Space { Param, Global, Shared, Local, Generic };

enum class IntBinOp { Add, Sub, Mul, Min, Max, Div, Rem, And, Or, Xor, Shl, Shr };
enum class FloatBinOp { Add, Sub, Mul, Min, Max, Div };
// Ordered comparisons plus the float unordered/NaN-aware forms. The "u"
// variants are true when either operand is NaN; Num/Nan test NaN-ness only.
enum class CmpOp { Eq, Ne, Lt, Le, Gt, Ge, Equ, Neu, Ltu, Leu, Gtu, Geu, Num, Nan };
enum class AtomOp { Add, Min, Max, And, Or, Xor, Exch, Cas };
enum class PredBinOp { And, Or, Xor };

// Vector loads/stores (v2/v4) carry 2 or 4 registers; scalar ops carry 1.
struct OpLd { Space space = Space::Generic; Type ty; std::vector<Reg> dsts; Addr addr; };
struct OpSt { Space space = Space::Generic; Type ty; Addr addr; std::vector<Operand> srcs; };
struct OpMov { Type ty; Reg dst; Operand src; };
// Vector forms of mov used by inline asm to pack/unpack sub-word registers:
//   mov.b32 %r, {%rs1, %rs2};      pack two 16-bit halves into 32 bits
//   mov.b32 {%rs1, %rs2}, %r;      unpack
struct OpMovPack { Type ty; Reg dst; std::vector<Operand> srcs; };
struct OpMovUnpack { Type ty; std::vector<Reg> dsts; Operand src; };
// Address-space conversion. PTX addresses in .shared/.local are offsets within
// that space's window; cvta converts them to/from generic addresses.
//   cvta.<space>.u64    d, a   -> generic  (d = window_base + a)
//   cvta.to.<space>.uNN d, a   -> space    (d = a - window_base)
// .global/.const already alias the generic space, so those are identity.
struct OpCvta { Type ty; Space space = Space::Generic; bool to_space = false; Reg dst; Operand src; };
// The bare modes (.rn/.rz/.rm/.rp) pick how a value is rounded into a narrower
// float. The "i" modes (.rni/.rzi/.rmi/.rpi) instead round to an integral
// value while keeping the float type -- they are what ceilf, floorf, truncf and
// roundf compile to, so conflating them with the bare modes leaves those
// intrinsics returning their input.
enum class Round { None, Rn, Rz, Rm, Rp, Rni, Rzi, Rmi, Rpi };
struct OpCvt { Type dst_ty; Type src_ty; Round round = Round::None; Reg dst; Operand src; };
struct OpNot { Type ty; Reg dst; Operand src; };   // bitwise not
struct OpNeg { Type ty; Reg dst; Operand src; };   // arithmetic negate (int/float)
struct OpAbs { Type ty; Reg dst; Operand src; };

// Single-operand math: the SFU-approximated transcendentals plus sqrt/rcp.
// VirtualGPU computes them at full host precision; results are within the
// documented approximation tolerance but not bit-identical to a real SFU
// (a documented divergence — see ARCHITECTURE.md).
enum class MathOp { Ex2, Lg2, Sin, Cos, Sqrt, Rsqrt, Rcp, Tanh };
struct OpMath { MathOp op; Type ty; Reg dst; Operand src; };

// Bitfield extract/insert.
struct OpBfe { Type ty; Reg dst; Operand a, b, c; };        // b=start, c=len
struct OpBfi { Type ty; Reg dst; Operand a, b, c, d; };     // insert a into b
struct OpBrev { Type ty; Reg dst; Operand src; };           // bit reverse
struct OpPopcClz { bool popc = false; Type ty; Reg dst; Operand src; };

// Warp shuffle. `pred_dst` is the optional "d|p" second destination.
enum class ShflMode { Up, Down, Bfly, Idx };
struct OpShfl { ShflMode mode; Reg dst; Reg pred_dst; Operand a, b, c, member_mask; };
// Warp vote/ballot across the active mask.
enum class VoteMode { All, Any, Uni, Ballot };
struct OpVote { VoteMode mode = VoteMode::All; bool ballot = false; Reg dst; Reg src; bool negate_src = false; };
// redux.sync.<op>.<type> d, a, membermask -- reduce a across the participating
// lanes of the warp and give every one of them the result.
enum class ReduxOp { Add, Min, Max, And, Or, Xor };
struct OpRedux { ReduxOp op = ReduxOp::Add; Type ty; Reg dst; Operand src; };
// cvt.rn.f16x2.f32 d, a, b -- convert two f32 and pack them into one register,
// a in the high half and b in the low half.
struct OpCvtF16x2 { Reg dst; Operand a, b; bool bf16 = false; };

// ldmatrix.sync.aligned.m8n8.xN[.trans].b16 {d...}, [addr]
// Loads N 8x8 matrices of 16-bit elements from shared memory. Row r of matrix i
// is at the address supplied by lane i*8+r, and each lane comes away with two
// consecutive elements of one row -- the layout an mma fragment expects.
// movmatrix.sync.aligned.m8n8.trans.b16 d, a
// Transposes an 8x8 matrix of 16-bit elements that the warp already holds in
// registers -- the same fragment layout ldmatrix produces, so this is the
// register-only counterpart to ldmatrix's .trans: it costs no shared memory
// round trip. Flash attention uses it to feed K^T to the second mma.
struct OpMovMatrix { Reg dst; Operand src; };

// ldmatrix.sync.aligned.m8n8.xN[.trans][.shared].b16
// When ".shared" is named, the address register holds an offset in the shared
// window; without it the register holds a generic address that cvta has already
// converted. Both forms occur: llama.cpp cvta's first, Triton does not.
struct OpLdMatrix {
  uint32_t count = 1;
  bool trans = false;
  bool shared_space = false;
  std::vector<Reg> dsts;
  Addr addr;
};

// mma.sync.aligned.m16n8kK.row.col.<dtype>.<atype>.<btype>.<ctype>
// The warp-wide tensor-core multiply-accumulate. Distinct from wmma, which is
// the older whole-fragment API: this one names the exact shape and the
// registers each lane holds.
enum class MmaElem { F16, BF16, TF32, S8, U8 };
struct OpMma {
  uint32_t k = 16;          // m and n are fixed at 16 and 8 for every shape here
  MmaElem ab_type = MmaElem::F16;
  bool ab_signed = true;    // for the integer types
  bool acc_f16 = false;     // accumulate in f16x2 registers rather than f32
  bool acc_int = false;     // s32 accumulate
  std::vector<Reg> d, a, b, c;
};
// mov.pred d, {0|1|%p} -- set a predicate from an immediate or copy another.
// Predicates live in their own register file, so this cannot go through the
// ordinary mov path that writes a 32/64-bit value.
struct OpMovPred { Reg dst; Operand src; };
struct OpPrmt { Reg dst; Operand a, b, c; };       // byte permute (default mode)
// copysign.f32/f64 d, a, b -- magnitude of b with the sign of a.
struct OpCopysign { Type ty; Reg dst; Operand a, b; };
// dp4a.{u32,s32}.{u32,s32} d, a, b, c -- four byte-wise products of a and b
// accumulated into c. Quantized inference leans on this heavily.
struct OpDp4a { bool a_signed = false; bool b_signed = false; Reg dst; Operand a, b, c; };
// bmsk.{clamp,wrap}.b32 d, a, b -- a contiguous mask of b bits starting at a.
struct OpBmsk { bool wrap = false; Reg dst; Operand a, b; };
// Extended-precision arithmetic. PTX has a single per-thread condition-code
// carry bit: ".cc" writes it, and the "addc"/"subc"/"madc" opcodes read it.
// Compilers chain these to synthesise wider-than-native adds -- Numba builds
// 64-bit index arithmetic out of them.
struct OpIntBin {
  IntBinOp op = IntBinOp::Add;
  Type ty;
  Reg dst;
  Operand a, b;
  bool carry_in = false;   // addc/subc: add the carry bit into the result
  bool carry_out = false;  // .cc: leave the carry-out in the condition code
};
struct OpMadLo {
  Type ty;
  Reg dst;
  Operand a, b, c;
  bool carry_in = false;
  bool carry_out = false;
};
struct OpMulWide { uint32_t src_bits = 32; bool is_signed = false; Reg dst; Operand a, b; };  // 32x32 -> 64
struct OpMadWide { bool is_signed = false; Reg dst; Operand a, b, c; };  // 32x32+64 -> 64
// High half of a same-width multiply. Compilers emit these to turn integer
// division by a constant into a multiply, so they show up in ordinary code.
struct OpMulHi { Type ty; Reg dst; Operand a, b; };
struct OpMadHi { Type ty; Reg dst; Operand a, b, c; bool carry_in = false; bool carry_out = false; };
struct OpShf { bool left = false; bool wrap = false; Reg dst; Operand a, b, c; };  // funnel shift b:a
// PTX names an explicit rounding mode on float arithmetic. Unlike .approx,
// which only relaxes accuracy, these change the result -- quantization kernels
// depend on .rz truncating -- so they are carried through and applied.
enum class FRound { Nearest, Zero, MinusInf, PlusInf };
struct OpFloatBin { FRound round = FRound::Nearest; FloatBinOp op = FloatBinOp::Add; Type ty; Reg dst; Operand a, b; };
struct OpFma { Type ty; Reg dst; Operand a, b, c; };
// Packed half2 SIMD: one 32-bit register holds two f16 lanes.
struct OpF16x2Bin { FloatBinOp op = FloatBinOp::Add; Reg dst; Operand a, b; };
struct OpF16x2Fma { Reg dst; Operand a, b, c; };
struct OpF16x2Neg { Reg dst; Operand src; };

// Tensor-core MMA (m16n16k16, f16 inputs, f32 accumulate). A warp-collective
// operation: the 32 lanes jointly hold the matrices.
//
// NOTE: PTX deliberately leaves the mapping of matrix elements to fragment
// registers UNSPECIFIED. VirtualGPU therefore defines its own self-consistent
// layout (see interpreter.cpp). Kernels that use wmma as an opaque
// load->mma->store pipeline, or that fill fragments uniformly, get
// hardware-matching results; kernels that depend on NVIDIA's exact
// undocumented element distribution may differ. Documented in ARCHITECTURE.md.
enum class MatLayout { Row, Col };
struct OpWmmaMma {
  MatLayout alayout, blayout;
  std::vector<Reg> d, a, b, c;
};
struct OpWmmaStore {
  MatLayout layout;
  Space space = Space::Generic;
  Addr addr;
  std::vector<Operand> src;
  Operand stride;
};
struct OpSetp { CmpOp cmp = CmpOp::Eq; Type ty; Reg dst; Operand a, b; };
struct OpSelp { Type ty; Reg dst; Operand a, b; Reg pred; };
struct OpPredBin { PredBinOp op = PredBinOp::And; Reg dst; Reg a, b; };
struct OpNotPred { Reg dst; Reg src; };
struct OpAtom { AtomOp op = AtomOp::Add; Space space = Space::Generic; Type ty; Reg dst; Addr addr; Operand b; Operand c; };
struct OpBra { size_t target = 0; std::string label; };  // target = instruction index
struct OpBar {};                                     // bar.sync 0
// An instruction with nothing to do here: a memory fence, or a backoff hint.
// Distinct from OpBar because a fence is *not* a barrier -- mapping membar onto
// bar.sync made every fence wait for the whole block, which a kernel that
// fences on one warp's path would have hung on.
struct OpNop {};
// activemask.b32 d -- the mask of lanes of this warp currently executing. Warp
// algorithms use it as the membership for a following .sync operation.
struct OpActiveMask { Reg dst; };
// trap aborts the launch. CUDA reports it as an unspecified launch failure,
// and a kernel that reaches it has detected something it cannot continue past,
// so it must not be silently skipped.
struct OpTrap {};

// Texture and surface access. The object is a 64-bit handle the host created,
// passed in as an ordinary kernel parameter; everything else about the fetch
// comes from the table it names (see exec/texture.hpp).
//
//   tex.<geom>.v4.<dtype>.<ctype>  {d0,d1,d2,d3}, [obj, {c0,...}]
//   suld.b.<geom>.<type>.<clamp>   {d0,...},      [obj, {x,y}]
//   sust.b.<geom>.<type>.<clamp>   [obj, {x,y}],  {s0,...}
//
// tex always writes four components even when the caller wants one -- the
// widest form is what ptxas emits regardless.
struct OpTex {
  uint32_t dims = 1;             // 1, 2 or 3
  Type dtype;                    // destination component type (f32, s32, u32)
  Type ctype;                    // coordinate type: f32 for sampled, s32 for fetch
  std::vector<Reg> dsts;         // always four
  Operand obj;                   // the texture object handle
  std::vector<Operand> coords;
};
struct OpSuld {
  uint32_t dims = 1;
  uint32_t bytes = 4;            // per component, from .b8/.b16/.b32/.b64
  std::vector<Reg> dsts;
  Operand obj;
  std::vector<Operand> coords;   // x is a *byte* offset, y and z are texel rows
};
struct OpSust {
  uint32_t dims = 1;
  uint32_t bytes = 4;
  Operand obj;
  std::vector<Operand> coords;
  std::vector<Operand> srcs;
};
// bar.red.{and,or}.pred d, 0, p  /  bar.red.popc.u32 d, 0, p
// A barrier that also reduces a predicate across every thread in the block and
// gives all of them the result. Unlike bar.sync it produces a value, so it
// cannot complete until every warp has arrived.
enum class BarRedOp { And, Or, Popc };
struct OpBarRed { BarRedOp op = BarRedOp::And; Reg dst; Reg src; bool negate_src = false; };
struct OpRet {};
// Call-sequence machinery (currently only the vprintf builtin is callable).
struct OpDeclSlot { std::string name; uint32_t size = 0; };            // ".param .b64 param0;" in body
struct OpStSlot { std::string slot; int64_t offset = 0; Type ty; Operand src; };
// cp.async.{ca,cg}.shared.global [dst], [src], cp-size{, src-size};
//
// A copy from global to shared that the *thread* does not wait on: it is
// issued, batched into a group with cp.async.commit_group, and awaited later
// with cp.async.wait_group N (at most N groups still outstanding) or
// cp.async.wait_all. This is what lets a tiled kernel fetch the next tile
// while computing on the current one, and it is the backbone of every modern
// attention and GEMM kernel.
//
// `src_size` is optional and may be smaller than the copy: the bytes past it
// are zero-filled rather than read, which is how kernels handle a tile that
// runs off the end of a tensor without a branch.
struct OpCpAsync {
  uint32_t bytes = 16;                 // 4, 8 or 16
  Addr dst;                            // shared
  Addr src;                            // global
  bool have_src_size = false;
  Operand src_size;                    // bytes actually read; the rest is zeroed
};

// The group operations. These carry no data: what they do is order the copies
// above against the reads that consume them.
struct OpCpAsyncGroup {
  enum class Kind { Commit, WaitGroup, WaitAll } kind = Kind::Commit;
  uint32_t keep = 0;                   // wait_group N: leave at most N outstanding
};

struct OpLdSlot { std::string slot; int64_t offset = 0; Type ty; Reg dst; };
struct OpCall { std::string callee; std::string retval_slot; std::vector<std::string> param_slots; };

using Op = std::variant<OpLd, OpSt, OpMov, OpMovPack, OpMovUnpack, OpCvta, OpCvt, OpNot, OpNeg, OpAbs, OpMath, OpBfe, OpBfi,
                        OpBrev, OpPopcClz, OpShfl, OpVote, OpPrmt, OpCopysign, OpDp4a, OpBmsk, OpTrap, OpTex, OpSuld, OpSust, OpBarRed, OpMovPred, OpRedux, OpCvtF16x2, OpLdMatrix, OpMma, OpIntBin, OpMadLo, OpMulWide, OpMadWide, OpMulHi, OpMadHi, OpShf,
                        OpFloatBin, OpFma, OpF16x2Bin, OpF16x2Fma, OpF16x2Neg, OpWmmaMma, OpWmmaStore, OpSetp, OpSelp, OpPredBin, OpNotPred, OpAtom, OpBra, OpBar,
                        OpRet, OpDeclSlot, OpStSlot, OpLdSlot, OpCall, OpCpAsync, OpCpAsyncGroup, OpMovMatrix, OpNop, OpActiveMask>;

struct Instr {
  size_t line = 0;                 // source line, for diagnostics
  bool has_pred = false;           // @%p / @!%p guard
  bool pred_negated = false;
  Reg pred;
  Op op;
  std::string text;                // original source text, for diagnostics
};

struct ParamDecl {
  std::string name;
  Type ty;
  uint32_t size = 0;   // bytes; scalars: ty.bytes(), aggregates: array size
  uint32_t align = 0;  // 0 = natural (size for scalars)
};

// A function-local .local depot (per-thread stack memory).
struct LocalDecl {
  std::string name;
  uint32_t size = 0;
  uint32_t align = 8;
  uint32_t offset = 0;  // within the per-thread local frame
};

// A .shared variable (per-block memory). `dynamic` marks the
// "extern .shared .b8 name[]" form whose size comes from the launch config.
struct SharedDecl {
  std::string name;
  uint32_t size = 0;
  uint32_t align = 8;
  uint32_t offset = 0;  // within the per-block shared frame
  bool dynamic = false;
};

struct EntryFn {
  std::string name;
  std::vector<ParamDecl> params;
  std::vector<Instr> body;
  std::map<std::string, Type> reg_decls;      // declared virtual registers
  // Dense register numbering used by the interpreter's flat register file.
  std::map<std::string, uint32_t> reg_ids;
  std::map<std::string, bool> reg_wide;   // name -> lives in the 64-bit file
  uint32_t num_regs = 0;                  // ids are dense within each file
  uint32_t num_regs32 = 0;
  uint32_t num_regs64 = 0;
  std::map<std::string, LocalDecl> locals;    // .local depots
  uint32_t local_frame_size = 0;              // total per-thread local bytes
  // Launch bounds from __launch_bounds__. `.maxntid` caps the block size the
  // kernel was compiled for and `.reqntid` fixes it exactly; exceeding either
  // is a launch failure on hardware, so both are enforced.
  std::array<uint32_t, 3> max_ntid{0, 0, 0};
  std::array<uint32_t, 3> req_ntid{0, 0, 0};
  uint32_t min_ctas_per_sm = 0;
  // Memoized register analysis. Held here rather than in a pointer-keyed
  // side table: a freed module's address can be reused by the next one, and
  // such a cache then hands back another kernel's register count.
  mutable bool regs_analyzed = false;
  mutable uint32_t cached_regs_per_thread = 0;
  mutable uint32_t cached_pred_regs = 0;
  mutable uint32_t cached_peak_live = 0;
  std::map<std::string, SharedDecl> shared;   // .shared variables (per block)
  uint32_t static_shared_size = 0;            // statically declared shared bytes
  bool uses_dynamic_shared = false;
};

// A module-scope .global/.const variable, materialized into device memory at
// module load.
struct GlobalVar {
  std::string name;
  uint32_t align = 1;
  uint64_t size = 0;
  std::vector<uint8_t> init;  // empty or size bytes
  // A pointer-valued global can be initialised with another symbol's address
  // ("= my_array;"). The address is not known until the module is loaded, so
  // the name is carried here and resolved then.
  std::string init_symbol;
};

struct Module {
  std::string version;       // ".version 9.0"
  std::string target;        // ".target sm_86"
  uint32_t address_size = 64;
  std::vector<EntryFn> entries;
  std::vector<GlobalVar> globals;
  std::vector<SharedDecl> module_shared;  // module-scope .shared variables

  const EntryFn* find_entry(const std::string& name) const {
    for (const auto& e : entries)
      if (e.name == name) return &e;
    return nullptr;
  }
};

}  // namespace vgpu::ptx
