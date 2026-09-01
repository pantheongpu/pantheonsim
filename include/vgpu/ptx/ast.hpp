// PTX abstract syntax — the subset VirtualGPU implements.
//
// This is intentionally NOT the full PTX ISA. Instructions are added
// incrementally, driven by real kernels; anything unrecognized is rejected at
// module load with Err::UnsupportedPtx naming the instruction, line, and
// kernel (never silently misbehaving).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace vgpu::ptx {

// Scalar type of an operand/instruction, e.g. ".s32" -> {Kind::S, 32}.
struct Type {
  enum class Kind { B, U, S, F, Pred };
  Kind kind = Kind::B;
  uint32_t bits = 32;

  uint32_t bytes() const { return bits / 8; }
  bool is_float() const { return kind == Kind::F; }
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
};

struct RegOperand { std::string name; };            // %r1, %rd4, %f2, %p1 ...
struct ImmInt { int64_t value; };                   // 42, -1, 0x10
struct ImmFloatBits { uint64_t bits; uint32_t width; };  // 0f3F800000 / 0d...
struct SregOperand { Sreg reg; };
// A bare identifier naming a module .global variable or a function-local
// depot ("mov.u64 %rd, $str;" / "mov.u64 %SPL, __local_depot0;").
struct SymbolOperand { std::string name; };

using Operand = std::variant<RegOperand, ImmInt, ImmFloatBits, SregOperand, SymbolOperand>;

// A memory operand: [base + offset].
struct Addr {
  enum class Base { Reg, EntryParam, CallSlot };
  Base base_kind = Base::Reg;
  std::string base;  // register / parameter / call-slot name
  int64_t offset = 0;
};

enum class Space { Param, Global, Shared, Local, Generic };

enum class IntBinOp { Add, Sub, Mul, Min, Max, Div, Rem, And, Or, Xor, Shl, Shr };
enum class FloatBinOp { Add, Sub, Mul, Min, Max, Div };
enum class CmpOp { Eq, Ne, Lt, Le, Gt, Ge };
enum class AtomOp { Add, Min, Max, And, Or, Xor, Exch, Cas };
enum class PredBinOp { And, Or, Xor };

// Vector loads/stores (v2/v4) carry 2 or 4 registers; scalar ops carry 1.
struct OpLd { Space space; Type ty; std::vector<std::string> dsts; Addr addr; };
struct OpSt { Space space; Type ty; Addr addr; std::vector<Operand> srcs; };
struct OpMov { Type ty; std::string dst; Operand src; };
struct OpCvta { Type ty; std::string dst; Operand src; };  // all address spaces alias: identity
enum class Round { None, Rn, Rz, Rm, Rp, Rni, Rzi };
struct OpCvt { Type dst_ty; Type src_ty; Round round = Round::None; std::string dst; Operand src; };
struct OpNot { Type ty; std::string dst; Operand src; };   // bitwise not
struct OpNeg { Type ty; std::string dst; Operand src; };   // arithmetic negate (int/float)
struct OpPrmt { std::string dst; Operand a, b, c; };       // byte permute (default mode)
struct OpIntBin { IntBinOp op; Type ty; std::string dst; Operand a, b; };
struct OpMadLo { Type ty; std::string dst; Operand a, b, c; };
struct OpMulWide { bool is_signed; std::string dst; Operand a, b; };  // 32x32 -> 64
struct OpShf { bool left; bool wrap; std::string dst; Operand a, b, c; };  // funnel shift b:a
struct OpFloatBin { FloatBinOp op; Type ty; std::string dst; Operand a, b; };
struct OpFma { Type ty; std::string dst; Operand a, b, c; };
struct OpSetp { CmpOp cmp; Type ty; std::string dst; Operand a, b; };
struct OpSelp { Type ty; std::string dst; Operand a, b; std::string pred; };
struct OpPredBin { PredBinOp op; std::string dst; std::string a, b; };
struct OpNotPred { std::string dst; std::string src; };
struct OpAtom { AtomOp op; Type ty; std::string dst; Addr addr; Operand b; Operand c; };  // c: cas only
struct OpBra { size_t target; std::string label; };  // target = instruction index
struct OpBar {};                                     // bar.sync 0
struct OpRet {};
// Call-sequence machinery (currently only the vprintf builtin is callable).
struct OpDeclSlot { std::string name; uint32_t size; };            // ".param .b64 param0;" in body
struct OpStSlot { std::string slot; int64_t offset; Type ty; Operand src; };
struct OpLdSlot { std::string slot; int64_t offset; Type ty; std::string dst; };
struct OpCall { std::string callee; std::string retval_slot; std::vector<std::string> param_slots; };

using Op = std::variant<OpLd, OpSt, OpMov, OpCvta, OpCvt, OpNot, OpNeg, OpPrmt, OpIntBin, OpMadLo, OpMulWide, OpShf,
                        OpFloatBin, OpFma, OpSetp, OpSelp, OpPredBin, OpNotPred, OpAtom, OpBra, OpBar,
                        OpRet, OpDeclSlot, OpStSlot, OpLdSlot, OpCall>;

struct Instr {
  size_t line = 0;                 // source line, for diagnostics
  bool has_pred = false;           // @%p / @!%p guard
  bool pred_negated = false;
  std::string pred;
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

struct EntryFn {
  std::string name;
  std::vector<ParamDecl> params;
  std::vector<Instr> body;
  std::map<std::string, Type> reg_decls;      // declared virtual registers
  std::map<std::string, LocalDecl> locals;    // .local depots
  uint32_t local_frame_size = 0;              // total per-thread local bytes
};

// A module-scope .global/.const variable, materialized into device memory at
// module load.
struct GlobalVar {
  std::string name;
  uint32_t align = 1;
  uint64_t size = 0;
  std::vector<uint8_t> init;  // empty or size bytes
};

struct Module {
  std::string version;       // ".version 9.0"
  std::string target;        // ".target sm_86"
  uint32_t address_size = 64;
  std::vector<EntryFn> entries;
  std::vector<GlobalVar> globals;

  const EntryFn* find_entry(const std::string& name) const {
    for (const auto& e : entries)
      if (e.name == name) return &e;
    return nullptr;
  }
};

}  // namespace vgpu::ptx
