// PTX abstract syntax — the minimal subset VirtualGPU implements.
//
// This is intentionally NOT the full PTX ISA. Instructions are added
// incrementally, driven by tests; anything unrecognized is rejected at module
// load with Err::UnsupportedPtx naming the instruction, line, and kernel
// (never silently misbehaving).
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

using Operand = std::variant<RegOperand, ImmInt, ImmFloatBits, SregOperand>;

// A memory operand: [base + offset], base is a register or a kernel parameter.
struct Addr {
  bool base_is_param = false;
  std::string base;  // register name or parameter name
  int64_t offset = 0;
};

enum class Space { Param, Global, Shared, Local, Generic };

enum class IntBinOp { Add, Sub, Mul, Min, Max, Div, Rem, And, Or, Xor, Shl, Shr };
enum class FloatBinOp { Add, Sub, Mul, Min, Max, Div };
enum class CmpOp { Eq, Ne, Lt, Le, Gt, Ge };

struct OpLd { Space space; Type ty; std::string dst; Addr addr; };
struct OpSt { Space space; Type ty; Addr addr; Operand src; };
struct OpMov { Type ty; std::string dst; Operand src; };
struct OpCvtaToGlobal { Type ty; std::string dst; Operand src; };
struct OpIntBin { IntBinOp op; Type ty; std::string dst; Operand a, b; };
struct OpMadLo { Type ty; std::string dst; Operand a, b, c; };
struct OpMulWide { bool is_signed; std::string dst; Operand a, b; };  // 32x32 -> 64
struct OpFloatBin { FloatBinOp op; Type ty; std::string dst; Operand a, b; };
struct OpFma { Type ty; std::string dst; Operand a, b, c; };
struct OpSetp { CmpOp cmp; Type ty; std::string dst; Operand a, b; };
struct OpSelp { Type ty; std::string dst; Operand a, b; std::string pred; };
struct OpBra { size_t target; std::string label; };  // target = instruction index
struct OpBar {};                                     // bar.sync 0
struct OpRet {};

using Op = std::variant<OpLd, OpSt, OpMov, OpCvtaToGlobal, OpIntBin, OpMadLo, OpMulWide, OpFloatBin,
                        OpFma, OpSetp, OpSelp, OpBra, OpBar, OpRet>;

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
};

struct EntryFn {
  std::string name;
  std::vector<ParamDecl> params;
  std::vector<Instr> body;
  std::map<std::string, Type> reg_decls;  // declared virtual registers
};

struct Module {
  std::string version;       // ".version 8.3"
  std::string target;        // ".target sm_90"
  uint32_t address_size = 64;
  std::vector<EntryFn> entries;

  const EntryFn* find_entry(const std::string& name) const {
    for (const auto& e : entries)
      if (e.name == name) return &e;
    return nullptr;
  }
};

}  // namespace vgpu::ptx
